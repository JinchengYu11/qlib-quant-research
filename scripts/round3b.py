"""
第 3 轮 B：因子精简对比。
- 用真实因子名做重要性分析
- 剔除高相关冗余因子(>0.9，保留高重要性一方) + 零增益因子
- 全因子 vs 精简因子，均用 T2 策略(n_drop=1)回测对比
"""
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = Path("/Users/cedricyu/qlib量化研究/results/runs/round3")
OUT.mkdir(parents=True, exist_ok=True)
PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v2"
BT_START, BT_END = "2020-10-01", "2026-05-15"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)
LGB_PARAMS = dict(objective="l2", colsample_bytree=0.8879, learning_rate=0.0421,
                  subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
                  max_depth=8, num_leaves=210, num_threads=8, verbosity=-1)


def ann(x, n=252): return x.mean() * n
def vol(x, n=252): return x.std() * np.sqrt(n)
def sharpe(x): return ann(x) / vol(x) if vol(x) > 0 else float("nan")
def mdd(x):
    eq = (1 + x).cumprod()
    return (eq / eq.cummax() - 1).min()


def build_dataset():
    dh = dict(
        start_time="2010-01-04", end_time="2026-05-19",
        fit_start_time="2010-01-04", fit_end_time="2017-12-31",
        instruments="csi300",
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=["Ref($close, -2) / Ref($close, -1) - 1"],
    )
    handler = Alpha158(**dh)
    return DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })


def train_model(Xtr, ytr, Xva, yva, cols):
    dtr = lgb.Dataset(Xtr[cols], label=ytr)
    dva = lgb.Dataset(Xva[cols], label=yva)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return booster


def eval_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"]))
    ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    return ic.mean(), ric.mean()


def run_bt(pred):
    strat = TopkDropoutStrategy(signal=pred, topk=50, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time=BT_START, end_time=BT_END, strategy=strat, executor=ex,
                      benchmark="SH000300", account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd["1day"][0]
    rep.index = pd.to_datetime(rep.index)
    ret, bench, cost, turn = rep["return"], rep["bench"], rep["cost"], rep["turnover"]
    net = ret - cost
    ex_net = net - bench
    return dict(excess_net_ann=ann(ex_net) * 100, excess_ir=sharpe(ex_net),
                gross_excess_ann=ann(ret - bench) * 100, turnover=turn.mean() * 100,
                cost_ann=ann(cost) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100)


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    ds = build_dataset()

    print("[1] 准备数据 ...", flush=True)
    tr = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    va = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    te_feat = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    te_label = ds.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)

    Xtr, ytr = tr["feature"], tr["label"].iloc[:, 0]
    Xva, yva = va["feature"], va["label"].iloc[:, 0]
    cols_all = list(Xtr.columns)
    print(f"    全因子数: {len(cols_all)}", flush=True)

    # --- 全因子模型 ---
    print("[2] 训练全因子模型 ...", flush=True)
    bm_full = train_model(Xtr, ytr, Xva, yva, cols_all)
    imp = pd.Series(bm_full.feature_importance(importance_type="gain"), index=cols_all)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(OUT / "factor_importance_named.csv", header=["gain"])
    print("    Top10 因子:", flush=True)
    for n, v in imp.head(10).items():
        print(f"      {n:12s} {v:10.1f}", flush=True)
    zero = imp[imp == 0].index.tolist()
    print(f"    零增益因子 {len(zero)} 个: {zero}", flush=True)

    # --- 相关性去冗余 ---
    print("[3] 计算相关性，剔除冗余 ...", flush=True)
    corr = Xtr.corr().abs()
    drop = set(zero)
    pairs = []
    for i in range(len(cols_all)):
        for j in range(i + 1, len(cols_all)):
            c = corr.iloc[i, j]
            if c > 0.9:
                pairs.append((cols_all[i], cols_all[j], float(c)))
    # 高相关对里，丢重要性较低的一方
    for a, b, c in sorted(pairs, key=lambda x: -x[2]):
        if a in drop or b in drop:
            continue
        loser = a if imp.get(a, 0) < imp.get(b, 0) else b
        drop.add(loser)
    cols_keep = [c for c in cols_all if c not in drop]
    print(f"    冗余对 {len(pairs)} 对 → 剔除 {len(drop)} 因子，保留 {len(cols_keep)}", flush=True)
    json.dump({"dropped": sorted(drop), "kept": cols_keep},
              open(OUT / "factor_selection.json", "w"), indent=1, ensure_ascii=False)

    # --- 精简因子模型 ---
    print("[4] 训练精简因子模型 ...", flush=True)
    bm_trim = train_model(Xtr, ytr, Xva, yva, cols_keep)

    # --- 预测 + IC + 回测 ---
    print("[5] 预测 / IC / 回测对比 (均用 T2: n_drop=1, topk=50) ...", flush=True)
    results = {}
    for tag, bm, cols in [("全因子(158)", bm_full, cols_all), ("精简因子", bm_trim, cols_keep)]:
        pred = pd.Series(bm.predict(te_feat[cols]), index=te_feat.index)
        ic, ric = eval_ic(pred, te_label.iloc[:, 0])
        m = run_bt(pred)
        m["IC"], m["RankIC"], m["n_factor"] = ic, ric, len(cols)
        results[tag] = m
        print(f"  {tag}: 因子数={len(cols)} IC={ic:.4f} RankIC={ric:.4f} "
              f"净超额={m['excess_net_ann']:+.2f}% IR={m['excess_ir']:+.3f} "
              f"换手={m['turnover']:.1f}% 净夏普={m['net_sharpe']:.3f} 回撤={m['net_mdd']:.1f}%", flush=True)

    pd.DataFrame(results).T.to_csv(OUT / "factor_compare.csv")
    print("\n[DONE] 结果存 results/runs/round3/", flush=True)


if __name__ == "__main__":
    main()
