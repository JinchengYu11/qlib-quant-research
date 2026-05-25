"""
第 3 轮：降换手参数扫描 + 因子重要性/相关性分析。
训练一次 LightGBM，复用预测值跑多组 TopkDropoutStrategy 回测。
"""
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

PROJ = Path("/Users/cedricyu/qlib量化研究")
OUT = PROJ / "results" / "runs" / "round3"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v2"
BT_START, BT_END = "2020-10-01", "2026-05-15"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)


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
    dataset = DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })
    return dataset


def run_backtest(pred, topk, n_drop, hold_thresh):
    strategy = TopkDropoutStrategy(signal=pred, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pmd, _ = backtest(
        start_time=BT_START, end_time=BT_END,
        strategy=strategy, executor=executor,
        benchmark="SH000300", account=1e8, exchange_kwargs=EXCHANGE,
    )
    report = pmd["1day"][0]
    report.index = pd.to_datetime(report.index)
    ret, bench, cost, turn = report["return"], report["bench"], report["cost"], report["turnover"]
    net = ret - cost
    excess_net = net - bench
    return dict(
        net_ann=ann(net) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100,
        excess_net_ann=ann(excess_net) * 100, excess_ir=sharpe(excess_net),
        gross_excess_ann=ann(ret - bench) * 100,
        turnover=turn.mean() * 100, cost_ann=ann(cost) * 100,
    )


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    print("[1] 构建数据集 + 训练 LightGBM ...", flush=True)
    dataset = build_dataset()
    model = LGBModel(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421,
                     subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
                     max_depth=8, num_leaves=210, num_threads=8)
    model.fit(dataset)

    # 因子重要性
    booster = model.model
    feat_names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    fi = pd.DataFrame({"feature": feat_names, "gain": gain, "split": split})
    fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)
    fi.to_csv(OUT / "factor_importance.csv", index=False)
    print(f"[2] 因子重要性已存 ({len(fi)} 因子)。Top10:", flush=True)
    print(fi.head(10).to_string(index=False), flush=True)
    zero_gain = fi[fi["gain"] == 0]
    print(f"    零增益(完全没用)因子: {len(zero_gain)} 个", flush=True)

    # 因子相关性（用 test 段特征）
    test_feat = dataset.prepare("test", col_set="feature")
    corr = test_feat.corr().abs()
    redundant = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if c > 0.9:
                redundant.append((cols[i], cols[j], round(float(c), 3)))
    print(f"[3] 高相关因子对 (|corr|>0.9): {len(redundant)} 对", flush=True)
    with open(OUT / "redundant_pairs.json", "w") as f:
        json.dump(redundant, f, indent=1, ensure_ascii=False)

    # 预测
    print("[4] 生成测试集预测 ...", flush=True)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred.name = "score"

    # 降换手扫描
    configs = [
        ("baseline", dict(topk=50, n_drop=5, hold_thresh=1)),
        ("T1", dict(topk=50, n_drop=2, hold_thresh=1)),
        ("T2", dict(topk=50, n_drop=1, hold_thresh=1)),
        ("T3", dict(topk=50, n_drop=2, hold_thresh=5)),
        ("T4", dict(topk=50, n_drop=1, hold_thresh=10)),
        ("T5", dict(topk=30, n_drop=2, hold_thresh=5)),
    ]
    print("[5] 降换手参数扫描 ...", flush=True)
    rows = []
    for name, cfg in configs:
        m = run_backtest(pred, **cfg)
        m["name"] = name
        m.update(cfg)
        rows.append(m)
        print(f"  {name}: topk={cfg['topk']} n_drop={cfg['n_drop']} hold={cfg['hold_thresh']} "
              f"| 净超额 {m['excess_net_ann']:+.2f}% | IR {m['excess_ir']:+.3f} "
              f"| 换手 {m['turnover']:.1f}% | 成本 {m['cost_ann']:.1f}% | 净夏普 {m['net_sharpe']:.3f}", flush=True)

    res = pd.DataFrame(rows)[["name", "topk", "n_drop", "hold_thresh",
                              "excess_net_ann", "excess_ir", "gross_excess_ann",
                              "turnover", "cost_ann", "net_ann", "net_sharpe", "net_mdd"]]
    res.to_csv(OUT / "turnover_sweep.csv", index=False)
    print("\n[DONE] 结果已存 results/runs/round3/", flush=True)


if __name__ == "__main__":
    main()
