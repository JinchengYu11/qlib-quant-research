"""
第 4 轮：估值类基本面因子对比。
Alpha158 (158 量价) vs Alpha158Plus (158 量价 + 10 估值)，均用第 3 轮最优策略 (n_drop=1)。
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path("/Users/cedricyu/qlib量化研究")
sys.path.insert(0, str(PROJ / "factors"))

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from alpha158_plus import Alpha158Plus, VALUE_NAMES

OUT = PROJ / "results" / "runs" / "round4"
OUT.mkdir(parents=True, exist_ok=True)
PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v2"
BT_START, BT_END = "2020-10-01", "2026-05-15"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)

DH = dict(
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
LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)


def ann(x, n=252): return x.mean() * n
def vol(x, n=252): return x.std() * np.sqrt(n)
def sharpe(x): return ann(x) / vol(x) if vol(x) > 0 else float("nan")
def mdd(x):
    eq = (1 + x).cumprod()
    return (eq / eq.cummax() - 1).min()


def eval_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"]))
    ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    return ic.mean(), ric.mean(), ic.mean() / ic.std(), ric.mean() / ric.std()


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
    yearly = {int(y): float(g.sum() * 100) for y, g in ex_net.groupby(ex_net.index.year)}
    return dict(excess_net_ann=ann(ex_net) * 100, excess_ir=sharpe(ex_net),
                gross_excess_ann=ann(ret - bench) * 100, turnover=turn.mean() * 100,
                cost_ann=ann(cost) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100,
                yearly=yearly)


def run_handler(handler_cls, tag):
    print(f"\n=== {tag} ===", flush=True)
    handler = handler_cls(**DH)
    dataset = DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })
    model = LGBModel(**LGB)
    model.fit(dataset)

    # 因子名（顺序）
    feat_cols = list(dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I).columns)
    booster = model.model
    gain = booster.feature_importance(importance_type="gain")
    imp = pd.Series(gain, index=feat_cols).sort_values(ascending=False)

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
    ic, ric, icir, ricir = eval_ic(pred, label)
    m = run_bt(pred)
    m.update(dict(IC=ic, RankIC=ric, ICIR=icir, RankICIR=ricir, n_factor=len(feat_cols)))
    print(f"  因子数={len(feat_cols)}  IC={ic:.4f}  RankIC={ric:.4f}  ICIR={icir:.3f}", flush=True)
    print(f"  净超额={m['excess_net_ann']:+.2f}%  IR={m['excess_ir']:+.3f}  "
          f"换手={m['turnover']:.1f}%  净夏普={m['net_sharpe']:.3f}  回撤={m['net_mdd']:.1f}%", flush=True)
    return m, imp


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    res = {}
    m158, imp158 = run_handler(Alpha158, "Alpha158 (基线 158 量价)")
    res["Alpha158"] = m158
    mplus, impplus = run_handler(Alpha158Plus, "Alpha158Plus (158 量价 + 10 估值)")
    res["Alpha158Plus"] = mplus

    # 估值因子在模型里的重要性排名
    print("\n=== 10 个估值因子的重要性排名 (Alpha158Plus, 共 168 因子) ===", flush=True)
    rank = {name: int(impplus.index.get_loc(name)) + 1 for name in VALUE_NAMES if name in impplus.index}
    for name in VALUE_NAMES:
        if name in impplus.index:
            print(f"  {name:8s} gain={impplus[name]:9.1f}  排名 {rank[name]}/168", flush=True)

    # 落盘
    out = {k: {kk: vv for kk, vv in v.items() if kk != "yearly"} for k, v in res.items()}
    for k in res:
        out[k]["yearly"] = res[k]["yearly"]
    json.dump(out, open(OUT / "compare.json", "w"), indent=1, ensure_ascii=False)
    impplus.to_csv(OUT / "importance_plus.csv", header=["gain"])

    print("\n=== 对比汇总 ===", flush=True)
    print(f"{'指标':16s} {'Alpha158':>14s} {'Alpha158Plus':>14s}", flush=True)
    for key, label in [("n_factor", "因子数"), ("IC", "IC"), ("RankIC", "RankIC"),
                       ("ICIR", "ICIR"), ("excess_net_ann", "净超额%"),
                       ("excess_ir", "信息比率"), ("turnover", "日均换手%"),
                       ("net_sharpe", "净夏普"), ("net_mdd", "净回撤%")]:
        a, b = res["Alpha158"][key], res["Alpha158Plus"][key]
        print(f"{label:16s} {a:14.4f} {b:14.4f}", flush=True)
    print("\n[DONE] 结果存 results/runs/round4/", flush=True)


if __name__ == "__main__":
    main()
