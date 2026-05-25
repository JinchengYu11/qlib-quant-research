"""
第 6 轮：CSI500 vs CSI300 universe 对比。
都用 Alpha158 + 默认超参 + n_drop=1 策略，唯一差异是股票池和基准。
CSI500 部分基于当前 500 只成分（有幸存者偏差，结论方向性参考）。
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
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = PROJ / "results" / "runs" / "round6"
OUT.mkdir(parents=True, exist_ok=True)
PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v3"
BT_START, BT_END = "2020-10-01", "2026-05-15"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)
LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)

DH_BASE = dict(
    start_time="2010-01-04", end_time="2026-05-19",
    fit_start_time="2010-01-04", fit_end_time="2017-12-31",
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


def ann(x, n=252): return x.mean() * n
def vol(x, n=252): return x.std() * np.sqrt(n)
def sharpe(x): return ann(x) / vol(x) if vol(x) > 0 else float("nan")
def mdd(x):
    eq = (1 + x).cumprod()
    return (eq / eq.cummax() - 1).min()


def run_bt(pred, benchmark):
    strat = TopkDropoutStrategy(signal=pred, topk=50, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time=BT_START, end_time=BT_END, strategy=strat, executor=ex,
                      benchmark=benchmark, account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd["1day"][0]
    rep.index = pd.to_datetime(rep.index)
    ret, bench, cost, turn = rep["return"], rep["bench"], rep["cost"], rep["turnover"]
    net = ret - cost
    ex_net = net - bench
    yearly = {int(y): round(float(g.sum() * 100), 2) for y, g in ex_net.groupby(ex_net.index.year)}
    return dict(excess_net_ann=ann(ex_net) * 100, excess_ir=sharpe(ex_net),
                gross_excess_ann=ann(ret - bench) * 100, turnover=turn.mean() * 100,
                cost_ann=ann(cost) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100,
                strategy_ann=ann(net) * 100, bench_ann=ann(bench) * 100,
                yearly=yearly)


def run_universe(market, benchmark, tag):
    print(f"\n=== {tag} (universe={market}, benchmark={benchmark}) ===", flush=True)
    dh = dict(**DH_BASE, instruments=market)
    handler = Alpha158(**dh)
    ds = DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })
    model = LGBModel(**LGB)
    model.fit(ds)
    pred = model.predict(ds, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = ds.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"])).mean()
    ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    m = run_bt(pred, benchmark)
    m.update(dict(IC=float(ic), RankIC=float(ric.mean()), RankICIR=float(ric.mean()/ric.std()),
                  market=market, benchmark=benchmark))
    print(f"  因子=158  IC={ic:.4f}  RankIC={ric.mean():.4f}  RankICIR={m['RankICIR']:.3f}", flush=True)
    print(f"  策略年化(净)={m['strategy_ann']:+.2f}%  基准={m['bench_ann']:+.2f}%  净超额={m['excess_net_ann']:+.2f}%", flush=True)
    print(f"  信息比率={m['excess_ir']:+.3f}  换手={m['turnover']:.1f}%  净夏普={m['net_sharpe']:.3f}  回撤={m['net_mdd']:.1f}%", flush=True)
    return m


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    res = {}
    res["CSI300"] = run_universe("csi300", "SH000300", "CSI300 (Alpha158, n_drop=1)")
    res["CSI500"] = run_universe("csi500", "SH000905", "CSI500 (Alpha158, n_drop=1) - 当前成分有幸存者偏差")
    json.dump(res, open(OUT / "compare.json", "w"), indent=1, ensure_ascii=False)

    print("\n=== 对比 ===", flush=True)
    print(f"{'指标':16s} {'CSI300':>14s} {'CSI500':>14s}", flush=True)
    for k, lab in [("IC", "IC"), ("RankIC", "RankIC"), ("RankICIR", "RankICIR"),
                   ("strategy_ann", "策略年化%"), ("bench_ann", "基准年化%"),
                   ("excess_net_ann", "净超额%"), ("excess_ir", "信息比率"),
                   ("net_sharpe", "净夏普"), ("net_mdd", "净回撤%"), ("turnover", "换手%")]:
        a, b = res["CSI300"][k], res["CSI500"][k]
        print(f"{lab:16s} {a:14.4f} {b:14.4f}", flush=True)
    print(f"\n[DONE] 结果存 results/runs/round6/", flush=True)


if __name__ == "__main__":
    main()
