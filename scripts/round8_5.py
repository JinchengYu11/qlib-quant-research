"""
第 8.5 轮：在去偏 CSI500 (cn_data_v4) 上重新校准工具箱。
- A 线：策略参数扫描 (与第 7 轮同 7 组)
- B 线：Optuna 80 轮重新搜索超参 (目标=验证集 RankICIR)
- 与第 8 轮基线 (v3 调优超参直接搬过来) 对比
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

PROJ = Path("/Users/cedricyu/qlib量化研究")

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = PROJ / "results" / "runs" / "round8_5"
OUT.mkdir(parents=True, exist_ok=True)
PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v4"  # 去偏
BT_START, BT_END = "2020-10-01", "2026-05-15"
BENCHMARK = "SH000905"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)
LGB_DEFAULT = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
                   lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)
N_TRIALS = 80

DH = dict(
    start_time="2010-01-04", end_time="2026-05-19",
    fit_start_time="2010-01-04", fit_end_time="2017-12-31",
    instruments="csi500",
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


def run_bt(pred, topk, n_drop, hold_thresh):
    strat = TopkDropoutStrategy(signal=pred, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh)
    ex = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time=BT_START, end_time=BT_END, strategy=strat, executor=ex,
                      benchmark=BENCHMARK, account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd["1day"][0]
    rep.index = pd.to_datetime(rep.index)
    ret, bench, cost, turn = rep["return"], rep["bench"], rep["cost"], rep["turnover"]
    net = ret - cost
    ex_net = net - bench
    yearly = {int(y): round(float(g.sum() * 100), 2) for y, g in ex_net.groupby(ex_net.index.year)}
    return dict(excess_net_ann=ann(ex_net) * 100, excess_ir=sharpe(ex_net),
                gross_excess_ann=ann(ret - bench) * 100, turnover=turn.mean() * 100,
                cost_ann=ann(cost) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100,
                yearly=yearly)


def rank_ic_ir(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    ric = ric.dropna()
    if len(ric) < 2 or ric.std() == 0:
        return 0.0
    return ric.mean() / ric.std()


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    print("[1] 构建 CSI500 去偏数据集 + 训练默认超参模型 ...", flush=True)
    handler = Alpha158(**DH)
    ds = DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })
    model = LGBModel(**LGB_DEFAULT)
    model.fit(ds)
    pred_default = model.predict(ds, segment="test")
    if isinstance(pred_default, pd.DataFrame):
        pred_default = pred_default.iloc[:, 0]

    # === A 线：策略参数扫描 ===
    print("\n[2] A 线：v4 上策略扫描 (默认超参) ...", flush=True)
    configs = [
        ("baseline-ndrop1", dict(topk=50, n_drop=1, hold_thresh=1)),
        ("T2-ndrop2", dict(topk=50, n_drop=2, hold_thresh=1)),
        ("T3-top30", dict(topk=30, n_drop=1, hold_thresh=1)),
        ("T4-top85", dict(topk=85, n_drop=1, hold_thresh=1)),
        ("T5-hold5", dict(topk=50, n_drop=1, hold_thresh=5)),
        ("T6-hold10", dict(topk=50, n_drop=1, hold_thresh=10)),
    ]
    sweep = []
    for name, cfg in configs:
        m = run_bt(pred_default, **cfg)
        m["name"] = name
        m.update(cfg)
        sweep.append(m)
        print(f"  {name}: topk={cfg['topk']} n_drop={cfg['n_drop']} hold={cfg['hold_thresh']} "
              f"| 净超额={m['excess_net_ann']:+.2f}% IR={m['excess_ir']:+.3f} "
              f"换手={m['turnover']:.1f}% 净夏普={m['net_sharpe']:.3f}", flush=True)
    pd.DataFrame(sweep).drop(columns="yearly").to_csv(OUT / "v4_strategy_sweep.csv", index=False)

    # === B 线：Optuna 超参优化 (v4 上重新搜) ===
    print(f"\n[3] B 线：Optuna {N_TRIALS} 轮 (v4 验证集) ...", flush=True)
    tr = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    va = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    te_feat = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    Xtr, ytr = tr["feature"], tr["label"].iloc[:, 0]
    Xva, yva = va["feature"], va["label"].iloc[:, 0]
    dtr = lgb.Dataset(Xtr, label=ytr, params={"feature_pre_filter": False})
    dva = lgb.Dataset(Xva, label=yva, params={"feature_pre_filter": False})

    def objective(trial):
        params = dict(
            objective="l2", verbosity=-1, num_threads=8,
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 511),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 500, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 1000, log=True),
        )
        bm = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = pd.Series(bm.predict(Xva), index=Xva.index)
        ricir = rank_ic_ir(pred, yva)
        trial.set_user_attr("best_iter", bm.best_iteration)
        return ricir

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    done = {"n": 0}
    def cb(st, tr):
        done["n"] += 1
        if done["n"] % 10 == 0:
            print(f"  [{done['n']}/{N_TRIALS}] 当前最优验证 RankICIR = {st.best_value:.4f}", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])

    best = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iter", 200)
    print(f"\n[4] v4 上最优超参 (验证 RankICIR={study.best_value:.4f}):", flush=True)
    for k, v in best.items():
        print(f"    {k}: {v}", flush=True)

    # 用 v4 最优超参重训测试
    final_params = dict(objective="l2", verbosity=-1, num_threads=8, **best)
    bm = lgb.train(final_params, dtr, num_boost_round=max(int(best_iter), 50),
                   valid_sets=[dva], callbacks=[lgb.log_evaluation(0)])
    pred_tuned = pd.Series(bm.predict(te_feat), index=te_feat.index)
    m_tuned = run_bt(pred_tuned, topk=50, n_drop=1, hold_thresh=1)
    print(f"\n[5] v4 调优超参 + n_drop=1 回测:", flush=True)
    print(f"    净超额={m_tuned['excess_net_ann']:+.2f}%  IR={m_tuned['excess_ir']:+.3f}  净夏普={m_tuned['net_sharpe']:.3f}  回撤={m_tuned['net_mdd']:.1f}%", flush=True)

    # 同时跑最优策略参数 × v4 最优超参 的组合（看是否能再叠加）
    print(f"\n[6] 额外测试：v4 调优超参 + v4 最优策略 (n_drop=2):", flush=True)
    m_combo = run_bt(pred_tuned, topk=50, n_drop=2, hold_thresh=1)
    print(f"    净超额={m_combo['excess_net_ann']:+.2f}%  IR={m_combo['excess_ir']:+.3f}  净夏普={m_combo['net_sharpe']:.3f}", flush=True)

    json.dump({"v4_sweep": sweep, "v4_tuned_ndrop1": m_tuned, "v4_tuned_ndrop2": m_combo,
               "v4_best_params": best, "v4_valid_rankicir": float(study.best_value),
               "v3_baseline_for_reference": {"net_excess": 3.91, "IR": 0.473, "net_sharpe": 0.586}},
              open(OUT / "round8_5_result.json", "w"), indent=1, ensure_ascii=False)

    print("\n=== 第 8.5 轮综合对比 ===", flush=True)
    print(f"v3 调优超参 + n_drop=1 (上轮基线):  净超额=+3.91%  IR=0.473  (来自第8轮)")
    base = next(s for s in sweep if s["name"] == "baseline-ndrop1")
    print(f"v4 默认超参 + n_drop=1:           净超额={base['excess_net_ann']:+.2f}%  IR={base['excess_ir']:.3f}")
    best_sweep = max(sweep, key=lambda s: s["excess_ir"])
    print(f"v4 默认超参 + 最优策略 {best_sweep['name']:14s}: 净超额={best_sweep['excess_net_ann']:+.2f}%  IR={best_sweep['excess_ir']:.3f}")
    print(f"v4 调优超参 + n_drop=1:           净超额={m_tuned['excess_net_ann']:+.2f}%  IR={m_tuned['excess_ir']:.3f}")
    print(f"v4 调优超参 + n_drop=2:           净超额={m_combo['excess_net_ann']:+.2f}%  IR={m_combo['excess_ir']:.3f}")
    print(f"\n[DONE] 结果存 results/runs/round8_5/", flush=True)


if __name__ == "__main__":
    main()
