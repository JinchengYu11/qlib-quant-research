"""
第 5 轮：LightGBM 超参数优化（Optuna）。
- 在 Alpha158Plus（168 因子）上搜索
- 防过拟合：搜索目标 = 验证集(2018-2019) RankICIR，绝不碰测试集
- 最优超参 → 测试集(2020-2026)回测一次，对比第 4 轮默认超参
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
sys.path.insert(0, str(PROJ / "factors"))

import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from alpha158_plus import Alpha158Plus

OUT = PROJ / "results" / "runs" / "round5"
OUT.mkdir(parents=True, exist_ok=True)
PROVIDER = "/Users/cedricyu/.qlib/qlib_data/cn_data_v2"
BT_START, BT_END = "2020-10-01", "2026-05-15"
EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)
N_TRIALS = 120

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


def ann(x, n=252): return x.mean() * n
def vol(x, n=252): return x.std() * np.sqrt(n)
def sharpe(x): return ann(x) / vol(x) if vol(x) > 0 else float("nan")
def mdd(x):
    eq = (1 + x).cumprod()
    return (eq / eq.cummax() - 1).min()


def rank_ic_ir(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    ric = ric.dropna()
    if len(ric) < 2 or ric.std() == 0:
        return 0.0, 0.0, 0.0
    return ric.mean(), ric.std(), ric.mean() / ric.std()


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
    yearly = {int(y): round(float(g.sum() * 100), 2) for y, g in ex_net.groupby(ex_net.index.year)}
    return dict(excess_net_ann=ann(ex_net) * 100, excess_ir=sharpe(ex_net),
                gross_excess_ann=ann(ret - bench) * 100, turnover=turn.mean() * 100,
                cost_ann=ann(cost) * 100, net_sharpe=sharpe(net), net_mdd=mdd(net) * 100,
                yearly=yearly)


def main():
    qlib.init(provider_uri=PROVIDER, region="cn")
    print("[1] 构建 Alpha158Plus 数据集 ...", flush=True)
    handler = Alpha158Plus(**DH)
    ds = DatasetH(handler=handler, segments={
        "train": ("2010-01-04", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test": ("2020-10-01", "2026-05-15"),
    })
    tr = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    va = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    te_feat = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    te_label = ds.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
    Xtr, ytr = tr["feature"], tr["label"].iloc[:, 0]
    Xva, yva = va["feature"], va["label"].iloc[:, 0]
    va_label = yva  # 验证集 label（RankIC 对单调变换不变）
    # feature_pre_filter=False：Optuna 会动态改 min_child_samples，必须关掉预过滤
    dtr = lgb.Dataset(Xtr, label=ytr, params={"feature_pre_filter": False})
    dva = lgb.Dataset(Xva, label=yva, params={"feature_pre_filter": False})
    print(f"    train={Xtr.shape}  valid={Xva.shape}  test={te_feat.shape}", flush=True)

    # --- Optuna 目标：验证集 RankICIR ---
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
        _, _, ricir = rank_ic_ir(pred, va_label)
        trial.set_user_attr("best_iter", bm.best_iteration)
        return ricir

    print(f"[2] Optuna 搜索 {N_TRIALS} 轮（目标=验证集 RankICIR）...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    done = {"n": 0}
    def cb(st, tr):
        done["n"] += 1
        if done["n"] % 10 == 0:
            print(f"  [{done['n']}/{N_TRIALS}] 当前最优验证 RankICIR = {st.best_value:.4f}", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])

    best = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iter", 200)
    print(f"\n[3] 最优超参（验证 RankICIR={study.best_value:.4f}）:", flush=True)
    for k, v in best.items():
        print(f"    {k}: {v}", flush=True)
    print(f"    best_iteration: {best_iter}", flush=True)
    json.dump({"best_params": best, "best_iter": int(best_iter),
               "valid_rankicir": float(study.best_value)},
              open(OUT / "best_params.json", "w"), indent=1)

    # --- 用最优超参在 train 上重训（固定轮数），测试集回测 ---
    print("\n[4] 用最优超参重训 + 测试集回测 ...", flush=True)
    final_params = dict(objective="l2", verbosity=-1, num_threads=8, **best)
    bm = lgb.train(final_params, dtr, num_boost_round=max(int(best_iter), 50),
                   valid_sets=[dva], callbacks=[lgb.log_evaluation(0)])
    pred_te = pd.Series(bm.predict(te_feat), index=te_feat.index)
    df = pd.concat([pred_te.rename("p"), te_label.rename("y")], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"])).mean()
    ric_s = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    m = run_bt(pred_te)
    m.update(dict(IC=float(ic), RankIC=float(ric_s.mean()),
                  RankICIR=float(ric_s.mean() / ric_s.std())))
    json.dump(m, open(OUT / "tuned_result.json", "w"), indent=1, ensure_ascii=False)

    print("\n=== 第 5 轮结果对比 ===", flush=True)
    print(f"{'指标':16s} {'第4轮默认超参':>16s} {'第5轮调优超参':>16s}", flush=True)
    base = dict(IC=0.0335, RankIC=0.0325, RankICIR=0.214,
                excess_net_ann=2.53, excess_ir=0.342, net_sharpe=0.289, net_mdd=-35.74)
    for key, lab in [("IC", "IC"), ("RankIC", "RankIC"), ("RankICIR", "RankICIR"),
                     ("excess_net_ann", "净超额%"), ("excess_ir", "信息比率"),
                     ("net_sharpe", "净夏普"), ("net_mdd", "净回撤%")]:
        print(f"{lab:16s} {base[key]:16.4f} {m[key]:16.4f}", flush=True)
    print(f"\n逐年净超额: {m['yearly']}", flush=True)
    print("\n[DONE] 结果存 results/runs/round5/", flush=True)


if __name__ == "__main__":
    main()
