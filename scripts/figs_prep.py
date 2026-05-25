"""阶段1：跑5组关键配置，把日序列报告存下来供画图脚本读取。
每组约 1-2 分钟。"""
import warnings; warnings.filterwarnings("ignore")
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

DATA = '/Users/cedricyu/qlib量化研究/results/figures_data'
import os; os.makedirs(DATA, exist_ok=True)

LGB_DEFAULT = dict(objective="l2", verbosity=-1, num_threads=4,
                   colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
                   lambda_l1=205.6999, lambda_l2=580.9768,
                   max_depth=8, num_leaves=210)
# v3 上 Optuna 调出来的（来自 round 7）
LGB_V3_TUNED = dict(objective="l2", verbosity=-1, num_threads=4, **json.load(
    open("/Users/cedricyu/qlib量化研究/results/runs/round7/round7_result.json"))["best_params"])

EXCHANGE = dict(limit_threshold=0.095, deal_price="close",
                open_cost=0.001, close_cost=0.002, min_cost=5)


def build_ds(provider, market):
    qlib.init(provider_uri=provider, region="cn")
    dh = dict(start_time="2010-01-04", end_time="2026-05-19",
              fit_start_time="2010-01-04", fit_end_time="2017-12-31",
              instruments=market,
              infer_processors=[
                  {"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                  {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
              learn_processors=[{"class":"DropnaLabel"},
                  {"class":"CSRankNorm","kwargs":{"fields_group":"label"}}],
              label=["Ref($close, -2) / Ref($close, -1) - 1"])
    return DatasetH(handler=Alpha158(**dh), segments={
        "train": ("2010-01-04","2017-12-31"),
        "valid": ("2018-01-01","2019-12-31"),
        "test": ("2020-10-01","2026-05-15")})


def run_config(tag, provider, market, benchmark, lgb_params, strategy_args, save_ic=False):
    print(f"\n=== {tag} ===", flush=True)
    ds = build_ds(provider, market)
    tr = ds.prepare("train", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    va = ds.prepare("valid", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    te_feat = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    te_label = ds.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:,0]
    Xtr,ytr = tr["feature"], tr["label"].iloc[:,0]
    Xva,yva = va["feature"], va["label"].iloc[:,0]
    dtr = lgb.Dataset(Xtr, label=ytr, params={"feature_pre_filter": False})
    dva = lgb.Dataset(Xva, label=yva, params={"feature_pre_filter": False})
    bm = lgb.train(lgb_params, dtr, num_boost_round=500, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pred = pd.Series(bm.predict(te_feat), index=te_feat.index)
    strat = TopkDropoutStrategy(signal=pred, **strategy_args)
    ex = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time="2020-10-01", end_time="2026-05-15", strategy=strat, executor=ex,
                      benchmark=benchmark, account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd["1day"][0]
    rep.index = pd.to_datetime(rep.index)
    rep.to_pickle(f"{DATA}/{tag}_report.pkl")
    pred.to_pickle(f"{DATA}/{tag}_pred.pkl")
    print(f"  报告已存 → {tag}_report.pkl  ({len(rep)} 天)", flush=True)

    if save_ic:
        # 计算 IC 时间序列
        df = pd.concat([pred.rename("p"), te_label.rename("y")], axis=1).dropna()
        ic = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"]))
        ric = df.groupby(level=0).apply(lambda g: g["p"].corr(g["y"], method="spearman"))
        ic.to_pickle(f"{DATA}/{tag}_ic.pkl")
        ric.to_pickle(f"{DATA}/{tag}_ric.pkl")
        # 因子重要性（前 20）
        cols = list(te_feat.columns)
        gain = bm.feature_importance(importance_type="gain")
        imp = pd.Series(gain, index=cols).sort_values(ascending=False)
        imp.head(30).to_pickle(f"{DATA}/{tag}_imp.pkl")
        print(f"  IC/重要性已存", flush=True)


if __name__ == "__main__":
    # 5 个关键配置
    run_config("R3_CSI300_baseline",
               "/Users/cedricyu/.qlib/qlib_data/cn_data_v4", "csi300", "SH000300",
               LGB_DEFAULT, dict(topk=50, n_drop=1, hold_thresh=1), save_ic=True)
    run_config("R6_CSI500biased_default",
               "/Users/cedricyu/.qlib/qlib_data/cn_data_v3", "csi500", "SH000905",
               LGB_DEFAULT, dict(topk=50, n_drop=1, hold_thresh=1))
    run_config("R7_CSI500biased_tuned",
               "/Users/cedricyu/.qlib/qlib_data/cn_data_v3", "csi500", "SH000905",
               LGB_V3_TUNED, dict(topk=50, n_drop=1, hold_thresh=1))
    run_config("R8_CSI500debiased_tunedhp",
               "/Users/cedricyu/.qlib/qlib_data/cn_data_v4", "csi500", "SH000905",
               LGB_V3_TUNED, dict(topk=50, n_drop=1, hold_thresh=1))
    run_config("R85_CSI500debiased_FINAL",
               "/Users/cedricyu/.qlib/qlib_data/cn_data_v4", "csi500", "SH000905",
               LGB_DEFAULT, dict(topk=30, n_drop=1, hold_thresh=1), save_ic=True)
    print("\n[DONE] 5 组数据已就绪 → results/figures_data/", flush=True)
