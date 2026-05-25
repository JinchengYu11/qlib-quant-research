"""
第 8 轮：用第 7 轮最优配置在 cn_data_v4 (CSI500 历史并集去幸存者偏差) 上回测。
对比含偏版（cn_data_v3 上的当前 500 成分）。
"""
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

OUT_DIR = '/Users/cedricyu/qlib量化研究/results/runs/round8'
import os; os.makedirs(OUT_DIR, exist_ok=True)


def run_one(provider, tag):
    qlib.init(provider_uri=provider, region="cn")
    best = json.load(open("/Users/cedricyu/qlib量化研究/results/runs/round7/round7_result.json"))["best_params"]
    dh = dict(start_time="2010-01-04", end_time="2026-05-19",
              fit_start_time="2010-01-04", fit_end_time="2017-12-31",
              instruments="csi500",
              infer_processors=[
                  {"class":"RobustZScoreNorm","kwargs":{"fields_group":"feature","clip_outlier":True}},
                  {"class":"Fillna","kwargs":{"fields_group":"feature"}}],
              learn_processors=[{"class":"DropnaLabel"},
                  {"class":"CSRankNorm","kwargs":{"fields_group":"label"}}],
              label=["Ref($close, -2) / Ref($close, -1) - 1"])
    ds = DatasetH(handler=Alpha158(**dh), segments={
        "train": ("2010-01-04","2017-12-31"),
        "valid": ("2018-01-01","2019-12-31"),
        "test": ("2020-10-01","2026-05-15")})
    tr=ds.prepare("train",col_set=["feature","label"],data_key=DataHandlerLP.DK_L)
    va=ds.prepare("valid",col_set=["feature","label"],data_key=DataHandlerLP.DK_L)
    te=ds.prepare("test",col_set="feature",data_key=DataHandlerLP.DK_I)
    Xtr,ytr=tr["feature"],tr["label"].iloc[:,0]
    Xva,yva=va["feature"],va["label"].iloc[:,0]
    dtr=lgb.Dataset(Xtr,label=ytr,params={"feature_pre_filter":False})
    dva=lgb.Dataset(Xva,label=yva,params={"feature_pre_filter":False})
    bm=lgb.train(dict(objective="l2",verbosity=-1,num_threads=4,**best),dtr,
                 num_boost_round=200,valid_sets=[dva],callbacks=[lgb.log_evaluation(0)])
    pred=pd.Series(bm.predict(te),index=te.index)
    strat=TopkDropoutStrategy(signal=pred,topk=50,n_drop=1,hold_thresh=1)
    ex=SimulatorExecutor(time_per_step="day",generate_portfolio_metrics=True)
    pmd,_=backtest(start_time="2020-10-01",end_time="2026-05-15",strategy=strat,executor=ex,
                   benchmark="SH000905",account=1e8,
                   exchange_kwargs=dict(limit_threshold=0.095,deal_price="close",
                                        open_cost=0.001,close_cost=0.002,min_cost=5))
    rep=pmd["1day"][0]; rep.index=pd.to_datetime(rep.index)
    ret,bench,cost=rep["return"],rep["bench"],rep["cost"]
    net=ret-cost; excess=net-bench

    def y(s): return s.mean()*252*100
    def vol(s): return s.std()*np.sqrt(252)*100
    def mdd(s):
        eq=(1+s).cumprod(); return (eq/eq.cummax()-1).min()*100

    yearly = {int(yr): round(float(g.sum()*100),2) for yr,g in excess.groupby(excess.index.year)}
    bench_y = {int(yr): round(float(bench[bench.index.year==yr].sum()*100),2) for yr,_ in excess.groupby(excess.index.year)}
    net_y = {int(yr): round(float(net[net.index.year==yr].sum()*100),2) for yr,_ in excess.groupby(excess.index.year)}

    m = dict(strategy_ann=y(net), bench_ann=y(bench),
             excess_net_ann=y(excess), excess_vol=vol(excess), excess_ir=y(excess)/vol(excess),
             net_vol=vol(net), net_sharpe=y(net)/vol(net), net_mdd=mdd(net),
             excess_mdd=mdd(excess), turnover=rep["turnover"].mean()*100,
             cost_ann=y(cost), n_days=len(rep), yearly=yearly, bench_yearly=bench_y, net_yearly=net_y)
    print(f"\n=== {tag} ===")
    print(f"  策略净 {m['strategy_ann']:+.2f}%/年  vol {m['net_vol']:.2f}%  夏普 {m['net_sharpe']:.3f}")
    print(f"  基准   {m['bench_ann']:+.2f}%/年  ")
    print(f"  净超额 {m['excess_net_ann']:+.2f}%/年  vol {m['excess_vol']:.2f}%  IR {m['excess_ir']:.3f}")
    print(f"  净回撤 {m['net_mdd']:.2f}%  超额回撤 {m['excess_mdd']:.2f}%")
    print(f"  日均换手 {m['turnover']:.2f}%  年化成本 {m['cost_ann']:.2f}%")
    print(f"  逐年净超额:")
    for yr in sorted(yearly):
        print(f"    {yr}: 净超额 {yearly[yr]:+6.2f}%  策略净 {net_y[yr]:+7.2f}%  基准 {bench_y[yr]:+7.2f}%")
    return m


if __name__ == "__main__":
    print("="*60)
    print("【第 7 轮配置】CSI500 + Alpha158 + 调优超参 + n_drop=1")
    print("="*60)

    res_v3 = run_one("/Users/cedricyu/.qlib/qlib_data/cn_data_v3",
                     "v3: CSI500 当前 500 成分 (⚠️ 有幸存者偏差)")
    res_v4 = run_one("/Users/cedricyu/.qlib/qlib_data/cn_data_v4",
                     "v4: CSI500 历史持仓并集 (✅ 已去幸存者偏差)")

    json.dump({"v3_current": res_v3, "v4_union": res_v4},
              open(OUT_DIR + "/compare.json", "w"), indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("【对比】当前成分（含偏）vs 历史并集（去偏）")
    print(f"{'='*60}")
    keys = [("strategy_ann","策略年化%"),("bench_ann","基准年化%"),
            ("excess_net_ann","净超额%"),("excess_ir","信息比率"),
            ("net_sharpe","净夏普"),("net_mdd","净回撤%"),
            ("excess_mdd","超额回撤%"),("turnover","换手%")]
    print(f"\n{'指标':14s} {'v3 (含偏)':>14s} {'v4 (去偏)':>14s}   差值")
    for k, lab in keys:
        a, b = res_v3[k], res_v4[k]
        print(f"{lab:14s} {a:14.3f} {b:14.3f}   {b-a:+.3f}")
