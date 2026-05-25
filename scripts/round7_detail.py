"""第7轮最优配置详细指标（年度拆解 + 夏普/IR 分解）。"""
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

if __name__ == "__main__":
    qlib.init(provider_uri="/Users/cedricyu/.qlib/qlib_data/cn_data_v3", region="cn")
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

    print("="*60)
    print("【第7轮最优：CSI500 + Alpha158 + 调优超参 + n_drop=1】")
    print("="*60)
    print(f"\n--- 全周期数字 ({rep.index[0].date()} ~ {rep.index[-1].date()}, {len(rep)} 个交易日) ---\n")
    print(f"基准 中证500:        年化 {y(bench):+.2f}%  年化波动 {vol(bench):.2f}%  基准夏普 {y(bench)/vol(bench):.3f}")
    print(f"策略 毛(无成本):     年化 {y(ret):+.2f}%  年化波动 {vol(ret):.2f}%")
    print(f"策略 净(扣成本):     年化 {y(net):+.2f}%  年化波动 {vol(net):.2f}%  净夏普 {y(net)/vol(net):.3f}")
    print(f"超额 净-基准:        年化 {y(excess):+.2f}%  年化波动 {vol(excess):.2f}%  信息比率 {y(excess)/vol(excess):.3f}")
    print()
    print(f"--- 为什么净夏普(1.04) 远低于 IR(1.50) ---")
    print(f"  策略波动 {vol(net):.1f}% 与基准波动 {vol(bench):.1f}% 几乎相等")
    print(f"  →  长只多头骑在中证500大盘上，绝对收益的方差由 market beta 主导")
    print(f"  →  夏普衡量\"含市场风险的总收益质量\"  →  被市场拖累")
    print(f"  超额波动只有 {vol(excess):.1f}%（剥掉了大盘那部分）")
    print(f"  →  IR 衡量\"纯 alpha 的稳定性\"  →  剥离市场后纯 alpha 很优秀")

    print(f"\n--- 逐年净超额 ---\n")
    for yr,g in excess.groupby(excess.index.year):
        bench_y=bench[bench.index.year==yr].sum()*100
        net_y=net[net.index.year==yr].sum()*100
        tag = "盈" if g.sum()>0 else "亏"
        print(f"  {yr}: 净超额 {g.sum()*100:+7.2f}%  |  策略净 {net_y:+7.2f}%  中证500 {bench_y:+7.2f}%  [{tag}]")
