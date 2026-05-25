"""第9轮 Step 1：Ridge 线性模型 baseline。
跟 LightGBM 对比，并存预测值供后续集成用。"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from pathlib import Path

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round9')
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path('/Users/cedricyu/qlib量化研究/results/figures_data')

PROVIDER = '/Users/cedricyu/.qlib/qlib_data/cn_data_v4'
BT_START, BT_END = '2020-10-01', '2026-05-15'
BENCHMARK = 'SH000905'
EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def ann(x): return x.mean() * 252 * 100
def vol(x): return x.std() * np.sqrt(252) * 100
def mdd(x):
    eq=(1+x).cumprod(); return (eq/eq.cummax()-1).min()*100


def run_bt(pred, topk=30, n_drop=1):
    strat = TopkDropoutStrategy(signal=pred, topk=topk, n_drop=n_drop, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time=BT_START, end_time=BT_END, strategy=strat, executor=ex,
                      benchmark=BENCHMARK, account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    ret, bench, cost = rep['return'], rep['bench'], rep['cost']
    net = ret - cost; ex_net = net - bench
    return dict(net_ann=ann(net), bench_ann=ann(bench), excess_ann=ann(ex_net),
                excess_ir=ann(ex_net)/vol(ex_net),
                net_sharpe=ann(net)/vol(net),
                excess_mdd=mdd(ex_net), net_mdd=mdd(net),
                turnover=rep['turnover'].mean()*100), rep


def eval_ic(pred, label):
    df = pd.concat([pred.rename('p'), label.rename('y')], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g['p'].corr(g['y']))
    ric = df.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
    return ic.mean(), ric.mean(), ic.mean()/ic.std(), ric.mean()/ric.std()


if __name__ == '__main__':
    qlib.init(provider_uri=PROVIDER, region='cn')
    print('[1] 加载数据 ...', flush=True)
    dh = dict(start_time='2010-01-04', end_time='2026-05-19',
              fit_start_time='2010-01-04', fit_end_time='2017-12-31',
              instruments='csi500',
              infer_processors=[
                  {'class':'RobustZScoreNorm','kwargs':{'fields_group':'feature','clip_outlier':True}},
                  {'class':'Fillna','kwargs':{'fields_group':'feature'}}],
              learn_processors=[{'class':'DropnaLabel'},
                  {'class':'CSRankNorm','kwargs':{'fields_group':'label'}}],
              label=['Ref($close, -2) / Ref($close, -1) - 1'])
    ds = DatasetH(handler=Alpha158(**dh), segments={
        'train': ('2010-01-04','2017-12-31'),
        'valid': ('2018-01-01','2019-12-31'),
        'test': ('2020-10-01','2026-05-15')})
    tr = ds.prepare('train', col_set=['feature','label'], data_key=DataHandlerLP.DK_L)
    va = ds.prepare('valid', col_set=['feature','label'], data_key=DataHandlerLP.DK_L)
    te_feat = ds.prepare('test', col_set='feature', data_key=DataHandlerLP.DK_I)
    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:,0]
    Xtr, ytr = tr['feature'].values, tr['label'].iloc[:,0].values
    Xva, yva = va['feature'].values, va['label'].iloc[:,0].values
    Xte = te_feat.values
    print(f'    train={Xtr.shape}  valid={Xva.shape}  test={Xte.shape}', flush=True)

    print('[2] 训练 Ridge (CV 选 alpha) ...', flush=True)
    Xtv = np.vstack([Xtr, Xva])
    ytv = np.concatenate([ytr, yva])
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000, 10000], cv=5)
    ridge.fit(Xtv, ytv)
    print(f'    最优 alpha = {ridge.alpha_}', flush=True)

    print('[3] 预测测试集 + IC ...', flush=True)
    pred = pd.Series(ridge.predict(Xte), index=te_feat.index)
    pred.to_pickle(DATA / 'R9_Ridge_pred.pkl')
    ic, ric, icir, ricir = eval_ic(pred, te_label)
    print(f'    IC={ic:.4f}  RankIC={ric:.4f}  ICIR={icir:.3f}  RankICIR={ricir:.3f}', flush=True)

    print('[4] 回测 (topk=30, n_drop=1) ...', flush=True)
    m, rep = run_bt(pred, topk=30, n_drop=1)
    rep.to_pickle(DATA / 'R9_Ridge_report.pkl')
    m.update(dict(IC=ic, RankIC=ric, ICIR=icir, RankICIR=ricir, alpha=ridge.alpha_))
    json.dump(m, open(OUT / 'ridge_result.json', 'w'), indent=1)

    print('\n=== Ridge vs LightGBM (R8.5 最终) on CSI500 去偏 ===', flush=True)
    r85 = {'IC':0.0315, 'RankIC':0.0267, 'excess_ann':4.40, 'excess_ir':0.445, 'net_sharpe':0.566}
    print(f'{"指标":12s} {"R8.5 LGB":>10s} {"R9 Ridge":>10s}', flush=True)
    for k, lab in [('IC','IC'),('RankIC','RankIC'),('excess_ann','净超额%'),
                   ('excess_ir','信息比率'),('net_sharpe','净夏普')]:
        print(f'{lab:12s} {r85[k]:10.4f} {m[k]:10.4f}', flush=True)
    print('\n[DONE] Ridge 预测+报告已存')
