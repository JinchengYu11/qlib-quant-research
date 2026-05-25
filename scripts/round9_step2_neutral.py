"""第9轮 Step 2：在最优配置上加行业中性化。
对比 with/without 行业中性化。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
sys.path.insert(0, '/Users/cedricyu/qlib量化研究/factors')
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from pathlib import Path

from industry_neutral import IndustryNeutral

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round9')
DATA = Path('/Users/cedricyu/qlib量化研究/results/figures_data')
IND_PATH = '/Users/cedricyu/qlib量化研究/data_raw/meta/industries.json'

LGB_DEFAULT = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
                   lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)
EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def build_ds(use_neutral: bool):
    base_infer = [
        {'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
        {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}},
    ]
    if use_neutral:
        # 在 norm 之后 + Fillna 之前插入行业中性化
        infer = [base_infer[0],
                 IndustryNeutral(industry_map_path=IND_PATH, fields_group='feature'),
                 base_infer[1]]
    else:
        infer = base_infer
    dh = dict(start_time='2010-01-04', end_time='2026-05-19',
              fit_start_time='2010-01-04', fit_end_time='2017-12-31',
              instruments='csi500',
              infer_processors=infer,
              learn_processors=[{'class': 'DropnaLabel'},
                  {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
              label=['Ref($close, -2) / Ref($close, -1) - 1'])
    return DatasetH(handler=Alpha158(**dh), segments={
        'train': ('2010-01-04', '2017-12-31'),
        'valid': ('2018-01-01', '2019-12-31'),
        'test': ('2020-10-01', '2026-05-15')})


def ann(x): return x.mean() * 252 * 100
def vol(x): return x.std() * np.sqrt(252) * 100
def mdd(x):
    eq=(1+x).cumprod(); return (eq/eq.cummax()-1).min()*100


def run(tag, use_neutral):
    print(f"\n=== {tag} ===", flush=True)
    ds = build_ds(use_neutral)
    model = LGBModel(**LGB_DEFAULT)
    model.fit(ds)
    pred = model.predict(ds, segment='test')
    if isinstance(pred, pd.DataFrame): pred = pred.iloc[:, 0]
    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:, 0]
    df = pd.concat([pred.rename('p'), te_label.rename('y')], axis=1).dropna()
    ic = df.groupby(level=0).apply(lambda g: g['p'].corr(g['y'])).mean()
    ric = df.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
    icir = ic / df.groupby(level=0).apply(lambda g: g['p'].corr(g['y'])).std()
    ricir = ric.mean() / ric.std()

    strat = TopkDropoutStrategy(signal=pred, topk=30, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15', strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    ret, bench, cost = rep['return'], rep['bench'], rep['cost']
    net = ret - cost; ex_net = net - bench
    m = dict(net_ann=ann(net), bench_ann=ann(bench), excess_ann=ann(ex_net),
             excess_ir=ann(ex_net)/vol(ex_net), net_sharpe=ann(net)/vol(net),
             excess_mdd=mdd(ex_net), net_mdd=mdd(net),
             turnover=rep['turnover'].mean()*100, IC=ic, RankIC=ric.mean(),
             ICIR=icir, RankICIR=ricir)
    print(f"  因子 IC={ic:.4f} RankIC={ric.mean():.4f} ICIR={icir:.3f}", flush=True)
    print(f"  策略净 {m['net_ann']:+.2f}% / 基准 {m['bench_ann']:+.2f}% / 净超额 {m['excess_ann']:+.2f}%", flush=True)
    print(f"  IR {m['excess_ir']:.3f} / 净夏普 {m['net_sharpe']:.3f} / 净回撤 {m['net_mdd']:.1f}% / 超额回撤 {m['excess_mdd']:.1f}%", flush=True)
    return m, pred, rep


if __name__ == '__main__':
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
    m_base, pred_base, rep_base = run('Baseline 无中性化 (R8.5 复现)', use_neutral=False)
    m_neu, pred_neu, rep_neu = run('+ 行业中性化', use_neutral=True)
    pred_neu.to_pickle(DATA / 'R9_IndNeutral_pred.pkl')
    rep_neu.to_pickle(DATA / 'R9_IndNeutral_report.pkl')
    json.dump({'baseline': m_base, 'neutral': m_neu},
              open(OUT / 'industry_neutral_result.json', 'w'), indent=1)

    print("\n=== 对比 ===")
    print(f"{'指标':12s} {'无中性化':>12s} {'+ 行业中性化':>14s}  差值")
    for k, lab in [('excess_ann','净超额%'),('excess_ir','信息比率'),
                   ('net_sharpe','净夏普'),('net_mdd','净回撤%'),
                   ('excess_mdd','超额回撤%'),('turnover','换手%'),
                   ('IC','IC'),('RankIC','RankIC')]:
        a, b = m_base[k], m_neu[k]
        print(f'{lab:12s} {a:12.4f} {b:14.4f}  {b-a:+.4f}')
    print('\n[DONE]')
