"""跑 LGB+Ridge 等权集成的回测。"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

if __name__ == '__main__':
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
    pred = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_EQens_pred.pkl')
    strat = TopkDropoutStrategy(signal=pred, topk=30, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15', strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8,
                      exchange_kwargs=dict(limit_threshold=0.095, deal_price='close',
                                           open_cost=0.001, close_cost=0.002, min_cost=5))
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    ret, bench, cost = rep['return'], rep['bench'], rep['cost']
    net = ret - cost; ex_net = net - bench
    def y(s): return s.mean()*252*100
    def vol(s): return s.std()*np.sqrt(252)*100
    def mdd(s):
        eq=(1+s).cumprod(); return (eq/eq.cummax()-1).min()*100
    print('=== LGB+Ridge 等权集成 回测 ===')
    print(f'  净超额={y(ex_net):+.2f}%   IR={y(ex_net)/vol(ex_net):.3f}')
    print(f'  净夏普={y(net)/vol(net):.3f}   净回撤={mdd(net):.1f}%   超额回撤={mdd(ex_net):.1f}%')
    print(f'  换手={rep["turnover"].mean()*100:.2f}%')
    rep.to_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_EQens_report.pkl')
