"""跑多种集成方案回测。"""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor


def run(pred, tag):
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
    print(f'{tag:30s}  净超额={y(ex_net):+6.2f}%  IR={y(ex_net)/vol(ex_net):+6.3f}  '
          f'净夏普={y(net)/vol(net):.3f}  超额回撤={mdd(ex_net):.1f}%')


if __name__ == '__main__':
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')

    # 取并集集成：每天取 LGB top K 和 Ridge top K 的并集，然后用合成 score 重排
    ridge = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_Ridge_pred.pkl')
    lgb = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    df = pd.concat([ridge.rename('ridge'), lgb.rename('lgb')], axis=1).dropna()

    # Voting/Intersection: 只买两边都进 top 60 的股票（差不多 top 30 的"安全交集"）
    def voting_score(g, k=60):
        if len(g) < k:
            return pd.Series(np.nan, index=g.index)
        # 标志位：是否在 top k
        in_top_r = g['ridge'].rank(ascending=False) <= k
        in_top_l = g['lgb'].rank(ascending=False) <= k
        both = (in_top_r & in_top_l).astype(int)
        # 在两边都进的，分数 = ridge_rank + lgb_rank（越小越好），转为分数
        score = pd.Series(np.where(both, -(g['ridge'].rank(ascending=False) + g['lgb'].rank(ascending=False)), -1e6), index=g.index)
        return score
    voting60 = df.groupby(level=0, group_keys=False).apply(voting_score)

    # 比较 5 个集成方案
    icw = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_ICw_pred.pkl')
    w7030 = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_70_30_pred.pkl')
    eqens = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_EQens_pred.pkl')

    print('=== 不同集成方案 (vs 单模型) ===')
    run(lgb, '单模 LGB (R8.5 最终)')
    run(ridge, '单模 Ridge')
    run(eqens, '等权集成 (z-score)')
    run(icw, 'IC 加权集成 (R0.53 L0.47)')
    run(w7030, '70/30 偏 LGB')
    run(voting60, 'Voting (都进 top 60 才买)')
