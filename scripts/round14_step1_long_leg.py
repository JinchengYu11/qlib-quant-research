"""R14 Step 1: 用 R8.5 预测值跑一次 qlib backtest 拿 long leg daily report, 落盘缓存.

输出: results/runs/round14/long_leg_daily.pkl
  pd.DataFrame indexed by date, columns:
    long_return:   net daily return after cost (策略层面)
    long_cost:     当日交易成本占比
    long_value:    EOD 账户总值 (qlib report['account'] 列)
    bench_return:  CSI500 当日收益 (qlib report['bench'] 列)
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'results' / 'runs' / 'round14'
OUT.mkdir(parents=True, exist_ok=True)

EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def main():
    print("[1] 加载 R8.5 预测值 ...", flush=True)
    pred = pd.read_pickle(ROOT / 'results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    print(f"    {len(pred)} 行", flush=True)

    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')

    print("[2] TopkDropout K=30 long-only backtest (复用 R8.5 配置) ...", flush=True)
    strat = TopkDropoutStrategy(signal=pred, topk=30, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15',
                      strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]
    rep.index = pd.to_datetime(rep.index)
    print(f"    报告 {len(rep)} 行, 列 {list(rep.columns)}", flush=True)

    out = pd.DataFrame({
        'long_value': rep['account'],
        'long_return': rep['return'] - rep['cost'],
        'long_cost': rep['cost'],
        'long_return_gross': rep['return'],
        'bench_return': rep['bench'],
        'turnover': rep['turnover'],
    }, index=rep.index)

    out.to_pickle(OUT / 'long_leg_daily.pkl')
    print(f"\n[3] 写出 {OUT / 'long_leg_daily.pkl'}", flush=True)
    print(f"    首日 {out.index[0]} long_value={out['long_value'].iloc[0]:,.0f}", flush=True)
    print(f"    末日 {out.index[-1]} long_value={out['long_value'].iloc[-1]:,.0f}", flush=True)
    print(f"    年化 long_return (净) = {out['long_return'].mean()*252*100:+.2f}%", flush=True)
    print(f"    年化 bench_return = {out['bench_return'].mean()*252*100:+.2f}%", flush=True)
    print(f"    净超额 = {(out['long_return']-out['bench_return']).mean()*252*100:+.2f}%", flush=True)
    print(f"    (R8.5 spec 数: 净超额 +4.40% — 应该高度一致)", flush=True)


if __name__ == '__main__':
    main()
