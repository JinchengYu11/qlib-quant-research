"""第 16 轮 (横向): R8.5 配置在 CSI300 上的可迁移性验证.

研究问题: R8.5 (Alpha158 + LGBM + TopK=30, IR 0.445 on CSI500) 配置 *完全不动* 移植到 CSI300,
         能不能继续到 IR 0.4+?

  如果是 -> R8.5 是 A 股可迁移的稳健 baseline
  如果不是 -> R8.5 是 CSI500-specific, 揭示 universe 边界

配置: 跟 R8.5 / R12 / R13 / R15 完全一致
  - cn_data_v4 (含 CSI300 历史并集 790 只 + CSI500 历史并集 1623 只)
  - Alpha158 (158 量价因子)
  - LightGBM (R7 Optuna 调优同款参数)
  - TopkDropoutStrategy(topk=30, n_drop=1, hold_thresh=1)
  - 同样的 train/valid/test split

唯一变化: instruments 从 'csi500' 改为 'csi300'
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/cedricyu/qlib量化研究')

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round16')
OUT.mkdir(parents=True, exist_ok=True)

LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)

EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def build_ds(universe):
    """跟 R8.5 同款 dataset 配置, 只换 instruments."""
    dh = dict(
        start_time='2010-01-04', end_time='2026-05-19',
        fit_start_time='2010-01-04', fit_end_time='2017-12-31',
        instruments=universe,
        infer_processors=[
            {'class':'RobustZScoreNorm','kwargs':{'fields_group':'feature','clip_outlier':True}},
            {'class':'Fillna','kwargs':{'fields_group':'feature'}}],
        learn_processors=[
            {'class':'DropnaLabel'},
            {'class':'CSRankNorm','kwargs':{'fields_group':'label'}}],
        label=['Ref($close, -2) / Ref($close, -1) - 1'])
    return DatasetH(handler=Alpha158(**dh), segments={
        'train': ('2010-01-04', '2017-12-31'),
        'valid': ('2018-01-01', '2019-12-31'),
        'test':  ('2020-10-01', '2026-05-15')})


def train_predict(ds, tag):
    print(f"\n=== {tag}: 训练 LGBM ...", flush=True)
    import time
    t0 = time.time()
    model = LGBModel(**LGB)
    model.fit(ds)
    print(f"    训练 {time.time()-t0:.1f}s", flush=True)
    pred = model.predict(ds, segment='test')
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:, 0]
    df_ic = pd.concat([pred.rename('p'), te_label.rename('y')], axis=1).dropna()
    ic = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'])).mean()
    ric_ts = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
    return pred, dict(IC=float(ic), RankIC=float(ric_ts.mean()),
                       RankICIR=float(ric_ts.mean()/ric_ts.std()))


def run_topk(pred, bench, K=30):
    strat = TopkDropoutStrategy(signal=pred, topk=K, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15',
                       strategy=strat, executor=ex,
                       benchmark=bench, account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    return rep


def metrics(series):
    a = series.mean() * 252
    v = series.std() * np.sqrt(252)
    eq = (1 + series).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(ann_pct=a*100, vol_pct=v*100,
                sharpe=a/v if v > 0 else float('nan'), mdd_pct=mdd*100)


def evaluate(pred, bench, tag):
    rL = run_topk(pred, bench, K=30)
    lo_net = rL['return'] - rL['cost']
    lo_excess = lo_net - rL['bench']
    m_lo = metrics(lo_excess); m_lo['name'] = f'{tag} long-only'

    rS = run_topk(-pred, bench, K=30)
    sh_net = -(rS['return'] - rS['cost'])
    l_net = rL['return'] - rL['cost']
    ls100 = 0.5 * (l_net + sh_net)
    m_ls = metrics(ls100); m_ls['name'] = f'{tag} LS-100% K=30'
    return [m_lo, m_ls]


def main():
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')

    summary = []
    ic_results = {}

    print("[1] CSI300 universe (R16 横向验证)...", flush=True)
    ds_300 = build_ds('csi300')
    pred_300, ic_300 = train_predict(ds_300, "CSI300")
    pred_300.to_pickle(OUT / 'pred_csi300.pkl')
    ic_results['CSI300'] = ic_300
    print(f"    IC={ic_300['IC']:.4f}  RankIC={ic_300['RankIC']:.4f}  RankICIR={ic_300['RankICIR']:.3f}", flush=True)
    results_300 = evaluate(pred_300, 'SH000300', "CSI300")
    summary.extend(results_300)
    for m in results_300:
        print(f"    {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    print("\n[2] CSI500 baseline 重跑 (sanity check, 应复现 R8.5/R12)...", flush=True)
    ds_500 = build_ds('csi500')
    pred_500, ic_500 = train_predict(ds_500, "CSI500")
    pred_500.to_pickle(OUT / 'pred_csi500.pkl')
    ic_results['CSI500'] = ic_500
    print(f"    IC={ic_500['IC']:.4f}  RankIC={ic_500['RankIC']:.4f}  RankICIR={ic_500['RankICIR']:.3f}", flush=True)
    results_500 = evaluate(pred_500, 'SH000905', "CSI500")
    summary.extend(results_500)
    for m in results_500:
        print(f"    {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    print(f"\n=== 横向对比 (R8.5 配置 不动, 只换 universe) ===", flush=True)
    print(f"{'配置':30s} {'年化':>8} {'vol':>7} {'Sharpe':>8} {'MDD':>8}", flush=True)
    for m in summary:
        print(f"{m['name']:30s} {m['ann_pct']:>+7.2f}% {m['vol_pct']:>6.1f}% "
              f"{m['sharpe']:>+8.3f} {m['mdd_pct']:>+7.1f}%", flush=True)

    print(f"\n=== IC 对比 ===", flush=True)
    for n, ic in ic_results.items():
        print(f"  {n}: IC={ic['IC']:.4f}  RankIC={ic['RankIC']:.4f}  RankICIR={ic['RankICIR']:.3f}", flush=True)

    json.dump({'summary': summary, 'ic_results': ic_results, 'lgb_config': LGB},
              open(OUT / 'r16_result.json', 'w'), indent=1, default=str)
    print(f"\n[DONE] 结果存 {OUT}", flush=True)


if __name__ == '__main__':
    main()
