"""第 13 轮 (B3): 财务因子 + Long-short 验证.

对比 4 个配置:
  A) R8.5 Alpha158 + long-only      (基线 1)
  B) R12 Alpha158 + LS-100% K=30    (基线 2)
  C) R13 Alpha158Finance + long-only
  D) R13 Alpha158Finance + LS-100% K=30  ← 我们希望突破 R12 的版本

如果 D > B: 财务因子真有增量
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
from pathlib import Path
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

from alpha158_finance import Alpha158Finance

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round13')
OUT.mkdir(parents=True, exist_ok=True)

LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)
EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def build_ds(handler_cls):
    dh = dict(start_time='2010-01-04', end_time='2026-05-19',
              fit_start_time='2010-01-04', fit_end_time='2017-12-31', instruments='csi500',
              infer_processors=[
                  {'class':'RobustZScoreNorm','kwargs':{'fields_group':'feature','clip_outlier':True}},
                  {'class':'Fillna','kwargs':{'fields_group':'feature'}}],
              learn_processors=[{'class':'DropnaLabel'},
                  {'class':'CSRankNorm','kwargs':{'fields_group':'label'}}],
              label=['Ref($close, -2) / Ref($close, -1) - 1'])
    return DatasetH(handler=handler_cls(**dh), segments={
        'train':('2010-01-04','2017-12-31'),'valid':('2018-01-01','2019-12-31'),
        'test':('2020-10-01','2026-05-15')})


def train_predict(ds):
    model = LGBModel(**LGB)
    model.fit(ds)
    pred = model.predict(ds, segment='test')
    if isinstance(pred, pd.DataFrame): pred = pred.iloc[:, 0]
    # 取 feature 重要性 + IC
    booster = model.model
    te_feat = ds.prepare('test', col_set='feature', data_key=DataHandlerLP.DK_I)
    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:, 0]
    df_ic = pd.concat([pred.rename('p'), te_label.rename('y')], axis=1).dropna()
    ic = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'])).mean()
    ric = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
    imp = pd.Series(booster.feature_importance(importance_type='gain'),
                    index=list(te_feat.columns)).sort_values(ascending=False)
    return pred, dict(IC=float(ic), RankIC=float(ric.mean()), RankICIR=float(ric.mean()/ric.std())), imp


def run_topk_backtest(pred, K=30):
    strat = TopkDropoutStrategy(signal=pred, topk=K, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15', strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    return rep


def metrics(series):
    a = series.mean() * 252; v = series.std() * np.sqrt(252)
    eq = (1+series).cumprod(); mdd = (eq/eq.cummax()-1).min()
    return dict(ann_pct=a*100, vol_pct=v*100, sharpe=a/v if v>0 else float('nan'), mdd_pct=mdd*100)


def evaluate_all(pred, tag):
    """跑 long-only + long-short 100% + long-short 200%, 返回 metrics."""
    # long-only
    rL = run_topk_backtest(pred, K=30)
    long_only_net = rL['return'] - rL['cost']
    long_only_excess = long_only_net - rL['bench']
    m_lo = metrics(long_only_excess); m_lo['name'] = f'{tag} long-only'

    # 空头 (反预测)
    rS = run_topk_backtest(-pred, K=30)
    short_net = -(rS['return'] - rS['cost'])
    long_net = rL['return'] - rL['cost']
    ls100 = 0.5 * (long_net + short_net)
    ls200 = 1.0 * (long_net + short_net)
    m_ls100 = metrics(ls100); m_ls100['name'] = f'{tag} LS-100% K=30'
    m_ls200 = metrics(ls200); m_ls200['name'] = f'{tag} LS-200% K=30'
    return [m_lo, m_ls100, m_ls200]


def main():
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
    summary = []

    # === A: Alpha158 (R8.5/R12 复现) ===
    print("\n=== A) Alpha158 (R8.5/R12 复现) ===", flush=True)
    ds_a = build_ds(Alpha158)
    pred_a, ic_a, imp_a = train_predict(ds_a)
    print(f"  IC={ic_a['IC']:.4f}  RankIC={ic_a['RankIC']:.4f}  RankICIR={ic_a['RankICIR']:.3f}", flush=True)
    results_a = evaluate_all(pred_a, "A158")
    summary.extend(results_a)
    for m in results_a:
        print(f"  {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% vol {m['vol_pct']:.1f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    # === B: Alpha158Finance ===
    print("\n=== B) Alpha158Finance (158 量价 + 9 财务) ===", flush=True)
    ds_b = build_ds(Alpha158Finance)
    pred_b, ic_b, imp_b = train_predict(ds_b)
    print(f"  IC={ic_b['IC']:.4f}  RankIC={ic_b['RankIC']:.4f}  RankICIR={ic_b['RankICIR']:.3f}", flush=True)
    results_b = evaluate_all(pred_b, "A158+FIN")
    summary.extend(results_b)
    for m in results_b:
        print(f"  {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% vol {m['vol_pct']:.1f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    # 财务因子重要性
    print(f"\n=== Alpha158Finance 中财务因子重要性 (gain 排名 / 167 因子) ===", flush=True)
    from alpha158_finance import FINANCE_NAMES
    rank_map = {n: int(imp_b.index.get_loc(n))+1 if n in imp_b.index else None for n in FINANCE_NAMES}
    for n in FINANCE_NAMES:
        r = rank_map.get(n)
        if r is None: continue
        print(f"  {n:10s}  gain={imp_b[n]:>8.1f}  排名 {r}/{len(imp_b)}", flush=True)

    # === 汇总对比 ===
    print(f"\n=== 汇总对比 ===", flush=True)
    print(f"{'配置':30s} {'年化':>8} {'vol':>7} {'Sharpe':>8} {'MDD':>8}", flush=True)
    for m in summary:
        print(f"{m['name']:30s} {m['ann_pct']:>+7.2f}% {m['vol_pct']:>6.1f}% {m['sharpe']:>+8.3f} {m['mdd_pct']:>+7.1f}%", flush=True)

    json.dump({"summary": summary,
               "alpha158_ic": ic_a, "alpha158finance_ic": ic_b,
               "finance_importance_ranks": rank_map},
              open(OUT / 'r13_result.json', 'w'), indent=1, default=str)
    print(f"\n[DONE] 结果存 {OUT}", flush=True)


if __name__ == '__main__':
    main()
