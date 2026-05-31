"""第 15 轮: 深度模型 (TabNet) on Alpha158 vs LGBM 基线.

研究问题: TabNet 这种 tabular-friendly 深度模型, 能不能在 R8.5 同样的 Alpha158 setup 上,
         超过 LightGBM 的 IR 0.445 (long-only) 或 Sharpe 0.716 (LS-100%) ?

对比 2 个配置:
  A) LGBM Alpha158 (R8.5/R12 复现)         <-- 跟 round13.py 第一组完全相同
  B) TabNet Alpha158                        <-- 同样数据, 同样 split, 同样评估

注: TabNet 默认 d_feat=158 跟 Alpha158 完美对齐 (Alpha158 是 cross-sectional tabular).
    Mac 没 CUDA, TabNet 跑 CPU. 用小 n_epochs + early_stop 控制时长.
    pretrain=False (skip 50 unsupervised epochs) 因为预训练对纯监督任务效果有限.

环境: conda activate qlib
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
sys.path.insert(0, '/Users/cedricyu/qlib量化研究/factors')

import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_tabnet import TabnetModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round15')
OUT.mkdir(parents=True, exist_ok=True)

# LGB: 用 round13 一致的 R7 Optuna 调优参数 (复现 R8.5/R12)
LGB = dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
           lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8)

# TabNet: 小心配置避免过长 (Mac CPU, 不预训练)
TABNET = dict(
    d_feat=158, out_dim=64, final_out_dim=1,
    batch_size=4096,
    n_d=32, n_a=32,            # 默认 64, 减小加速 + 防过拟
    n_shared=2, n_ind=2,
    n_steps=3,                  # 默认 5, 减为 3 加速
    n_epochs=30,                # 默认 100, 减为 30 配合 early_stop
    early_stop=10,
    relax=1.3, vbs=2048,
    optimizer="adam", loss="mse", lr=0.005,
    pretrain=False,             # 跳过 50 epoch 的预训练
    GPU=-1, seed=42,
)

EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def build_alpha158_ds():
    """跟 R8.5/R12/R13 同样的 Alpha158 数据集 + split."""
    dh = dict(
        start_time='2010-01-04', end_time='2026-05-19',
        fit_start_time='2010-01-04', fit_end_time='2017-12-31',
        instruments='csi500',
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


def train_predict(ds, model_cls, model_kwargs, tag):
    print(f"\n=== {tag}: 训练中 ...", flush=True)
    import time
    t0 = time.time()
    model = model_cls(**model_kwargs)
    model.fit(ds)
    train_time = time.time() - t0
    print(f"    {tag} 训练耗时 {train_time/60:.1f} min", flush=True)

    pred = model.predict(ds, segment='test')
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:, 0]
    df_ic = pd.concat([pred.rename('p'), te_label.rename('y')], axis=1).dropna()
    ic = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'])).mean()
    ric_ts = df_ic.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
    return pred, dict(IC=float(ic), RankIC=float(ric_ts.mean()),
                       RankICIR=float(ric_ts.mean() / ric_ts.std()),
                       train_time_min=train_time / 60)


def run_topk_backtest(pred, K=30):
    strat = TopkDropoutStrategy(signal=pred, topk=K, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15',
                       strategy=strat, executor=ex,
                       benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]
    rep.index = pd.to_datetime(rep.index)
    return rep


def metrics(series):
    a = series.mean() * 252
    v = series.std() * np.sqrt(252)
    eq = (1 + series).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(ann_pct=a*100, vol_pct=v*100,
                sharpe=a/v if v > 0 else float('nan'), mdd_pct=mdd*100)


def evaluate_all(pred, tag):
    """跟 round13 / R12 一样的评估: long-only, LS-100%, LS-200%."""
    rL = run_topk_backtest(pred, K=30)
    long_only_net = rL['return'] - rL['cost']
    long_only_excess = long_only_net - rL['bench']
    m_lo = metrics(long_only_excess); m_lo['name'] = f'{tag} long-only'

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

    print("[1] 构建 Alpha158 数据集 (R8.5/R12/R13 同款) ...", flush=True)
    ds = build_alpha158_ds()

    summary = []
    ic_results = {}

    # === A: LGBM Alpha158 (R8.5/R12 复现, 跟 R13 第一组完全一致) ===
    pred_lgb, ic_lgb = train_predict(ds, LGBModel, LGB, "LGBM_A158")
    pred_lgb.to_pickle(OUT / 'pred_lgbm.pkl')
    ic_results['LGBM_A158'] = ic_lgb
    print(f"  IC={ic_lgb['IC']:.4f}  RankIC={ic_lgb['RankIC']:.4f}  RankICIR={ic_lgb['RankICIR']:.3f}", flush=True)
    results_lgb = evaluate_all(pred_lgb, "LGBM")
    summary.extend(results_lgb)
    for m in results_lgb:
        print(f"  {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    # 保存 LGBM 子结果 (允许 TabNet 失败也有 LGBM 数据)
    json.dump({'summary': summary, 'ic_results': ic_results, 'tabnet_done': False},
              open(OUT / 'r15_result.json', 'w'), indent=1, default=str)

    # === B: TabNet Alpha158 ===
    pred_tab, ic_tab = train_predict(ds, TabnetModel, TABNET, "TabNet_A158")
    pred_tab.to_pickle(OUT / 'pred_tabnet.pkl')
    ic_results['TabNet_A158'] = ic_tab
    print(f"  IC={ic_tab['IC']:.4f}  RankIC={ic_tab['RankIC']:.4f}  RankICIR={ic_tab['RankICIR']:.3f}", flush=True)
    results_tab = evaluate_all(pred_tab, "TabNet")
    summary.extend(results_tab)
    for m in results_tab:
        print(f"  {m['name']:25s}: 年化 {m['ann_pct']:+.2f}% Sharpe {m['sharpe']:+.3f} MDD {m['mdd_pct']:+.1f}%", flush=True)

    # === 汇总 ===
    print(f"\n=== 汇总对比 ===", flush=True)
    print(f"{'配置':28s} {'年化':>8} {'vol':>7} {'Sharpe':>8} {'MDD':>8}", flush=True)
    for m in summary:
        print(f"{m['name']:28s} {m['ann_pct']:>+7.2f}% {m['vol_pct']:>6.1f}% "
              f"{m['sharpe']:>+8.3f} {m['mdd_pct']:>+7.1f}%", flush=True)

    json.dump({'summary': summary, 'ic_results': ic_results, 'tabnet_done': True,
               'lgb_config': LGB, 'tabnet_config': TABNET},
              open(OUT / 'r15_result.json', 'w'), indent=1, default=str)
    print(f"\n[DONE] 结果存 {OUT}", flush=True)


if __name__ == '__main__':
    main()
