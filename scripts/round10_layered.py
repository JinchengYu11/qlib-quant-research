"""第10轮 (B1): 分层 long-short 分析。
用 R8.5 最优配置的预测值，把 CSI500 每天按打分分 5 层 (quintile)，
看每层未来 2 日的等权收益累计 + 多空 spread。
诊断价值：告诉我们信号到底在哪里强、哪里弱。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.sans-serif": ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a5568",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1a365d",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round10')
OUT.mkdir(parents=True, exist_ok=True)
FIGS = Path('/Users/cedricyu/qlib量化研究/results/figures')

NLAYERS = 5  # 分 5 层 (quintile)
COLORS = ["#c53030", "#dd6b20", "#a0aec0", "#4299e1", "#2c7a7b"]  # Q1差→Q5好

def ann(x): return x.mean() * 252 * 100
def vol(x): return x.std() * np.sqrt(252) * 100
def sharpe(x): return ann(x) / vol(x) if vol(x) > 0 else float("nan")
def mdd(x):
    eq = (1+x).cumprod(); return (eq/eq.cummax()-1).min() * 100


def main():
    print("[1] 加载预测值 + 真实 label ...", flush=True)
    pred = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    print(f"    预测值: {len(pred)} 行, MultiIndex (date, stock)", flush=True)

    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
    dh = dict(start_time='2010-01-04', end_time='2026-05-19',
              fit_start_time='2010-01-04', fit_end_time='2017-12-31', instruments='csi500',
              infer_processors=[
                  {'class':'RobustZScoreNorm','kwargs':{'fields_group':'feature','clip_outlier':True}},
                  {'class':'Fillna','kwargs':{'fields_group':'feature'}}],
              learn_processors=[{'class':'DropnaLabel'},
                  {'class':'CSRankNorm','kwargs':{'fields_group':'label'}}],
              label=['Ref($close, -2) / Ref($close, -1) - 1'])
    ds = DatasetH(handler=Alpha158(**dh), segments={
        'train':('2010-01-04','2017-12-31'),'valid':('2018-01-01','2019-12-31'),
        'test':('2020-10-01','2026-05-15')})
    # 用 raw label (DK_I infer 不处理 label, 用 DK_L 但 CSRankNorm 不改 quintile 排序)
    te_label_raw = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_I).iloc[:,0]
    print(f"    label: {len(te_label_raw)} 行", flush=True)

    df = pd.concat([pred.rename('pred'), te_label_raw.rename('ret')], axis=1).dropna()
    print(f"    对齐后: {len(df)} 行", flush=True)

    print("[2] 每天按预测打分分 5 层 (quintile) ...", flush=True)
    df['layer'] = df.groupby(level=0)['pred'].transform(
        lambda x: pd.qcut(x.rank(method='first'), NLAYERS, labels=False, duplicates='drop'))
    # 0=最低层(Q1), 4=最高层(Q5)
    print(f"    每层平均股数: {df.groupby(['layer']).size() / df.index.get_level_values(0).nunique()}", flush=True)

    print("[3] 计算每层等权日收益 + spread ...", flush=True)
    layer_ret = df.groupby([df.index.get_level_values(0), 'layer'])['ret'].mean().unstack()
    layer_ret.columns = [f'Q{int(c)+1}' for c in layer_ret.columns]
    layer_cum = (1 + layer_ret).cumprod()
    spread = layer_ret['Q5'] - layer_ret['Q1']  # 多空 spread
    spread_cum = (1 + spread).cumprod()

    print("\n=== 每层关键指标 ===", flush=True)
    stats = pd.DataFrame({
        'mean_daily_ret_bps': layer_ret.mean() * 10000,
        'ann_return_pct': layer_ret.apply(ann),
        'ann_vol_pct': layer_ret.apply(vol),
        'sharpe': layer_ret.apply(sharpe),
        'mdd_pct': layer_ret.apply(mdd),
    }).round(3)
    print(stats.to_string(), flush=True)
    print(f"\n=== 多空 spread (Q5 - Q1) ===", flush=True)
    print(f"  日均 spread:       {spread.mean()*10000:.2f} bps")
    print(f"  年化 spread:        {ann(spread):.2f}%")
    print(f"  年化波动:           {vol(spread):.2f}%")
    print(f"  IR (Sharpe):       {sharpe(spread):.3f}")
    print(f"  最大回撤:           {mdd(spread):.1f}%")

    # 单调性检验
    print(f"\n=== 单调性 ===", flush=True)
    mono = layer_ret.mean().tolist()
    print(f"  Q1→Q5 日均收益: {[f'{x*10000:.1f}' for x in mono]} bps")
    is_mono = all(mono[i] <= mono[i+1] for i in range(NLAYERS-1))
    print(f"  完美单调: {'✅ YES' if is_mono else '❌ NO (信号头部强尾部弱)'}", flush=True)

    # 保存
    json.dump({
        'layer_stats': stats.to_dict(),
        'spread': dict(daily_bps=float(spread.mean()*10000), ann_pct=float(ann(spread)),
                       sharpe=float(sharpe(spread)), mdd_pct=float(mdd(spread))),
        'monotonic': is_mono,
    }, open(OUT / 'layered_result.json', 'w'), indent=1)
    layer_ret.to_csv(OUT / 'layer_daily_returns.csv')

    # === 可视化 ===
    print("[4] 生成图表 ...", flush=True)

    # 图 1: 5 条累计净值 + spread
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor='white',
                              gridspec_kw={'height_ratios':[2,1]})
    ax = axes[0]
    for i, col in enumerate(layer_cum.columns):
        is_extreme = col in ['Q1', 'Q5']
        ax.plot(layer_cum.index, layer_cum[col], label=f'{col} {"(最差)" if col=="Q1" else "(最好)" if col=="Q5" else ""}',
                color=COLORS[i], linewidth=2.5 if is_extreme else 1.4,
                alpha=1.0 if is_extreme else 0.7)
    ax.set_title('图 R10-1 | 5 层等权多头累计净值（无成本，纯信号能力）', loc='left')
    ax.set_ylabel('累计净值（起始 1.0）')
    ax.legend(loc='upper left', fontsize=10, framealpha=0)
    ax.axhline(1.0, color='#cbd5e0', linewidth=0.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(spread_cum.index, spread_cum.values, color='#2c5282', linewidth=2.5)
    ax.fill_between(spread_cum.index, 1, spread_cum.values,
                    where=(spread_cum.values >= 1), color='#2f855a', alpha=0.15)
    ax.fill_between(spread_cum.index, 1, spread_cum.values,
                    where=(spread_cum.values < 1), color='#c53030', alpha=0.15)
    ax.set_title(f'多空 spread (Q5 − Q1)  ·  年化 {ann(spread):.2f}%  ·  Sharpe {sharpe(spread):.2f}', loc='left')
    ax.set_ylabel('Spread 累计净值')
    ax.axhline(1.0, color='#cbd5e0', linewidth=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig11_layered.png')
    plt.close()
    print(f"    ✓ fig11_layered.png", flush=True)

    # 图 2: 单调性柱状图 (年化 Sharpe by layer)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    sharpes = layer_ret.apply(sharpe).values
    ret_ann = layer_ret.apply(ann).values
    bars = ax.bar(layer_cum.columns, ret_ann, color=COLORS, alpha=0.92,
                  edgecolor='white', linewidth=1)
    # 标 Sharpe
    for i, (r, s) in enumerate(zip(ret_ann, sharpes)):
        ax.text(i, r + (1 if r>=0 else -2), f'年化\n{r:+.1f}%\nSharpe\n{s:.2f}',
                ha='center', fontsize=9, fontweight='bold',
                color='#1a365d' if r>=0 else '#c53030')
    ax.set_title('图 R10-2 | 分层年化收益 + Sharpe — 信号单调性诊断', loc='left')
    ax.set_ylabel('年化收益 (%)')
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig12_layer_monotonicity.png')
    plt.close()
    print(f"    ✓ fig12_layer_monotonicity.png", flush=True)

    print(f"\n[DONE] 结果存 results/runs/round10/  图表存 results/figures/", flush=True)


if __name__ == '__main__':
    main()
