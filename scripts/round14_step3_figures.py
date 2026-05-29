"""R14 Step 3: 画 fig14_engineering_ls.png — 跟 fig13 同布局.

上面板: 5 条 NAV / Excess cumulative curves (C1/C2/C3 + R8.5 + R12)
下面板: Sharpe 横评柱图
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.sans-serif": ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial"],
    "font.size": 11, "axes.unicode_minus": False,
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlecolor": "#1a365d",
    "grid.alpha": 0.25, "grid.linestyle": "--",
})

ROOT = Path('/Users/cedricyu/qlib量化研究')
R14 = ROOT / 'results/runs/round14'
FIGS = ROOT / 'results/figures'
FIGS.mkdir(parents=True, exist_ok=True)


def metrics(daily_returns, name=""):
    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float('nan')
    eq = (1 + daily_returns).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(name=name, sharpe=sharpe, ann_ret=ann_ret*100, mdd=mdd*100)


def main():
    print("加载 5 条曲线 ...", flush=True)
    long_leg = pd.read_pickle(R14 / 'long_leg_daily.pkl')

    r85_daily = long_leg['long_return'] - long_leg['bench_return']
    r85_nav = (1 + r85_daily).cumprod()
    m_r85 = metrics(r85_daily, 'R8.5 long-only (净超额)')

    r12_daily = pd.read_pickle(ROOT / 'results/runs/round12/ls_LS-100pct_K_30.pkl')
    r12_nav = (1 + r12_daily).cumprod()
    m_r12 = metrics(r12_daily, 'R12 LS-100% K=30 (hypothetical)')

    r14 = {}
    for n in ['C1_1x_conservative', 'C2_beta_conservative', 'C3_1x_optimistic']:
        df = pd.read_pickle(R14 / f'nav_{n}.pkl')
        daily = df['nav'].pct_change().dropna()
        r14[n] = dict(curve=(df['nav'] / df['nav'].iloc[0]),
                      metrics=metrics(daily, n))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), facecolor='white',
                              gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]

    colors = {
        'C1_1x_conservative': '#1a365d',
        'C2_beta_conservative': '#2c7a7b',
        'C3_1x_optimistic': '#c89b3c',
    }
    for n, payload in r14.items():
        c = payload['curve']
        m = payload['metrics']
        ax.plot(c.index, c.values, label=f"R14 {n}  (Sharpe {m['sharpe']:+.2f})",
                color=colors[n], linewidth=2.2)

    ax.plot(r85_nav.index, r85_nav.values,
            label=f"R8.5 long-only 净超额  (IR {m_r85['sharpe']:+.2f})",
            color='#a0aec0', linestyle='--', linewidth=1.6)
    ax.plot(r12_nav.index, r12_nav.values,
            label=f"R12 LS-100% K=30 (hypothetical)  (Sharpe {m_r12['sharpe']:+.2f})",
            color='#9467bd', linestyle=':', linewidth=1.8)

    ax.set_title('图 R14-1 | 工程版 L/S vs 历史基线 (含 IC margin / basis / 手续费)', loc='left')
    ax.set_ylabel('累计净值 (相对起点)')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0)
    ax.axhline(1.0, color='#cbd5e0', linewidth=0.5)

    ax = axes[1]
    items = [
        ('R8.5\nlong-only', m_r85['sharpe'], '#a0aec0'),
        ('R14 C1\n1x 保守', r14['C1_1x_conservative']['metrics']['sharpe'], '#1a365d'),
        ('R14 C2\nbeta-adj', r14['C2_beta_conservative']['metrics']['sharpe'], '#2c7a7b'),
        ('R14 C3\n1x 乐观', r14['C3_1x_optimistic']['metrics']['sharpe'], '#c89b3c'),
        ('R12\nhypothetical', m_r12['sharpe'], '#9467bd'),
    ]
    names = [x[0] for x in items]; vals = [x[1] for x in items]; cols = [x[2] for x in items]
    ax.bar(names, vals, color=cols, alpha=0.92, edgecolor='white')
    ax.axhline(1.0, color='#2f855a', linestyle=':', alpha=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03 if v >= 0 else v - 0.08, f'{v:.2f}',
                ha='center', fontweight='bold', fontsize=9, color='#1a365d')
    ax.set_title('Sharpe / IR 横评', loc='left')
    ax.set_ylabel('Sharpe / IR')
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig14_engineering_ls.png')
    plt.close()
    print(f"  ✓ {FIGS / 'fig14_engineering_ls.png'}", flush=True)


if __name__ == '__main__':
    main()
