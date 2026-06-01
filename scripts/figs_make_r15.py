"""生成 fig15: R15 TabNet vs LGBM 对比 + R8.5 → R15 反证轨迹."""
import warnings; warnings.filterwarnings("ignore")
import json
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
FIGS = ROOT / 'results/figures'


def main():
    # R15 TabNet vs LGBM
    d15 = json.load(open(ROOT / 'results/runs/round15/r15_result.json'))
    lgb = {r['name'].replace('LGBM ', ''): r for r in d15['summary'] if r['name'].startswith('LGBM')}
    tab = {r['name'].replace('TabNet ', ''): r for r in d15['summary'] if r['name'].startswith('TabNet')}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white',
                              gridspec_kw={'width_ratios': [1, 1.2]})

    # === Panel A: LGBM vs TabNet 横评 ===
    ax = axes[0]
    configs = ['long-only', 'LS-100% K=30', 'LS-200% K=30']
    x = np.arange(len(configs))
    width = 0.35
    lgb_sharpe = [lgb[c]['sharpe'] for c in configs]
    tab_sharpe = [tab[c]['sharpe'] for c in configs]
    b1 = ax.bar(x - width/2, lgb_sharpe, width, label='LGBM (0.6 min train)', color='#1a365d', alpha=0.92, edgecolor='white')
    b2 = ax.bar(x + width/2, tab_sharpe, width, label='TabNet (417.5 min train, ×700)', color='#9467bd', alpha=0.92, edgecolor='white')

    for bars, vals in [(b1, lgb_sharpe), (b2, tab_sharpe)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontweight='bold', fontsize=9.5, color='#1a365d')

    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=0, fontsize=9.5)
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    ax.set_ylabel('Sharpe / IR')
    ax.set_title('图 R15-A | TabNet vs LGBM on Alpha158 (same setup)', loc='left')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0)
    ax.set_ylim(0, max(max(lgb_sharpe), max(tab_sharpe)) * 1.20)

    # === Panel B: 项目 7 反证轨迹 (Sharpe 时间序列) ===
    ax = axes[1]
    # 各轮代表 Sharpe
    rounds = [
        ('R8.5\nLGBM\nbaseline', 0.445, '#1a365d', 'baseline (天花板)'),
        ('R9\nensemble', 0.41, '#9aa6b8', '集成模型'),
        ('R9\nind-neutral', 0.40, '#9aa6b8', '行业中性'),
        ('R11\ntopk=50', 0.326, '#9aa6b8', '调 topk'),
        ('R12\nhypothetical\nLS', 0.716, '#2c7a7b', 'hypothetical (R14 揭穿)'),
        ('R13\n+Finance\nfactors', 0.356, '#9aa6b8', '加因子'),
        ('R14\nIC futures\nLS', 0.209, '#9aa6b8', '工程版 LS'),
        ('R15\nTabNet', 0.083, '#c53030', '深度模型'),
    ]
    names = [r[0] for r in rounds]
    vals = [r[1] for r in rounds]
    colors = [r[2] for r in rounds]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.92, edgecolor='white')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8.5, color='#1a365d')

    ax.axhline(0.445, color='#1a365d', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.text(len(names)-0.4, 0.46, 'R8.5 天花板 (0.445)', color='#1a365d',
            fontsize=8.5, ha='right', fontweight='bold')
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, fontsize=8, ha='center')
    ax.set_ylabel('Sharpe / IR')
    ax.set_title('图 R15-B | 7 个改进方向反证轨迹: 全部 ≤ R8.5', loc='left')
    ax.set_ylim(-0.05, 0.80)

    plt.tight_layout()
    out_path = FIGS / 'fig15_deeplearning.png'
    plt.savefig(out_path)
    plt.close()
    print(f"✓ {out_path}")


if __name__ == '__main__':
    main()
