"""第 12 轮 (B1++): 研究版 long-short 回测（修正版）。

用 qlib TopkDropoutStrategy 跑两次:
  - 多头: signal = pred, 选 top K (跟 R8.5 相同机制)
  - 空头: signal = -pred, 选"反 top K" (即原 bottom K), 收益取反

这样两边都有 R8.5 同款的持仓惯性 + n_drop=1 强制换股, 换手自然低。

3 个 gross 配置:
  - LS-100%: 50% long + 50% short (跟 R8.5 同敞口, 但 market-neutral)
  - LS-200%: 100% long + 100% short (2x 杠杆, 工业界 L/S 常配置)
  - LS-100% K=50: 放宽集中度

对比基准: 0 (market-neutral 不用 CSI500 基准)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

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

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round12')
OUT.mkdir(parents=True, exist_ok=True)
FIGS = Path('/Users/cedricyu/qlib量化研究/results/figures')

EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def run_topk_backtest(pred, K, n_drop=1):
    """用 qlib TopkDropoutStrategy 跑回测，返回每日 report"""
    strat = TopkDropoutStrategy(signal=pred, topk=K, n_drop=n_drop, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15', strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    return rep


def metrics(series, name=""):
    ann_ret = series.mean() * 252
    ann_vol = series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    eq = (1 + series).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(name=name, ann_ret=ann_ret*100, ann_vol=ann_vol*100,
                sharpe=sharpe, mdd=mdd*100)


def long_short_combine(rep_long, rep_short_inverse, gross_per_leg):
    """
    rep_long: 多头回测 (signal=pred)
    rep_short_inverse: "假装多头" 但 signal=-pred (实际选了原 bottom K, 但回测当多头跑)

    多头净日收益 = rep_long.return - rep_long.cost
    空头净日收益 = -(rep_short_inverse.return - rep_short_inverse.cost)
      (因为我们实际是做空 bottom K, 它们涨我们亏)

    组合 = gross_per_leg * (long_net + short_net)
    """
    long_net = rep_long['return'] - rep_long['cost']
    short_net = -(rep_short_inverse['return'] - rep_short_inverse['cost'])
    combined = gross_per_leg * (long_net + short_net)
    return combined, long_net, short_net


def main():
    print("[1] 加载 R8.5 预测值 ...", flush=True)
    pred = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    print(f"    {len(pred)} 行", flush=True)

    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')

    print("\n[2] 多头侧: TopkDropout(pred, K=30) ...", flush=True)
    rep_L_30 = run_topk_backtest(pred, K=30)
    print(f"    日均换手 {rep_L_30['turnover'].mean()*100:.1f}%, "
          f"年化收益 (净) {(rep_L_30['return']-rep_L_30['cost']).mean()*252*100:+.2f}%", flush=True)

    print("\n[3] 空头侧: TopkDropout(-pred, K=30) [反向预测当多头跑] ...", flush=True)
    rep_S_30 = run_topk_backtest(-pred, K=30)
    print(f"    日均换手 {rep_S_30['turnover'].mean()*100:.1f}%, "
          f"年化收益 (净, 反向前) {(rep_S_30['return']-rep_S_30['cost']).mean()*252*100:+.2f}%", flush=True)
    print(f"    [若做空, 年化收益 = {-(rep_S_30['return']-rep_S_30['cost']).mean()*252*100:+.2f}%]", flush=True)

    print("\n[4] 跑 K=50 / K=100 多空侧 ...", flush=True)
    rep_L_50 = run_topk_backtest(pred, K=50)
    rep_S_50 = run_topk_backtest(-pred, K=50)
    rep_L_100 = run_topk_backtest(pred, K=100)
    rep_S_100 = run_topk_backtest(-pred, K=100)

    print("\n[5] 组合 4 个 L/S 配置 ...", flush=True)
    configs = [
        ('LS-100% K=30', rep_L_30, rep_S_30, 0.5),
        ('LS-200% K=30', rep_L_30, rep_S_30, 1.0),
        ('LS-100% K=50', rep_L_50, rep_S_50, 0.5),
        ('LS-100% K=100', rep_L_100, rep_S_100, 0.5),
    ]
    summary = []
    all_curves = {}
    for tag, rL, rS, g in configs:
        combined, lnet, snet = long_short_combine(rL, rS, g)
        m = metrics(combined, tag)
        m['ann_long'] = lnet.mean() * 252 * 100
        m['ann_short'] = snet.mean() * 252 * 100
        summary.append(m)
        all_curves[tag] = combined
        print(f"  {tag:18s}: 年化 {m['ann_ret']:+6.2f}%  Vol {m['ann_vol']:5.2f}%  "
              f"Sharpe {m['sharpe']:+.3f}  MDD {m['mdd']:+.1f}%", flush=True)

    # 对比 R8.5 (long-only)
    long_only_85 = rep_L_30['return'] - rep_L_30['cost'] - rep_L_30['bench']
    m_85 = metrics(long_only_85, 'R8.5 long-only (净超额)')
    print(f"\n  {m_85['name']:18s}: 年化 {m_85['ann_ret']:+6.2f}%  Vol {m_85['ann_vol']:5.2f}%  "
          f"Sharpe {m_85['sharpe']:+.3f} (= IR)  MDD {m_85['mdd']:+.1f}%", flush=True)

    # ===== 保存 =====
    json.dump({'configs': summary, 'r85_long_only_excess': m_85},
              open(OUT / 'longshort_summary.json', 'w'), indent=1, default=str)
    for tag, curve in all_curves.items():
        curve.to_pickle(OUT / f"ls_{tag.replace('%','pct').replace(' ','_').replace('=','_')}.pkl")

    # ===== 可视化 =====
    print("\n[6] 生成图表 ...", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), facecolor='white',
                              gridspec_kw={'height_ratios':[2, 1]})

    ax = axes[0]
    colors_map = {
        'LS-100% K=30': '#1a365d',
        'LS-200% K=30': '#c89b3c',
        'LS-100% K=50': '#2c7a7b',
        'LS-100% K=100': '#9467bd',
    }
    for tag, curve in all_curves.items():
        eq = (1 + curve).cumprod()
        m = next(x for x in summary if x['name'] == tag)
        ax.plot(eq.index, eq.values, label=f"{tag}  (Sharpe {m['sharpe']:+.2f})",
                color=colors_map.get(tag), linewidth=2.2)
    # 加 R8.5 长只对比 (净超额)
    eq_85 = (1 + long_only_85).cumprod()
    ax.plot(eq_85.index, eq_85.values, label=f"R8.5 long-only 净超额  (IR {m_85['sharpe']:.2f})",
            color='#a0aec0', linestyle='--', linewidth=1.6)

    ax.set_title('图 R12-1 | Long-short 研究版回测 (qlib TopkDropout 机制, 含成本)', loc='left')
    ax.set_ylabel('累计净值')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0)
    ax.axhline(1.0, color='#cbd5e0', linewidth=0.5)

    ax = axes[1]
    names = [m['name'] for m in summary] + [m_85['name'].replace('(净超额)','')]
    sharpes = [m['sharpe'] for m in summary] + [m_85['sharpe']]
    cols = [colors_map[m['name']] for m in summary] + ['#a0aec0']
    bars = ax.bar(names, sharpes, color=cols, alpha=0.92, edgecolor='white')
    ax.axhline(1.0, color='#2f855a', linestyle=':', alpha=0.6)
    ax.axhline(1.5, color='#c89b3c', linestyle=':', alpha=0.6)
    ax.text(len(names)-0.5, 1.02, '可用 (1.0)', color='#2f855a', fontsize=8, ha='right')
    ax.text(len(names)-0.5, 1.52, '较好 (1.5)', color='#c89b3c', fontsize=8, ha='right')
    for i, s in enumerate(sharpes):
        ax.text(i, s + 0.05 if s >= 0 else s - 0.1, f'{s:.2f}',
                ha='center', fontweight='bold', fontsize=9, color='#1a365d')
    ax.set_title('Sharpe / IR 横评', loc='left')
    ax.set_ylabel('Sharpe / IR')
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=12, ha='right', fontsize=9.5)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig13_longshort.png')
    plt.close()
    print(f"  ✓ fig13_longshort.png", flush=True)

    print(f"\n[DONE] 结果存 results/runs/round12/", flush=True)


if __name__ == '__main__':
    main()
