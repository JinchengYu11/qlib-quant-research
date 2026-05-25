"""阶段2 v2：publication-quality 图表，参考头部量化研究风格。
统一配色 (navy + gold + teal)，高 DPI，干净排版。
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 中文 + publication-quality 默认设置
mpl.rcParams.update({
    "font.sans-serif": ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#4a5568",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlecolor": "#1a365d",
    "axes.titlepad": 10,
    "axes.labelcolor": "#4a5568",
    "axes.labelsize": 10,
    "xtick.color": "#4a5568",
    "ytick.color": "#4a5568",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.color": "#a0aec0",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})

# 配色板 (参考 Bridgewater/AQR 配色)
NAVY = "#1a365d"
GOLD = "#c89b3c"
TEAL = "#2c7a7b"
RED = "#c53030"
GREEN = "#2f855a"
ORANGE = "#dd6b20"
PURPLE = "#6b46c1"
GRAY = "#718096"
LIGHT_RED = "#fc8181"
LIGHT_GREEN = "#9ae6b4"
LIGHT_NAVY = "#4299e1"

DATA = Path("/Users/cedricyu/qlib量化研究/results/figures_data")
OUT = Path("/Users/cedricyu/qlib量化研究/results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# 5 个配置 - 更专业的命名
CONFIGS = [
    ("R3_CSI300_baseline", "CSI300 基线 (R3)", GRAY),
    ("R6_CSI500biased_default", "CSI500 含偏-默认 (R6)", LIGHT_NAVY),
    ("R7_CSI500biased_tuned", "CSI500 含偏-调优 (R7)", RED),
    ("R8_CSI500debiased_tunedhp", "CSI500 去偏-调优套用 (R8)", PURPLE),
    ("R85_CSI500debiased_FINAL", "CSI500 去偏-最终 (R8.5) ★", NAVY),
]


def load_report(tag):
    return pd.read_pickle(DATA / f"{tag}_report.pkl")


def cum(s, start=1.0):
    return start * (1 + s).cumprod()


def style_axis(ax, ylabel=None, xlabel=None):
    if ylabel: ax.set_ylabel(ylabel, color="#4a5568")
    if xlabel: ax.set_xlabel(xlabel, color="#4a5568")
    ax.tick_params(colors="#4a5568")


def fig1_cumulative():
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")
    for tag, label, color in CONFIGS:
        rep = load_report(tag)
        net = rep["return"] - rep["cost"]
        eq = cum(net)
        is_final = "★" in label
        ax.plot(eq.index, eq.values, label=label, color=color,
                linewidth=2.8 if is_final else 1.6,
                alpha=1.0 if is_final else 0.75,
                zorder=10 if is_final else 5)
    rep_c300 = load_report("R3_CSI300_baseline")
    rep_c500 = load_report("R85_CSI500debiased_FINAL")
    ax.plot(rep_c300.index, cum(rep_c300["bench"]), label="CSI300 指数",
            color="#cbd5e0", linestyle=":", linewidth=1.5)
    ax.plot(rep_c500.index, cum(rep_c500["bench"]), label="CSI500 指数",
            color="#a0aec0", linestyle="--", linewidth=1.5)
    ax.set_title("图 1 | 项目演进 - 累计净值对比（含交易成本）", loc="left")
    ax.text(0, 1.06, "5.4 年测试期 2020-10 ~ 2026-05 | 起始净值 1.00",
            transform=ax.transAxes, fontsize=10, color="#718096")
    style_axis(ax, ylabel="累计净值")
    ax.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0)
    ax.axhline(1.0, color="#cbd5e0", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_cumulative_net.png")
    plt.close()
    print("✓ fig1")


def fig2_yearly():
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    cfgs = [(t, l, c) for t, l, c in CONFIGS if any(x in t for x in ["R3", "R6", "R7", "R85"])]
    years = list(range(2020, 2027))
    width = 0.2
    x = np.arange(len(years))
    for i, (tag, label, color) in enumerate(cfgs):
        rep = load_report(tag)
        ex_net = rep["return"] - rep["cost"] - rep["bench"]
        yearly = {int(y): g.sum() * 100 for y, g in ex_net.groupby(ex_net.index.year)}
        vals = [yearly.get(y, 0) for y in years]
        is_final = "★" in label
        ax.bar(x + i * width, vals, width, label=label, color=color,
               alpha=1.0 if is_final else 0.85,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(cfgs) - 1) / 2)
    ax.set_xticklabels(years, color="#4a5568")
    ax.set_title("图 2 | 逐年净超额对比", loc="left")
    ax.text(0, 1.04, "策略净收益 − 基准收益（CSI300 或 CSI500 指数对应）",
            transform=ax.transAxes, fontsize=10, color="#718096")
    style_axis(ax, ylabel="当年净超额 (%)")
    ax.axhline(0, color="#4a5568", linewidth=0.6)
    ax.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_yearly_excess.png")
    plt.close()
    print("✓ fig2")


def fig3_summary():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    labels, colors = [], []
    excess, ir, sharpe = [], [], []
    for tag, label, color in CONFIGS:
        rep = load_report(tag)
        net = rep["return"] - rep["cost"]
        ex = net - rep["bench"]
        labels.append(label.replace(" ★", "").replace("基线 ", "").replace("默认 ", "").replace("调优 ", "调优").replace("调优套用 ", "调优套"))
        colors.append(color)
        excess.append(ex.mean() * 252 * 100)
        ir.append((ex.mean() * 252) / (ex.std() * np.sqrt(252)))
        sharpe.append((net.mean() * 252) / (net.std() * np.sqrt(252)))
    metric_specs = [
        (axes[0], excess, "净年化超额", "%", "+.2f"),
        (axes[1], ir, "信息比率", "", ".3f"),
        (axes[2], sharpe, "净夏普", "", ".3f"),
    ]
    for ax, vals, title, unit, fmt in metric_specs:
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.92,
                      edgecolor="white", linewidth=1)
        best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor(GOLD)
        bars[best_idx].set_linewidth(2.5)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.5, color="#4a5568")
        style_axis(ax)
        ax.axhline(0, color="#4a5568", linewidth=0.5)
        for i, v in enumerate(vals):
            txt = format(v, fmt)
            ax.text(i, v + (0.03 * max(abs(min(vals)), abs(max(vals)))) * (1 if v >= 0 else -1.5),
                    txt, ha="center", fontsize=9, fontweight="bold",
                    color=NAVY if i == best_idx else "#4a5568")
    fig.suptitle("图 3 | 项目五个关键配置三大指标横评（金色描边=最优）",
                 fontsize=13, fontweight="bold", color=NAVY, x=0.05, y=1.0, ha="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_summary_bars.png")
    plt.close()
    print("✓ fig3")


def fig4_bias():
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), facecolor="white",
                             gridspec_kw={"height_ratios": [2, 1.3]})
    rep_b = load_report("R7_CSI500biased_tuned")
    rep_d = load_report("R8_CSI500debiased_tunedhp")
    # 上：累计净值
    ax = axes[0]
    eq_b = cum(rep_b["return"] - rep_b["cost"])
    eq_d = cum(rep_d["return"] - rep_d["cost"])
    bench = cum(rep_b["bench"])
    ax.plot(eq_b.index, eq_b.values, label="v3 含偏（当前 500 成分）",
            color=RED, linewidth=2.6)
    ax.plot(eq_d.index, eq_d.values, label="v4 去偏（历史并集 1623 只）",
            color=GREEN, linewidth=2.6)
    ax.plot(bench.index, bench.values, label="CSI500 指数",
            color="#a0aec0", linestyle="--", linewidth=1.4)
    ax.fill_between(eq_b.index, eq_b.values, eq_d.values,
                    where=(eq_b.values > eq_d.values), color=RED, alpha=0.12,
                    label="幸存者偏差虚高区")
    ax.set_title("图 4 | 幸存者偏差视觉化 — 同模型同策略，仅数据集不同", loc="left")
    style_axis(ax, ylabel="累计净值")
    ax.legend(loc="upper left", fontsize=10, framealpha=0)
    ax.axhline(1.0, color="#cbd5e0", linewidth=0.6)
    # 下：净超额累计
    ax = axes[1]
    ex_b = ((rep_b["return"] - rep_b["cost"]) - rep_b["bench"]).cumsum() * 100
    ex_d = ((rep_d["return"] - rep_d["cost"]) - rep_d["bench"]).cumsum() * 100
    ax.plot(ex_b.index, ex_b.values, label="v3 含偏 累计超额",
            color=RED, linewidth=2.2)
    ax.plot(ex_d.index, ex_d.values, label="v4 去偏 累计超额",
            color=GREEN, linewidth=2.2)
    ax.fill_between(ex_b.index, ex_b.values, ex_d.values, color=RED, alpha=0.12)
    gap = (ex_b - ex_d).iloc[-1]
    ax.annotate(f"虚高累计 {gap:.0f} pp\n(年化 {gap/5.4:.1f} pp)",
                xy=(ex_b.index[-1], (ex_b.iloc[-1] + ex_d.iloc[-1]) / 2),
                xytext=(-200, 0), textcoords="offset points",
                fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    style_axis(ax, ylabel="累计净超额 (%)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0)
    ax.axhline(0, color="#4a5568", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_bias_viz.png")
    plt.close()
    print("✓ fig4")


def fig5_heatmap():
    rep = load_report("R85_CSI500debiased_FINAL")
    ex = rep["return"] - rep["cost"] - rep["bench"]
    df = pd.DataFrame({"ex": ex})
    df["year"] = df.index.year
    df["month"] = df.index.month
    m = df.groupby(["year", "month"])["ex"].sum().unstack() * 100
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
    vmax = max(abs(m.min().min()), abs(m.max().max()))
    im = ax.imshow(m.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{i}月" for i in range(1, 13)], color="#4a5568")
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels(m.index, color="#4a5568")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(v) < vmax * 0.5 else "white",
                        fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("月度净超额 (%)", color="#4a5568")
    cbar.ax.tick_params(colors="#4a5568")
    ax.set_title("图 5 | 最终配置月度净超额热力图 (CSI500 去偏 R8.5)", loc="left", pad=15)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_monthly_heatmap.png")
    plt.close()
    print("✓ fig5")


def fig6_rolling():
    rep = load_report("R85_CSI500debiased_FINAL")
    net = rep["return"] - rep["cost"]
    ex = net - rep["bench"]
    win = 60
    roll_ir = (ex.rolling(win).mean() * 252) / (ex.rolling(win).std() * np.sqrt(252))
    roll_sp = (net.rolling(win).mean() * 252) / (net.rolling(win).std() * np.sqrt(252))
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    ax.plot(roll_ir.index, roll_ir.values, label="滚动信息比率 IR", color=NAVY, linewidth=2)
    ax.plot(roll_sp.index, roll_sp.values, label="滚动净夏普", color=GOLD, linewidth=2, alpha=0.85)
    ax.axhline(0, color="#4a5568", linewidth=0.5)
    ax.axhline(1, color=GREEN, linestyle=":", linewidth=1, alpha=0.7, label="“可用”线 (>1)")
    ax.fill_between(roll_ir.index, 0, roll_ir.values,
                    where=(roll_ir.values > 0), color=GREEN, alpha=0.08)
    ax.fill_between(roll_ir.index, 0, roll_ir.values,
                    where=(roll_ir.values < 0), color=RED, alpha=0.08)
    ax.set_title(f"图 6 | 最终配置 {win} 日滚动 IR / 净夏普 — 时间稳定性", loc="left")
    style_axis(ax, ylabel=f"{win} 日滚动指标")
    ax.legend(loc="upper right", fontsize=10, framealpha=0)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_rolling_ir.png")
    plt.close()
    print("✓ fig6")


def fig7_underwater():
    rep = load_report("R85_CSI500debiased_FINAL")
    net = rep["return"] - rep["cost"]
    ex = net - rep["bench"]
    eq_n = cum(net); eq_e = cum(ex)
    dd_n = (eq_n / eq_n.cummax() - 1) * 100
    dd_e = (eq_e / eq_e.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor="white")
    ax.fill_between(dd_n.index, dd_n.values, 0, color=NAVY, alpha=0.4,
                    label=f"策略净回撤  最大 {dd_n.min():.1f}%")
    ax.fill_between(dd_e.index, dd_e.values, 0, color=GREEN, alpha=0.55,
                    label=f"超额回撤    最大 {dd_e.min():.1f}%")
    ax.plot(dd_n.index, dd_n.values, color=NAVY, linewidth=1.2)
    ax.plot(dd_e.index, dd_e.values, color=GREEN, linewidth=1.2)
    ax.axhline(0, color="#4a5568", linewidth=0.5)
    ax.set_title("图 7 | 最终配置水下回撤图 — 距离历史新高的距离", loc="left")
    style_axis(ax, ylabel="回撤 (%)")
    ax.legend(loc="lower left", fontsize=10, framealpha=0)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_underwater.png")
    plt.close()
    print("✓ fig7")


def fig8_importance():
    imp = pd.read_pickle(DATA / "R3_CSI300_baseline_imp.pkl").head(20)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    colors = [NAVY if i < 5 else (GOLD if i < 10 else GRAY) for i in range(len(imp))]
    ax.barh(range(len(imp))[::-1], imp.values, color=colors,
            edgecolor="white", linewidth=0.5, alpha=0.92)
    ax.set_yticks(range(len(imp))[::-1])
    ax.set_yticklabels(imp.index, color="#4a5568")
    ax.set_title("图 8 | LightGBM 因子重要性 Top 20 (CSI300 R3 模型)", loc="left")
    ax.text(0, 1.04, "颜色：深蓝 Top 5 / 金色 6-10 / 灰色 11-20",
            transform=ax.transAxes, fontsize=9.5, color="#718096")
    style_axis(ax, xlabel="LightGBM gain")
    for i, v in enumerate(imp.values):
        ax.text(v, len(imp) - i - 1, f" {v:.0f}", va="center", fontsize=9, color="#4a5568")
    plt.tight_layout()
    plt.savefig(OUT / "fig8_factor_importance.png")
    plt.close()
    print("✓ fig8")


def fig9_ic():
    ic = pd.read_pickle(DATA / "R85_CSI500debiased_FINAL_ic.pkl")
    ric = pd.read_pickle(DATA / "R85_CSI500debiased_FINAL_ric.pkl")
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, facecolor="white",
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(ic.index, ic.cumsum().values, label=f"累计 IC  (均值 {ic.mean():.4f})",
            color=NAVY, linewidth=2.2)
    ax.plot(ric.index, ric.cumsum().values, label=f"累计 Rank IC  (均值 {ric.mean():.4f})",
            color=GOLD, linewidth=2.2)
    ax.set_title("图 9 | 最终配置 IC 时间序列 — 信号生命力", loc="left")
    style_axis(ax, ylabel="累计 IC")
    ax.legend(loc="upper left", fontsize=10, framealpha=0)
    ax.axhline(0, color="#4a5568", linewidth=0.5)
    ax = axes[1]
    colors = [GREEN if v > 0 else RED for v in ic.values]
    ax.bar(ic.index, ic.values, color=colors, alpha=0.7, width=2)
    ax.axhline(0, color="#4a5568", linewidth=0.5)
    style_axis(ax, ylabel="日 IC")
    pos_rate = (ic > 0).mean() * 100
    ax.set_title(f"日 IC 柱状  ·  正比例 {pos_rate:.1f}%", loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig9_ic_timeseries.png")
    plt.close()
    print("✓ fig9")


def fig10_correlation():
    rets = {}
    for tag, label, _ in CONFIGS:
        rep = load_report(tag)
        rets[label.replace(" ★", "").replace("基线 ", "").replace("默认 ", "")] = rep["return"] - rep["cost"]
    rets["CSI500 指数"] = load_report("R85_CSI500debiased_FINAL")["bench"]
    rets["CSI300 指数"] = load_report("R3_CSI300_baseline")["bench"]
    df = pd.DataFrame(rets).dropna()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=42, ha="right", fontsize=9, color="#4a5568")
    ax.set_yticklabels(corr.index, fontsize=9, color="#4a5568")
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=9,
                    fontweight="bold" if i == j else "normal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("相关系数", color="#4a5568")
    cbar.ax.tick_params(colors="#4a5568")
    ax.set_title("图 10 | 各配置日收益相关性矩阵", loc="left", pad=15)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig10_correlation.png")
    plt.close()
    print("✓ fig10")


if __name__ == "__main__":
    print("=== 生成 publication-quality 图表 ===\n")
    fig1_cumulative()
    fig2_yearly()
    fig3_summary()
    fig4_bias()
    fig5_heatmap()
    fig6_rolling()
    fig7_underwater()
    fig8_importance()
    fig9_ic()
    fig10_correlation()
    print(f"\n[DONE] 10 张图已升级 → {OUT}")
