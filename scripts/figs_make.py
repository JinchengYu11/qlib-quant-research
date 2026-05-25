"""阶段2：用准备好的日序列数据生成 10 张图表 + 1 张拼接合集图。
输出到 results/figures/。"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

# 中文字体
mpl.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.size"] = 11

DATA = Path("/Users/cedricyu/qlib量化研究/results/figures_data")
OUT = Path("/Users/cedricyu/qlib量化研究/results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# 5 个配置的标签
CONFIGS = [
    ("R3_CSI300_baseline", "CSI300 R3 (n_drop=1)", "#1f77b4"),
    ("R6_CSI500biased_default", "CSI500 含偏 R6 (默认)", "#ff7f0e"),
    ("R7_CSI500biased_tuned", "CSI500 含偏 R7 (调优)", "#d62728"),
    ("R8_CSI500debiased_tunedhp", "CSI500 去偏 R8 (调优套用)", "#9467bd"),
    ("R85_CSI500debiased_FINAL", "★ 最终 CSI500 去偏 R8.5 (默认+topk30)", "#2ca02c"),
]


def load_report(tag):
    return pd.read_pickle(DATA / f"{tag}_report.pkl")


def net_excess_series(rep):
    return rep["return"] - rep["cost"] - rep["bench"]


def cum(s, start=1.0):
    return start * (1 + s).cumprod()


# ====================================================================
# 图 1: 累计净值对比
# ====================================================================
def fig1_cumulative():
    fig, ax = plt.subplots(figsize=(14, 6))
    # 5 个策略的累计净值（用 return - cost）
    for tag, label, color in CONFIGS:
        rep = load_report(tag)
        net = rep["return"] - rep["cost"]
        eq = cum(net)
        ax.plot(eq.index, eq.values, label=label, color=color,
                linewidth=2.5 if "★" in label else 1.5,
                alpha=1.0 if "★" in label else 0.8)
    # 两条基准
    rep_c300 = load_report("R3_CSI300_baseline")
    rep_c500 = load_report("R85_CSI500debiased_FINAL")
    ax.plot(rep_c300.index, cum(rep_c300["bench"]), label="CSI300 指数(基准)",
            color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.plot(rep_c500.index, cum(rep_c500["bench"]), label="CSI500 指数(基准)",
            color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_title("图1: 项目演进 - 累计净值对比 (5.4 年 含成本)", fontsize=14, fontweight="bold")
    ax.set_ylabel("累计净值 (起始 1.0)")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.axhline(1.0, color="gray", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_cumulative_net.png")
    plt.close()
    print("✓ fig1_cumulative_net.png")


# ====================================================================
# 图 2: 逐年净超额
# ====================================================================
def fig2_yearly_bars():
    fig, ax = plt.subplots(figsize=(14, 6))
    cfgs = [(t, l, c) for t, l, c in CONFIGS if "R3" in t or "R6" in t or "R7" in t or "R85" in t]
    years = list(range(2020, 2027))
    width = 0.2
    x = np.arange(len(years))
    for i, (tag, label, color) in enumerate(cfgs):
        rep = load_report(tag)
        ex_net = net_excess_series(rep)
        yearly = {int(y): g.sum() * 100 for y, g in ex_net.groupby(ex_net.index.year)}
        vals = [yearly.get(y, 0) for y in years]
        ax.bar(x + i * width, vals, width, label=label, color=color,
               alpha=1.0 if "★" in label else 0.85)
    ax.set_xticks(x + width * (len(cfgs) - 1) / 2)
    ax.set_xticklabels(years)
    ax.set_ylabel("当年净超额 (%)")
    ax.set_title("图2: 逐年净超额对比 (策略-基准)", fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_yearly_excess.png")
    plt.close()
    print("✓ fig2_yearly_excess.png")


# ====================================================================
# 图 3: 多指标横评柱状图（3 panel）
# ====================================================================
def fig3_summary():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    labels, colors = [], []
    excess, ir, sharpe = [], [], []
    for tag, label, color in CONFIGS:
        rep = load_report(tag)
        net = rep["return"] - rep["cost"]
        ex = net - rep["bench"]
        labels.append(label.replace("★ 最终 ", "").replace("CSI", "C"))
        colors.append(color)
        excess.append(ex.mean() * 252 * 100)
        ir.append((ex.mean() * 252) / (ex.std() * np.sqrt(252)))
        sharpe.append((net.mean() * 252) / (net.std() * np.sqrt(252)))

    for ax, vals, title, ylabel in [
        (axes[0], excess, "年化净超额 (%)", "%"),
        (axes[1], ir, "信息比率 (IR)", ""),
        (axes[2], sharpe, "净夏普", ""),
    ]:
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        # 标星最优
        best = np.argmax(vals)
        bars[best].set_edgecolor("red")
        bars[best].set_linewidth(2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.02 * max(abs(min(vals)), abs(max(vals)))) * (1 if v >= 0 else -1),
                    f"{v:+.2f}" if title.startswith("年化") else f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("图3: 项目五个关键配置 - 三大指标横评", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_summary_bars.png")
    plt.close()
    print("✓ fig3_summary_bars.png")


# ====================================================================
# 图 4: 幸存者偏差可视化 (含偏 vs 去偏)
# ====================================================================
def fig4_bias():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    rep_biased = load_report("R7_CSI500biased_tuned")
    rep_debiased = load_report("R8_CSI500debiased_tunedhp")
    # 同样的超参 + n_drop=1，唯一差别 = 数据集（含/去偏）

    # 上：累计净值
    ax = axes[0]
    eq_b = cum(rep_biased["return"] - rep_biased["cost"])
    eq_d = cum(rep_debiased["return"] - rep_debiased["cost"])
    bench = cum(rep_biased["bench"])
    ax.plot(eq_b.index, eq_b.values, label="v3 含偏 (当前 500 成分)",
            color="#d62728", linewidth=2.2)
    ax.plot(eq_d.index, eq_d.values, label="v4 去偏 (历史并集 1623 只)",
            color="#2ca02c", linewidth=2.2)
    ax.plot(bench.index, bench.values, label="CSI500 基准", color="gray",
            linestyle="--", linewidth=1.2)
    ax.fill_between(eq_b.index, eq_b.values, eq_d.values,
                    where=(eq_b.values > eq_d.values), color="red", alpha=0.15,
                    label="↑ 幸存者偏差虚高")
    ax.set_ylabel("累计净值")
    ax.set_title("图4: 幸存者偏差可视化 - 同模型同策略，仅数据集不同", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="gray", linewidth=0.5)

    # 下：净超额累计差距
    ax = axes[1]
    ex_b = (rep_biased["return"] - rep_biased["cost"]) - rep_biased["bench"]
    ex_d = (rep_debiased["return"] - rep_debiased["cost"]) - rep_debiased["bench"]
    cum_ex_b = ex_b.cumsum() * 100
    cum_ex_d = ex_d.cumsum() * 100
    gap = cum_ex_b - cum_ex_d
    ax.plot(cum_ex_b.index, cum_ex_b.values, label="v3 含偏 累计超额",
            color="#d62728", linewidth=2)
    ax.plot(cum_ex_d.index, cum_ex_d.values, label="v4 去偏 累计超额",
            color="#2ca02c", linewidth=2)
    ax.fill_between(cum_ex_b.index, cum_ex_b.values, cum_ex_d.values,
                    color="red", alpha=0.15)
    final_gap = gap.iloc[-1]
    ax.annotate(f"虚高累计 {final_gap:.0f} 个百分点",
                xy=(cum_ex_b.index[-1], cum_ex_b.iloc[-1]),
                xytext=(-180, -20), textcoords="offset points",
                fontsize=11, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_ylabel("累计净超额 (%)")
    ax.set_title("含偏 vs 去偏 - 累计超额收益差距")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_bias_viz.png")
    plt.close()
    print("✓ fig4_bias_viz.png")


# ====================================================================
# 图 5: 最终配置月度热力图
# ====================================================================
def fig5_heatmap():
    rep = load_report("R85_CSI500debiased_FINAL")
    net = rep["return"] - rep["cost"]
    df = pd.DataFrame({"net": net, "ex": net - rep["bench"]})
    df["year"] = df.index.year
    df["month"] = df.index.month
    monthly_ex = df.groupby(["year", "month"])["ex"].sum().unstack() * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    vmax = max(abs(monthly_ex.min().min()), abs(monthly_ex.max().max()))
    im = ax.imshow(monthly_ex.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
    ax.set_yticks(range(len(monthly_ex.index)))
    ax.set_yticklabels(monthly_ex.index)
    for i in range(monthly_ex.shape[0]):
        for j in range(monthly_ex.shape[1]):
            v = monthly_ex.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=9, color="black" if abs(v) < vmax * 0.5 else "white",
                        fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("月度净超额 (%)")
    ax.set_title("图5: 最终配置月度净超额热力图 (CSI500 去偏 R8.5)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig5_monthly_heatmap.png")
    plt.close()
    print("✓ fig5_monthly_heatmap.png")


# ====================================================================
# 图 6: 60 日滚动 IR 和滚动夏普
# ====================================================================
def fig6_rolling():
    rep = load_report("R85_CSI500debiased_FINAL")
    net = rep["return"] - rep["cost"]
    ex = net - rep["bench"]
    win = 60
    roll_ir = (ex.rolling(win).mean() * 252) / (ex.rolling(win).std() * np.sqrt(252))
    roll_sp = (net.rolling(win).mean() * 252) / (net.rolling(win).std() * np.sqrt(252))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(roll_ir.index, roll_ir.values, label="滚动信息比率 (IR)",
            color="#2ca02c", linewidth=1.8)
    ax.plot(roll_sp.index, roll_sp.values, label="滚动净夏普",
            color="#1f77b4", linewidth=1.8, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="green", linestyle=":", alpha=0.5, label="\"可用\" 线 (>1)")
    ax.fill_between(roll_ir.index, 0, roll_ir.values,
                    where=(roll_ir.values > 0), color="green", alpha=0.1)
    ax.fill_between(roll_ir.index, 0, roll_ir.values,
                    where=(roll_ir.values < 0), color="red", alpha=0.1)
    ax.set_ylabel(f"{win} 日滚动指标")
    ax.set_title(f"图6: 最终配置 {win} 日滚动 IR / 净夏普 - 看时间稳定性",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_rolling_ir.png")
    plt.close()
    print("✓ fig6_rolling_ir.png")


# ====================================================================
# 图 7: 水下回撤图 (underwater drawdown)
# ====================================================================
def fig7_underwater():
    rep = load_report("R85_CSI500debiased_FINAL")
    net = rep["return"] - rep["cost"]
    ex = net - rep["bench"]
    eq_net = cum(net)
    eq_ex = cum(ex)
    dd_net = (eq_net / eq_net.cummax() - 1) * 100
    dd_ex = (eq_ex / eq_ex.cummax() - 1) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd_net.index, dd_net.values, 0, color="#1f77b4", alpha=0.4,
                    label=f"策略净回撤 (max {dd_net.min():.1f}%)")
    ax.fill_between(dd_ex.index, dd_ex.values, 0, color="#2ca02c", alpha=0.5,
                    label=f"超额回撤 (max {dd_ex.min():.1f}%)")
    ax.plot(dd_net.index, dd_net.values, color="#1f77b4", linewidth=1)
    ax.plot(dd_ex.index, dd_ex.values, color="#2ca02c", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("距历史新高的距离 (%)")
    ax.set_title("图7: 最终配置水下回撤图 - 距离历史新高的距离",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_underwater.png")
    plt.close()
    print("✓ fig7_underwater.png")


# ====================================================================
# 图 8: 因子重要性 Top 20
# ====================================================================
def fig8_importance():
    imp = pd.read_pickle(DATA / "R3_CSI300_baseline_imp.pkl").head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(imp))[::-1], imp.values, color="#1f77b4", alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(imp))[::-1])
    ax.set_yticklabels(imp.index)
    ax.set_xlabel("LightGBM gain (越大越重要)")
    ax.set_title("图8: LightGBM 因子重要性 Top 20 (CSI300 R3 模型)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(imp.values):
        ax.text(v, len(imp) - i - 1, f" {v:.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig8_factor_importance.png")
    plt.close()
    print("✓ fig8_factor_importance.png")


# ====================================================================
# 图 9: IC 时间序列 (累计+日 + 信号生命力)
# ====================================================================
def fig9_ic():
    ic = pd.read_pickle(DATA / "R85_CSI500debiased_FINAL_ic.pkl")
    ric = pd.read_pickle(DATA / "R85_CSI500debiased_FINAL_ric.pkl")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    # 上：累计 IC
    ax = axes[0]
    ax.plot(ic.index, ic.cumsum().values, label=f"累计 IC (均值 {ic.mean():.4f})",
            color="#2ca02c", linewidth=2)
    ax.plot(ric.index, ric.cumsum().values, label=f"累计 Rank IC (均值 {ric.mean():.4f})",
            color="#1f77b4", linewidth=2)
    ax.set_title("图9: 最终配置 IC 时间序列 - 信号生命力可视化",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("累计 IC")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    # 下：日 IC 柱状（绿涨红跌）
    ax = axes[1]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in ic.values]
    ax.bar(ic.index, ic.values, color=colors, alpha=0.7, width=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("日 IC")
    pos_rate = (ic > 0).mean() * 100
    ax.set_title(f"日 IC 柱状 - 正比例 {pos_rate:.1f}%")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig9_ic_timeseries.png")
    plt.close()
    print("✓ fig9_ic_timeseries.png")


# ====================================================================
# 图 10: 策略相关性矩阵
# ====================================================================
def fig10_correlation():
    rets = {}
    for tag, label, _ in CONFIGS:
        rep = load_report(tag)
        rets[label.replace("★ 最终 ", "").replace("CSI", "C")] = rep["return"] - rep["cost"]
    # 基准
    rets["C500 指数(基准)"] = load_report("R85_CSI500debiased_FINAL")["bench"]
    rets["C300 指数(基准)"] = load_report("R3_CSI300_baseline")["bench"]
    df = pd.DataFrame(rets).dropna()
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(corr.index, fontsize=8.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("图10: 各配置日收益相关性矩阵", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig10_correlation.png")
    plt.close()
    print("✓ fig10_correlation.png")


# ====================================================================
# 主控
# ====================================================================
if __name__ == "__main__":
    print(f"\n=== 生成图表到 {OUT} ===\n")
    fig1_cumulative()
    fig2_yearly_bars()
    fig3_summary()
    fig4_bias()
    fig5_heatmap()
    fig6_rolling()
    fig7_underwater()
    fig8_importance()
    fig9_ic()
    fig10_correlation()
    print(f"\n[DONE] 10 张图已存到 {OUT}")
