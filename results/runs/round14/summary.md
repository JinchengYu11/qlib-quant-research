# R14 (B1+++) 工程版 long-short — IC 期货对冲

**假设**：把 R12 hypothetical short 换成真实可执行的 CSI500 期货 (IC) 对冲后，alpha 还剩多少？

**实现**：3 个独立 engine 模块（[ic_data](../../../engine/ic_data.py) + [ic_engine](../../../engine/ic_engine.py) + [account](../../../engine/account.py)），跑 3 组配置：

- **C1**: 1x notional + 保守 finance (14% margin / 0% cash / 5% basis)
- **C2**: 60d rolling beta-adjusted + 保守 finance
- **C3**: 1x notional + 乐观 finance (1.5% cash / 3% basis)

测试期 2020-10-01 ~ 2026-05-15（与 R8.5 / R12 一致），long leg 复用 R8.5 模型预测（不重训）。

## 结果

| 配置 | 年化收益 | Vol | Sharpe | MDD | 备注 |
|---|---|---|---|---|---|
| **R12 LS-100% K=30 (hypothetical)** | +7.32% | 10.22% | **0.717** | -14.21% | 项目最高 (含 hypothetical short alpha) |
| **R8.5 long-only 净超额** | +4.90% | 9.83% | **0.498** | -15.70% | long-only 天花板 |
| R14 C3 1x 乐观 | +4.10% | 11.93% | **0.344** | -16.90% | basis 3% / 现金 1.5% |
| R14 C2 beta-adj 保守 | +3.05% | 12.47% | **0.244** | -24.24% | 60d 滚动 beta |
| **R14 C1 1x 保守 (baseline)** | +2.62% | 12.53% | **0.209** | -21.00% | basis 5% / 现金 0% |

**IC margin 占用峰值**：C1 24.78% / C2 21.06% / C3 24.78% of 1 亿初始资金
**Margin call 触发**：C1 1160 / C2 1119 / C3 936 次（研究 log，未强平）
**实测 IC 基差**：测试期内平均 -0.525% (per-day 快照)，按季度展期年化约 **-2.1%**

### 模型局限（已知 NAV 偏差，但 Sharpe 不受影响）

实现里把 IC margin 当作"在我们账户里冻结的资金"加入 NAV，但 1 亿初始资金已经被 qlib
全部投入 long stocks，没有额外现金可"冻结"posting margin。等价于：模型隐式假设了
~14% 的额外资金（实际有 1.14 亿）。

**影响**：
- 绝对年化和 vol 都偏高约 14%（真实约束下 long 只能配 87.7M，绝对收益缩 ~13%）
- **Sharpe 不受影响**（年化和 vol 同比例缩放，比例不变）

R14 的核心结论（Sharpe 排序、basis drag 是 alpha 杀手、R12 突破无法兑现）**均基于 Sharpe**，
所以结论 robust。如果要追求绝对收益精度的工程版，需要在长腿配置 87.7% 资金 + 12.3% margin pool
重跑（预计绝对年化 C1 ≈ +2.0%，仍远不如 R8.5）。

## 关键发现

### 1. R12 的 Sharpe 0.72 几乎全部来自 hypothetical short — 不是 market-neutral 本身

| | R8.5 (long-only) | R14 C1 (real LS) | R12 (hypothetical LS) |
|---|---|---|---|
| Sharpe | 0.498 | 0.209 | 0.717 |

C1 的实盘可执行版 Sharpe **0.21**，**比 R8.5 long-only 还低**（−0.29）。
R12 比 R8.5 高的 0.22 几乎都来自**做空 bottom K 个股**的 reverse-alpha，**用 CSI500 期货空头根本拿不到这部分**。

这是项目首次定量证明：R12 的"突破"在 A 股实盘约束下基本无法实现。

### 2. Basis drag 是工程版的主要 alpha 杀手

C3 (basis 3%) vs C1 (basis 5%) 的 Sharpe 差异：0.34 − 0.21 = **+0.13**，对应约 +1.5pp 年化。
basis drag 每 1pp 大概拖 0.07 Sharpe。

**实测年化 basis ≈ -2.1%**（按季度展期估算）。
即使用实测值替代保守假设，预期 Sharpe 大约在 0.35-0.45 之间，仍**不会超过 R8.5**。

### 3. Beta-adjusted hedge 改善有限

C2 vs C1 Sharpe：0.244 vs 0.209，改善 +0.035。
原因：CSI500 TopK=30 选股组合的 beta vs CSI500 接近 1（A 股 alpha 模型选股暴露主要在风格/行业上，beta 暴露已经接近指数），beta-adjust 边际收益小。

### 4. MDD 比基线差

| | R8.5 | R14 C1 | R12 |
|---|---|---|---|
| MDD | -15.7% | -21.0% | -14.2% |

C1 MDD 比 R8.5 还差 5.3pp。原因：basis drag 在熊市/震荡期会**持续**消耗 PnL，叠加 alpha 不稳，让回撤更深更久。

## 结论

**工程版 long-short 在 A 股不能复现 R12 突破**。具体来说：

- R14 C1 baseline Sharpe = 0.21（保守假设），C3 = 0.34（乐观假设）
- 与 R8.5 long-only 净超额 IR 0.50 相比，工程版 LS 是**降级**而非升级
- R12 hypothetical Sharpe 0.72 的 alpha 约 70% 来自**做空 bottom K 个股的反向选股能力**，市场中性化本身只贡献 ~30%
- 项目的真实"天花板"还是 **R8.5 long-only**（IR 0.45）

## 下一步

A 股 long-short 实盘要进一步推进，可能的方向：

- **做空 bottom K 个股**：必须解决融券难题。可行的是先聚焦"两融标的"子集（约 1000 只），在标的范围内做受限 long-short。预期 Sharpe 0.4-0.6（部分恢复）。
- **跨期对冲优化**：实测 basis 只有 -0.5%/day（年化 ~-2%），不到保守假设的一半。如果用 IC 当月+下月组合对冲，可能再降 basis 成本。
- **R8.5 升级**：继续做 long-only，等 R13 财务因子完成后看能否突破 IR 0.50。这是更现实的方向。

**结论**：本轮"反证"价值大于"突破"价值——它砸碎了 R12 的乐观天花板，确认 R8.5 long-only 才是项目的真实研究产物。

---

## 复现命令

```bash
conda activate qlib

# 一次性: 采 IC 数据 (~30 秒)
python scripts/collect_ic_futures.py

# 跑 long leg (复用 R8.5 模型, ~30 秒)
python scripts/round14_step1_long_leg.py

# 跑 3 组 LS 配置 (~10 秒)
python scripts/round14_step2_engineering_ls.py

# 出图
python scripts/round14_step3_figures.py

# 跑 engine 单元测试
python engine/ic_data.py
python engine/ic_engine.py
python engine/account.py
```

## 文件清单

- `engine/ic_data.py` — IC 日线加载 + 3 个测试
- `engine/ic_engine.py` — IC 头寸引擎 (margin/PnL/basis drag) + 7 个测试
- `engine/account.py` — 资金账户聚合 + 8 个测试
- `scripts/collect_ic_futures.py` — akshare IC 数据采集
- `scripts/round14_step1_long_leg.py` — R8.5 long leg 缓存
- `scripts/round14_step2_engineering_ls.py` — C1/C2/C3 主回测
- `scripts/round14_step3_figures.py` — 可视化
- `data_raw/futures/IC0_daily.csv` — 2017-01 ~ 2026-05, 2270 行
- `data_raw/futures/basis_daily.csv` — 同上, 含 basis_pct 列
- `results/runs/round14/` — long leg cache + 3 个 nav pickle + summary.json + summary.md
- `results/figures/fig14_engineering_ls.png` — 5 条 NAV 曲线 + Sharpe 横评
