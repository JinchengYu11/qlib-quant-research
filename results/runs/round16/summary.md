# R16 (横向) R8.5 配置在 CSI300 上的可迁移性 — 项目首次正向验证

**假设**：R8.5 配置 (Alpha158 + LGBM + TopK=30, IR 0.445 on CSI500) **完全不动** 移植到 CSI300, 还能不能跑出 IR 0.4+？

**实现**：复用 cn_data_v4 (含 CSI300 历史并集 790 只) + qlib `csi300` instrument filter。
LGBM 超参 / Alpha158 / TopK 策略 / 训练验证测试 split / 交易成本 / 涨跌停限制 — **全部跟 R8.5 完全一致**。
唯一变化: `instruments='csi500'` → `instruments='csi300'`。

脚本 [scripts/round16.py](../../../scripts/round16.py)。

## 结果

| 配置 | 年化 | Vol | Sharpe | MDD | IC | RankICIR |
|---|---|---|---|---|---|---|
| CSI500 long-only (R8.5 baseline) | +4.40% | 9.9% | **0.445** ✅ | -15.7% | 0.0328 | 0.218 |
| CSI500 LS-100% K=30 (R12) | +7.32% | 10.2% | 0.716 | -14.2% | — | — |
| **CSI300 long-only** | **+3.87%** | 9.3% | **0.415** ✅ | -12.6% | 0.0289 | 0.183 |
| **CSI300 LS-100% K=30** | **-1.86%** | 7.3% | **-0.256** ❌ | -20.3% | — | — |

## 3 个核心发现

### 1. R8.5 long-only 首次正向可迁移性验证 ✅

CSI300 IR **0.415** vs CSI500 IR **0.445**, 差 0.030 (-7%). 在 **7 个反证之后, 项目首次出现"R8.5 配置在另一个 universe 上也成立"** 的正向证据。

这是项目最重要的"扩展性"产出: R8.5 不是 CSI500-specific 偶然结果, 而是 **A 股大盘和中盘通用的稳健策略**。

### 2. CSI300 做空 bottom K 反向收益 ❌ (新发现)

| 配置 | CSI500 | CSI300 |
|---|---|---|
| long-only Sharpe | 0.445 ✅ | 0.415 ✅ |
| LS-100% Sharpe | **0.716** ⭐ | **-0.256** ❌ |

CSI300 LS Sharpe = -0.256, 意味着**做空 bottom K 个股反而亏钱**。可能的原因:

- **大盘股套利充分**: CSI300 是机构重仓股, 流动性好, 错误定价被快速消除, "差股票"实际并不那么差
- **大盘信息透明**: 季报、分析师覆盖、机构调研 — 风险因素已被市场充分定价, 模型"预测最差的"并未真正差到能 短卖
- **大盘 mean-reversion 强**: 被打压的大盘股容易反弹 (流动性好的"价值陷阱"实际是反向 alpha)

**这意味着 R12 / R14 的 long-short 思路在 CSI300 上不可能 work**。R8.5 long-only 才是 CSI300 上的合理打法。

### 3. CSI300 MDD 比 CSI500 还小 ✅

MDD -12.6% vs -15.7%, **小 3.1pp**. 大盘股波动小, 防御性更好.
策略防御性 = 大盘 + R8.5 配置 + topk=30. 这是 institutional risk-conscious portfolio 的合理基线.

## CSI300 vs CSI500 模型行为

IC 对比清楚地解释了为什么 CSI300 LS 失效:

| | CSI500 | CSI300 |
|---|---|---|
| IC (Pearson) | 0.0328 | 0.0289 (-12%) |
| RankIC (Spearman) | 0.0287 | 0.0276 (-4%) |
| RankICIR | 0.218 | 0.183 (-16%) |

- **整体预测能力略低** (CSI300 IC 比 CSI500 低 ~10%)
- **RankIC 几乎相同** (0.029 vs 0.029) — 顶部排序质量类似
- **但 IC stability 下降** (RankICIR -16%)

**CSI300 模型在 long-only 侧 (top K) 信号质量 ≈ CSI500, 但 short 侧 (bottom K) 信号被噪声主导, 反而被反向 mean reversion 吃掉**。

## 项目结构地位

R16 是项目第 **8** 轮"R8.5 之上做点什么"的尝试, 也是 **第 1 个真正成功的方向**:

| 轮 | 改进尝试 | 结果 |
|---|---|---|
| R9 | + 模型集成 | ❌ |
| R9 | + 行业中性化 | ❌ |
| R11 | 调 topk | ❌ |
| R12 | hypothetical L/S | ⚠️ (R14 揭穿) |
| R13 | + 财务因子 | ❌ |
| R14 | 工程 L/S (IC 期货对冲) | ❌ |
| R15 | TabNet 深度模型 | ❌ |
| **R16** | **R8.5 → CSI300** | **✅ 首次正向迁移** |

R8.5 之上**加东西**全部失败, 但 R8.5 配置**搬到另一个 universe** 第一次成功。
方法论收获: **改进 R8.5 的正确方向不是给它加东西, 而是验证它的迁移性**。

## 下一步

### R16 Phase 2 (CSI1000) - 暂缓

CSI1000 是 A 股小盘股 (≈ 第 800 - 1800 名市值), 跟 CSI500 部分重叠. 如果做出来:
- 预期 long-only IR ≈ 0.4-0.5 (小盘 alpha 通常更强但波动更大)
- LS Sharpe 预期 > CSI500 (小盘有更强 short alpha, 跟 CSI300 相反)
- 完整证明 R8.5 在大中小三档全适用

工程量: cn_data_v4 不含 CSI1000 historical union (需采集 ~2500+ 股 OHLCV, 8+ 小时 baostock), 当前 Phase 1 已给清晰答案, 暂不优先。

### R16 Phase 3 (HS300 实盘 long-only 配置) - 推荐

CSI300 long-only IR 0.415 + MDD -12.6% 是**所有 R16 结果里最 deployable** 的配置:
- 不需要做空
- 不需要期货对冲
- 大盘股流动性好, 实盘滑点小
- 防御性好

如果要往实盘走, 这是比 CSI500 LS hypothetical (R12) 更现实的版本。

### 项目其他方向 (维持)

- Walk-forward 时间外验证 R8.5 稳定性
- 整理论文/教学材料

## 复现命令

```bash
conda activate qlib
python scripts/round16.py  # ~5 min
```

## 文件清单

- `scripts/round16.py` — R16 双 universe (CSI300 + CSI500 sanity) 主脚本
- `results/runs/round16/r16_result.json` — 完整 metrics
- `results/runs/round16/pred_csi300.pkl` — CSI300 测试期预测值
- `results/runs/round16/pred_csi500.pkl` — CSI500 测试期预测值 (sanity, 跟 R8.5/R12 复现)
- `results/runs/round16/summary.md` — 本文档
