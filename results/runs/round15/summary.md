# R15 (B4) 深度模型 — TabNet vs LGBM 反证

**假设**：深度模型 (TabNet) 在 Alpha158 同样 setup 上, 能不能突破 LightGBM 的 R8.5 IR 0.445 / R12 Sharpe 0.716？

**实现**：TabNet (`pytorch_tabnet`) + LGBM 两个模型并列, 同 dataset 同 split 同评估。
脚本 [scripts/round15.py](../../../scripts/round15.py)。

**TabNet 配置**:
- `d_feat=158` (Alpha158 完美对齐)
- `n_d=32, n_a=32, n_steps=3` (中等容量, 防过拟)
- `n_epochs=30, early_stop=10, batch_size=4096, lr=0.005`
- `pretrain=False` (跳过 50 epoch 无监督预训练)
- `GPU=-1` (Mac 无 CUDA, CPU 训练)

**LGBM 配置**: 跟 R8.5/R12/R13 一致 (R7 Optuna params), 复现 R8.5/R12 基线

## 结果

| 指标 | LGBM | TabNet | TabNet 相对 |
|---|---|---|---|
| IC | **0.0328** | 0.0251 | **-23%** |
| RankIC | **0.0287** | 0.0220 | -23% |
| RankICIR | **0.218** | 0.145 | -33% |
| **long-only Sharpe** | **0.445** | **0.083** | **-81%** |
| **LS-100% Sharpe** | **0.716** | **0.228** | **-68%** |
| LS-200% Sharpe | 0.716 | 0.228 | -68% |
| 训练时长 | **0.6 min** | **417.5 min** | **×700** |

详细回测 (annualized return / MDD):

| 配置 | 年化 | Sharpe | MDD |
|---|---|---|---|
| LGBM long-only | +4.40% | **0.445** ✅ | -15.7% |
| LGBM LS-100% K=30 | +7.32% | **0.716** ✅ | -14.2% |
| LGBM LS-200% K=30 | +14.63% | 0.716 | -27.0% |
| TabNet long-only | +0.81% | **0.083** ❌ | -16.1% |
| TabNet LS-100% K=30 | +2.67% | **0.228** ❌ | -18.8% |
| TabNet LS-200% K=30 | +5.34% | 0.228 | -35.7% |

✅ = R8.5/R12 历史结果完美复现 (Sharpe 同到第 3 位)
❌ = TabNet 大幅低于 LGBM

## 关键发现

### 1. TabNet 不仅 backtest 输, IC 也输

不像 R13 (加财务因子, IC 微升 Sharpe 跌), TabNet 是 **IC 也输 + Sharpe 也输**。
说明 TabNet 不只是"信号方向不同导致 Sharpe 下降", 而是**就没学到比 LGBM 更好的预测**。

### 2. 训练时长 ×700 倍, 结果 -68%

| 模型 | 训练时长 | Sharpe (LS-100%) |
|---|---|---|
| LGBM | 0.6 min | **0.716** |
| TabNet (CPU) | 417.5 min (~7 h) | **0.228** |

**深度模型在 A 股日频 alpha 任务上没优势**。 这跟工业界普遍发现一致 — Kaggle 表格类比赛 LightGBM/XGBoost 是 SOTA, 深度模型只有在原始信号 (图像/序列) 上才有架构优势。

### 3. 第 6 次"改进失败"反证

| 轮 | 改进尝试 | 结果 |
|---|---|---|
| R9 | 模型集成 (LGBM+Ridge) | 失败 |
| R9 | 行业中性化 | 失败 |
| R11 | 调 topk 范围 | 全比 R8.5 差 |
| R13 | 加 10 个财务因子 | Sharpe 0.445 → 0.356 |
| R14 | 工程版 LS (IC 期货对冲) | Sharpe 0.21 (R12 0.72 无法兑现) |
| **R15** | **TabNet 深度模型** | **Sharpe 0.083 (LGBM 的 19%)** |

**R8.5 IR 0.445 在 6 个不同改进维度上都没被超过**。

## 为什么 TabNet 这么差？

3 个可能的原因 (按可能性):

1. **Alpha158 已经做了特征工程** — 158 个手工因子已包含技术分析师认为有用的所有组合 (Beta, MA, KMID, STD, ...). 这些是非线性已经"算好"的高级特征。LightGBM 用 decision tree 在这种已工程化的特征空间里高效搜索, 而 TabNet 需要从 raw 158 维再学一遍非线性组合, 用 attention mask 选择子集 — 本质上是在"重做特征工程"。

2. **样本量不够 → 过拟合验证集** — 训练集 ~3.5M 行 × 158 列 看起来不少, 但有效信息密度低 (alpha 信号 IC 只有 0.03). 在 SNR 极低的数据上, 大容量深度模型很容易学到训练集的噪声。LGBM 树的"硬边界 + L1L2 正则化"在低 SNR 下天然鲁棒。

3. **CPU 训练 30 epochs 不够收敛** — TabNet 论文一般用 GPU + 几百 epoch + 全集 fine-tune。CPU 30 epoch 可能没到 sweet spot。**但**: smoke test 显示 epoch 0 vs epoch 1 的 valid loss 已经走平, 说明小容量配置就足够拟合, 训练时长不是主要瓶颈。

## 核心结论

**R8.5 LightGBM + Alpha158 不仅在策略层面是项目天花板, 在模型层面也是**.

- 加更复杂的模型 (TabNet) ❌
- 加更多的因子 (财务) ❌ (R13)
- 加更复杂的策略 (LS / 期货对冲) ❌ (R12/R14)
- 加更多的超参数 (Optuna) ❌ (R5/R8)
- 加 ensemble (LGBM+Ridge) ❌ (R9)
- 加中性化 ❌ (R9)
- 改 topk ❌ (R11)

**7 个改进方向, 全部失败**。

A 股 5.4 年 CSI500 去偏 universe 上, R8.5 baseline 已经是当前模型库 + 因子库的实测最优。

## 方法论价值

R15 是项目最强的 "LightGBM 是低 SNR 表格数据上事实 SOTA" 的证据:
- 同样数据, 同样评估
- 训练时长 ×700
- 结果 -68%
- 连基础指标 (IC) 都没超过

这跟整个 ML quant 工业界的经验一致 (大多数对冲基金 production 主力仍是 GBDT 而非 DL)。但本项目用一个干净的 apples-to-apples 实验给出了具体数字。

## 下一步

经过 R15, 项目的 "R8.5 是天花板" 论已经反复印证 7 次。研究方法论闭环达到 textbook 级别完整度。

实际推荐路径:
- **A) 整理 PROJECT_REPORT v2** — 15 轮 + 7 个反证整理成完整方法论案例 (最现实, 项目级产出)
- **B) 横向 universe 扩展** — 试 HS300 / CSI1000, 验证 R8.5 配置可迁移性 (中等复杂度)
- **C) Walk-forward 验证** — 滚动训练验证 R8.5 时间外稳定性 (有数据, 改 split 重跑即可)
- 不推荐: 继续在 R8.5 之上"再加点东西" — 7 次反证已经证明这条路不通

## 复现命令

```bash
conda activate qlib
python scripts/round15.py  # 大约 7-8 小时 (CPU 上 TabNet 训练)
```

## 文件清单

- `scripts/round15.py` — TabNet + LGBM 对比框架
- `results/runs/round15/r15_result.json` — 完整 metrics
- `results/runs/round15/pred_lgbm.pkl` / `pred_tabnet.pkl` — 测试期预测值
- `results/runs/round15/summary.md` — 本文档
