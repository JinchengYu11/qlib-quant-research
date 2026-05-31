# R13 (B3) 财务因子叠加测试 — 第二次"反证型"发现

**假设**：在 R8.5 量价 Alpha158 基础上加 10 个核心财务因子 (Alpha158Finance)，能不能改进 R8.5 long-only IR 0.45 或 R12 hypothetical LS Sharpe 0.72？

**数据采集踩坑全程** (前后约 10 天)：

| 路径 | 结果 |
|---|---|
| baostock | 单股 21s × 1838 ≈ 10 小时, 50% 进度后停滞, 仅 175 只完成 |
| akshare `stock_financial_analysis_indicator` (新浪源) | 4-worker 触发 IP 黑名单, 75 只后停; 后来该接口本身返回空 |
| Wind (用户 Windows 跑) | 长期 stash, 没回来 |
| **Tushare** (本会话切换) | 账户 100 积分, 所有财务接口限频 1次/小时, 1838 股 = 76 天 ✗ |
| 网易 quotes.money.163 | 整个子域 502 (废) |
| **akshare `stock_financial_abstract` (东财源)** ✅ | **1.5s/股稳定, 1815/1838 = 98.7% 成功** |

最终用东财源 8 小时左右采集完成 (含偶发 sina 连接失败 backoff)。详见 [scripts/collect_financials_em.py](../../../scripts/collect_financials_em.py)。

**实现**：
- 10 财务字段 (ROE/ROA/NPM/GPM/EPS/DEBT_RATIO/EXP_RATIO/NI/REV/OCF), 详见 [factors/alpha158_finance.py](../../../factors/alpha158_finance.py)
- PIT 处理用法定披露截止日 (5/1, 9/1, 11/1, 次年 5/1), 详见 [factors/financial_pit_ak.py](../../../factors/financial_pit_ak.py)
- 写入 `cn_data_v4` 上每股 10 个新 `.day.bin` 字段
- 跑 4 配置 × 长/短 = 6 组对比，详见 [scripts/round13.py](../../../scripts/round13.py)

## 结果

| 配置 | 年化 | Vol | Sharpe | MDD |
|---|---|---|---|---|
| **A158 long-only** | **+4.40%** | 9.9% | **0.445** ✅ | -15.7% |
| **A158 LS-100% K=30** | **+7.32%** | 10.2% | **0.716** ✅ | -14.2% |
| A158 LS-200% K=30 | +14.63% | 20.4% | 0.716 | -27.0% |
| A158+FIN long-only | +3.58% | 10.1% | **0.356** ❌ | -19.4% |
| A158+FIN LS-100% K=30 | +4.33% | 10.9% | **0.396** ❌ | -17.9% |
| A158+FIN LS-200% K=30 | +8.67% | 21.9% | 0.396 | -33.9% |

✅ = R8.5/R12 历史结果完美复现 (Sharpe 都到第 3 位)
❌ = 加财务因子后下降

## IC 与 Sharpe 的剪刀差 (第 N 次印证)

| 指标 | Alpha158 | Alpha158+FIN | 差 |
|---|---|---|---|
| IC | 0.0328 | **0.0338** | +0.001 (微升) |
| RankIC | 0.0287 | **0.0298** | +0.001 (微升) |
| RankICIR | 0.218 | **0.225** | +0.007 (微升) |
| **long-only Sharpe** | **0.445** | **0.356** | **-0.089** (大降) |
| **LS-100% Sharpe** | **0.716** | **0.396** | **-0.320** (暴跌) |

**全部 IC 指标都微升**，但 **Sharpe 大跌** — 项目里第 5 次出现"IC 涨 ≠ 回测涨"（前几次 R4 R5 R9 R10 都见过）。本轮是迄今最大反差：IC 升 3%，Sharpe 降 45%。

## 财务因子 importance ranking (168 因子里)

| 因子 | gain | 排名 (1=最强) |
|---|---|---|
| EPS | 633.7 | **39** / 168 (中等偏上) |
| NI | 206.9 | 84 |
| ROA | 176.8 | 90 |
| NPM | 115.2 | 110 |
| ROE | 104.6 | 113 |
| GPM | 101.7 | 116 |
| DEBT_RATIO | 61.5 | 129 |
| REV | 57.3 | 130 |
| OCF | 20.6 | **159** (近垫底) |
| EXP_RATIO | 19.8 | **160** (近垫底) |

只有 EPS (基本每股收益) 进入前 1/4，其他都靠后。LGBM 自己说："这些财务信号我用不大上"。

## 为什么 IC 涨而 Sharpe 跌？

3 个假设 (按可能性):

1. **频率失配**：财务因子季度更新，量价因子日更新。LGBM 把财务当稳态背景，但 top-K 选股是日频排序，财务信号让边际换股决策被噪声推动，**增加换手 = 吞掉 alpha**。

2. **隐式 size/value tilt**：NI/REV 是绝对值因子，间接编码了 size。模型可能学到了"大盘股 + 高净利"组合，但这部分在 2020-2026 测试期 (中小盘 outperform 周期) 跑输了量价信号自己的选择。

3. **验证集过拟合**：多 10 个低信号噪声特征，LGBM 在 2018-2019 验证集上轻微过拟，结果 2020+ 测试集失效。RankICIR 微升 (0.218→0.225) 但 backtest 降，跟过拟合特征。

## 核心结论

**R13 + R14 共同盖棺：R8.5 (Alpha158 long-only IR 0.445) 是项目真天花板**。

| 维度 | 结论 |
|---|---|
| 模型集成 (R9) | 全部失败 |
| 行业中性化 (R9) | 失败 |
| 调 topk (R11) | 全比 R8.5 差 |
| Hypothetical short (R12) | Sharpe 0.72 但 (R14 反证) 工程版只剩 0.21 |
| **财务因子 (R13)** | **Sharpe 反而下降** |

R8.5 的 IR 0.445 在 5 个不同维度上都没被超过。**这就是项目的真实研究产物**。

## 14 轮研究最大的方法论收获

**"IC 涨 ≠ 回测涨"在本项目里被印证 5 次**，是机器学习量化研究里最反直觉、最容易忽略、也最容易"自我安慰"的陷阱。R13 是教科书级 case：
- 数据更全 (158 + 10 = 168 因子) — 直觉上"应该"更强
- IC / RankIC / RankICIR 都涨 — 教科书指标都通过
- 但实战回测大跌 50%

**没有什么比"在含成本、长测试期、真实 universe 上跑 backtest"更能反映 alpha 真实价值**。这就是为什么 R8.5 用 1 套朴素模型 + 1 套朴素策略 + 5.4 年长测，是项目最可信的结论。

## 下一步

研究方法论闭环已完成 (14 轮 + 5 个反证维度)。若要继续推 R8.5：

- **横向扩展**: 试 Transformer/LSTM 取代 LightGBM, 看深度模型能不能挤出剩余 alpha
- **Universe 扩展**: 试 HS300 / CSI1000 / 全 A 等不同股票池, 看 R8.5 配置是否在其他池也适用
- **时间外验证**: 做 walk-forward (滚动训练) 而非 fixed split, 验证 R8.5 在不同时间段的稳定性
- **写论文/报告**: 14 轮研究 + 5 个反证 是一个完整的"机器学习量化的常见陷阱"教学案例，方法论价值很高

**最现实**：把 R10-R14 整合进 PROJECT_REPORT v2，固化方法论产出。

## 复现命令

```bash
conda activate qlib

# 一次性: 采集 1815 股财务数据 (~8 小时)
python scripts/collect_financials_em.py

# 写入 qlib bins (~30 秒)
python scripts/build_financial_bins_em.py

# 跑 R13 4 组 (~30 秒)
python scripts/round13.py
```

## 文件清单

- `scripts/collect_financials_em.py` — 东财 collector (akshare 包装)
- `scripts/build_financial_bins_em.py` — 写入 cn_data_v4 bins
- `factors/financial_pit_ak.py` — PIT 处理 helper (复用)
- `factors/alpha158_finance.py` — Alpha158 + 10 财务因子 handler
- `scripts/round13.py` — 4 配置对比框架
- `data_raw/financials_em/` — 1815 个股票 CSV (gitignored)
- `results/runs/round13/r13_result.json` — 完整 metrics 结果
