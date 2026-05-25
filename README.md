# A 股量化策略研究项目

> **9 轮迭代实证 · 从零基础到真实可信 alpha 策略**

[![Status](https://img.shields.io/badge/status-research%20complete-success)]() [![Test Period](https://img.shields.io/badge/test%20period-5.4%20years-blue)]() [![Net IR](https://img.shields.io/badge/Net%20IR-0.445-c89b3c)]() [![Net Excess](https://img.shields.io/badge/Net%20Excess-%2B4.40%25-c89b3c)]()

## 项目最终成果

**最优配置**：`CSI500 历史并集 + Alpha158 + LightGBM 默认超参 + TopkDropoutStrategy(topk=30, n_drop=1)`

| 指标 | 数值 |
|---|---|
| 测试期 | 2020-10 ~ 2026-05（5.4 年，含真实历史成分变更） |
| **净年化超额** | **+4.40%** |
| **信息比率** | **0.445** |
| 净夏普 | 0.566 |
| 超额最大回撤 | -12.6% |
| 日均换手 / 年化成本 | 6.8% / 2.5% |
| 2022 熊市表现 | 中证500 跌 20.4% / 策略仅跌 5.6%（防御性突出） |

📄 **[完整 PDF 报告 → PROJECT_REPORT.pdf](results/PROJECT_REPORT.pdf)** （~35 页，含 10 张可视化）

---

## 项目演进战绩

| 轮次 | 配置 | 净超额 | IR | 关键发现 |
|---|---|---|---|---|
| R1 | CSI300 9 个月基线 | +8.85% ★ | 1.04 ★ | 短测试集海市蜃楼 |
| R2 | CSI300 5.4 年长测 | −0.97% | −0.12 | 现实一巴掌 |
| R3 | + 降换手 (n_drop=1) | +2.81% | 0.37 | 救活了 |
| R4 | + 估值因子 | +2.53% | 0.34 | 信号好回测平 |
| R5 | + Optuna 调超参 | +1.79% | 0.24 | 过拟合验证集 |
| R6 | 换 CSI500（含偏） | +9.01% | 0.87 | 突破！但含偏 |
| R7 | CSI500 + 调优（含偏） | +14.94% ★ | 1.50 ★ | 含偏巅峰 |
| R8 | CSI500 去偏 + 调优套用 | +3.91% | 0.47 | 真相浮出（-11pp） |
| **R8.5** | **CSI500 去偏 + 默认 + topk=30** | **+4.40%** | **0.45** | **★ 最终最优** |
| R9 | 模型集成 + 行业中性化 | 全部失败 | 全部下降 | 确认上限 |

★ = 含方法论隐患的数字（短期偏差 / 幸存者偏差）

---

## 6 个核心方法论收获

1. **长测试集是底线**——9 个月 +8.85% 在 5.4 年立刻打回 −0.97%
2. **降换手是免费午餐**——模型不变、n_drop 5→1、净超额 −0.97% → +2.81%
3. **IC 涨 ≠ 回测涨**——项目 5 次出现该反直觉现象
4. **战场决定上限**——CSI500 vs CSI300 同模型 IR 翻倍多
5. **简单胜过复杂**——默认超参 + topk=30 打败 200 轮 Optuna 搜索
6. **幸存者偏差实际虚高 11 pp**（中盘股尤甚），远超经典估计 2-4 pp

---

## 项目结构

```
qlib量化研究/
├── PROJECT_REPORT.pdf              ← 完整报告（~35 页）
├── PROJECT_REPORT.md               ← markdown 源版
├── CLAUDE.md                       ← Claude Code 项目规则
├── README.md                       ← 本文件
│
├── configs/                        ← qrun YAML 配置
├── factors/                        ← 自定义 handler
│   ├── alpha158_plus.py            ← Alpha158 + 10 估值因子
│   └── industry_neutral.py         ← 行业中性化 Processor
├── scripts/                        ← 25+ 工程脚本
│   ├── baostock_collector.py       ← OHLCV 多进程采集
│   ├── valuation_collector.py      ← 估值采集
│   ├── fetch_val_loop.py           ← 单连接循环兜底
│   ├── build_qlib_data.py          ← CSV → qlib bin 转换
│   ├── round[1-9].py / round8_5.py ← 各轮回测脚本
│   ├── figs_make_v2.py             ← publication-quality 图表
│   └── build_report.py             ← PDF 报告生成器
└── results/
    ├── research_log.md             ← 全程研究日志（9 轮记录）
    ├── runs/                       ← 每轮 summary.md + JSON
    └── figures/                    ← 10 张 PNG 图表
```

> **数据未推入 git**（原始 CSV ~780 MB）。要复现需运行 `scripts/baostock_collector.py` 等采集脚本，约 4-6 小时（baostock 限流）。

---

## 复现步骤

### 1. 环境

```bash
# Apple Silicon 注意：lightgbm 需要 libomp
conda create -n qlib python=3.10 -y
conda activate qlib
pip install "numpy<2" pyqlib lightgbm jupyter matplotlib baostock optuna playwright
conda install -c conda-forge llvm-openmp -y  # macOS arm64 需要

playwright install chromium  # PDF 报告生成用
```

### 2. 拉数据（需联网，约 4-6 小时）

```bash
# CSI300 历史并集 OHLCV（790 只）
python scripts/baostock_collector.py --start 2010-01-01 --end 2026-05-19 --universe csi300 --workers 4

# CSI500 历史并集 OHLCV（1623 只）
python scripts/build_csi500_history.py
python scripts/collect_csi500_union.py 4

# 估值数据（可选，最终配置不依赖）
python scripts/fetch_val_loop.py

# 行业归属（如要做 R9 中性化实验）
python scripts/collect_industries.py
```

### 3. 构建 qlib 数据集

```bash
python scripts/build_qlib_data.py --output ~/.qlib/qlib_data/cn_data_v4
```

### 4. 跑最终最优配置

```bash
# 用 round8_5.py（含完整 Optuna 搜索 + 策略扫描）
python scripts/round8_5.py
```

### 5. 生成可视化报告

```bash
python scripts/figs_prep.py       # 重跑 5 组关键配置存日序列
python scripts/figs_make_v2.py    # 生成 10 张图表
python scripts/build_report.py    # 生成 PDF 报告
```

---

## 未来路线图

**短期（1-2 天）**：组合优化器、分层 long-only 分析、风险因子归因
**中期（1-2 周）**：财务因子（季度 ROE/营收/毛利）、一致预期、CSI1000、LSTM/Transformer
**长期（数月）**：高频微观结构、另类数据（卫星/新闻 NLP）、实盘小规模验证

详见 [PROJECT_REPORT.pdf](results/PROJECT_REPORT.pdf) 第 7 节。

---

## 技术栈

- **数据**：[baostock](http://baostock.com)（免费 A 股数据）+ 自建 qlib bin 转换
- **框架**：[Microsoft Qlib](https://github.com/microsoft/qlib)
- **模型**：LightGBM（默认超参）/ Optuna 超参优化 / sklearn Ridge
- **回测**：qlib TopkDropoutStrategy + SimulatorExecutor
- **可视化**：matplotlib（自定义 publication-quality 样式）
- **报告**：HTML + CSS → Playwright Chromium → PDF

---

## License

MIT - 仅供研究学习参考。回测盈利不等于实盘盈利，**不构成投资建议**。

---

> *"好的研究不是找到漂亮的数字，而是排除所有可能让数字漂亮的虚假理由。"*
