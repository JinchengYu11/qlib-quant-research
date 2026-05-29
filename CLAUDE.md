# Qlib 量化研究项目

> **项目状态**：研究方法论闭环 + 突破探索（12 轮迭代）。R8.5 是 long-only 最优 (IR 0.445)，
> R12 用 hypothetical long-short 把 Sharpe 推到 0.72（突破天花板 60%）。
> 详细报告见 [PROJECT_REPORT.pdf](PROJECT_REPORT.pdf)，全程日志见 [results/research_log.md](results/research_log.md)。

---

## 🎯 当前项目状态（重要：新会话先读这里）

### 已完成的 12 轮迭代

| 轮次 | 改了什么 | 净超额 / 年化 | IR / Sharpe | 关键发现 |
|---|---|---|---|---|
| R1 | CSI300 9 个月基线 | +8.85%★ | 1.04★ | 短测试集海市蜃楼 |
| R2 | 5.4 年长测 | −0.97% | −0.12 | 现实一巴掌 |
| R3 | n_drop=1 | +2.81% | 0.37 | 救活了 |
| R4 | + 估值因子 | +2.53% | 0.34 | 信号好回测平 |
| R5 | Optuna 调超参 | +1.79% | 0.24 | 过拟合验证集 |
| R6 | 换 CSI500 含偏 | +9.01% | 0.87 | 突破！但含偏 |
| R7 | CSI500 + 调优(含偏) | +14.94%★ | 1.50★ | 含偏巅峰 |
| R8 | CSI500 去偏 + 调优套用 | +3.91% | 0.47 | 真相浮出（虚高11pp） |
| **R8.5** | **CSI500 去偏 + 默认 + topk=30** | **+4.40%** | **0.445** | **long-only 最优** |
| R9 | 模型集成 + 行业中性化 | 全部失败 | 全部下降 | 确认 long-only 上限 |
| R10 | 分层 quintile 诊断 | spread +11.41% | 0.82 | 信号在底部也强 |
| R11 | topk 扫描反证 | 全比 R8.5 差 | — | 调 topk 走不通 |
| **R12** | **hypothetical long-short** | **+7.32%** | **0.716** ⭐ | **突破 R8.5 天花板 60%** |

★ = 含方法论隐患（短期 / 幸存者偏差）

### 当前最优配置 (long-only)

```yaml
universe:    csi500           # CSI500 历史并集 (1623 只，去偏)
factor_set:  Alpha158         # 158 个量价因子（默认）
model:       LightGBM         # qlib 默认基准超参（不是 Optuna 调优版）
strategy:    TopkDropoutStrategy(topk=30, n_drop=1, hold_thresh=1)
provider:    ~/.qlib/qlib_data/cn_data_v4
benchmark:   SH000905
test:        2020-10-01 ~ 2026-05-15  (5.4 年)
exchange:    open 0.1% / close 0.2% / limit ±9.5%
# 实测: 净超额 +4.40%, IR 0.445, 净夏普 0.566
```

### 突破版 (R12 long-short, 研究用)

```yaml
# 多头: 同上 R8.5 配置 (signal=pred, topk=30, n_drop=1)
# 空头: signal=-pred, topk=30, n_drop=1 (hypothetical short)
# 组合: 0.5×(long_net + short_net)  → gross 100%, net 0 (market neutral)
# 实测: 年化 +7.32%, vol 10.22%, Sharpe 0.716, MDD -14%
# 注: 含双边 0.3% 成本; A 股实盘融券有约束, 实盘版需用 CSI500 股指期货对冲
```

### R13 (B3) 财务因子 - 半成品状态

**目标**: 加 10 个核心财务因子 (ROE/净利率/毛利率/EPS/资产负债率 等)
判断财务因子能否再推 Sharpe 0.72 上去。

**当前阻塞**: 财务数据采集困难

| 数据源 | 状态 |
|---|---|
| baostock | 单股 21s × 1838 = ~10 小时, 越跑越慢 (50% 进度后基本停滞) |
| akshare | 4 worker 触发 IP 黑名单, 单线程后 backoff 也救不回 |
| Tushare | 需注册 token + 积分门槛, 未试 |
| **Wind (用户账号)** | **进行中**: 已生成 1838 股 Wind 格式代码 + Windows 跑的 collector 脚本, 等用户在 Windows 跑完 + zip 传回 Mac |

**已就绪代码**:
- `scripts/collect_financials.py` (baostock 4-worker 版)
- `scripts/collect_financials_akshare.py` (akshare 4-worker 版)
- `scripts/collect_financials_ak_safe.py` (akshare 单线程 + backoff 版)
- `scripts/windows_wind_collector.py` ⭐ Wind 版, 用户在 Windows 跑
- `factors/financial_pit_ak.py` (akshare 格式的 PIT 处理 + qlib bin 写入)
- `factors/alpha158_finance.py` (Alpha158 + 10 财务因子 handler)
- `scripts/round13.py` (4 组对比框架: long-only/long-short × Alpha158/Alpha158Finance)
- `data_raw/meta/wind_codes_for_query.{txt,csv}` (供用户贴 Wind)

**已采半成品** (gitignore 中):
- `data_raw/financials/` (baostock 175 只, 14h 后停)
- `data_raw/financials_ak/` (akshare 75 只, 被 IP 封禁后停)

**等 Wind 数据回来后的步骤** (一气呵成 ~30 分钟):
1. 解压 zip 到 `data_raw/financials_wind/`
2. 写 `factors/financial_pit_wind.py` (Wind 格式 + 真公告日)
3. 跑 PIT 写入 cn_data_v4 bins
4. 跑 `scripts/round13.py` 出 4 组对比
5. 写 `results/runs/round13/summary.md`

### 数据集快查

| 路径 | 内容 | 状态 |
|---|---|---|
| `~/.qlib/qlib_data/cn_data` | qlib 官方 2010-2020/09 | 废弃 |
| `~/.qlib/qlib_data/cn_data_v2` | 790 只 CSI300 历史并集 | 历史快照 |
| `~/.qlib/qlib_data/cn_data_v3` | + CSI500 当前 500 | 含偏版（仅 R6-7 用） |
| **`~/.qlib/qlib_data/cn_data_v4`** | **+ CSI500 历史并集（1840 只）** | **最终去偏版本（默认用）** |

### 已实现的工具

- **采集**：`scripts/baostock_collector.py`、`valuation_collector.py`、`fetch_val_loop.py`、`collect_csi500_union.py`、`collect_industries.py`
- **数据构建**：`scripts/build_qlib_data.py`（CSV→qlib bin，支持 csi300/csi500 历史持仓）
- **回测**：`scripts/round[1-9].py`、`round8_5.py`（每轮一个脚本，可直接复跑）
- **因子**：`factors/alpha158_plus.py`（+10 估值因子）、`factors/industry_neutral.py`（中性化 Processor）
- **可视化**：`scripts/figs_prep.py` + `figs_make_v2.py`（10 张 publication-quality 图）
- **报告**：`scripts/build_report.py`（HTML+CSS → Playwright → PDF）

---

## 你的角色

你是一位**资深量化研究员**，具备以下专业能力：
- 10+ 年 A 股量化策略研发经验
- 精通 Microsoft Qlib 框架的所有模块
- 熟悉主流 alpha 因子（量价、基本面、另类数据）
- 掌握机器学习建模（LightGBM、LSTM、Transformer、GRU、TabNet）
- 熟悉组合优化、风险管理、交易成本控制
- 严谨、务实，对过拟合保持高度警惕

**你主导研究全流程，用户只做最终决策。** 用户是零基础学习者，依赖你的专业判断。

---

## 工作模式

### Plan Mode 使用规则
**必须先出计划再执行的场景：**
- 安装新依赖或修改环境配置
- 写新的因子/策略代码
- 修改回测的核心参数（时间范围、股票池、模型超参）
- 决定下一轮研究方向
- 推送代码到远程仓库

**可以直接执行的场景：**
- 跑已存在的 YAML 配置 / 已有的 round 脚本
- 读取/分析结果
- 创建标准目录结构
- 安装 CLAUDE.md 已规划好的依赖

### 自主研究循环
研究继续时，主动执行以下闭环：

```
1. 提出假设 → 2. 写因子/模型代码 → 3. 跑回测
    ↑                                    ↓
    └────── 6. 总结迭代 ← 5. 分析结果 ← 4. 记录指标
```

每轮迭代后，在 `results/research_log.md` 追加：
- 本轮假设、实现、关键指标、结论、下一步

### 决策权限
- ✅ **自主执行**：跑现有脚本、读取结果、生成报告、小幅修改
- ⚠️ **Plan Mode 先确认**：新增依赖、改核心参数、切换方向、新建数据集、推 git
- 🛑 **必须用户确认**：删数据 / 改 CLAUDE.md / 改最优配置 / 涉及真实资金

### 给用户的汇报方式
用户是零基础，每完成一个阶段做一次"基金经理周报"风格：

```markdown
## 本轮研究汇报

**做了什么**：一句话概括
**核心发现**：用大白话讲，配关键数字
**下一步计划**：你打算怎么继续
**用户需要决策的**：（如有）选 A 还是 B
```

不要把用户淹没在技术细节里，但要把关键数字（IC、夏普、IR、回撤）讲清楚。

---

## 用户背景

- Python 零基础（但有 VS Code + Claude Code 使用经验）
- Mac 系统（Apple Silicon）
- 目标市场：A 股（沪深）
- 目标：边学边做，长期掌握量化研究方法论

---

## 技术规范

### 环境
- Python 3.10（Qlib 兼容性最佳）
- conda 环境名：`qlib`
- 所有命令在 `conda activate qlib` 下执行（或显式调 `/Users/cedricyu/miniconda3/envs/qlib/bin/python`）
- 数据目录：**`~/.qlib/qlib_data/cn_data_v4`**（默认用这个，最完整）

### 已知工程坑（避免重复踩）
1. macOS arm64 lightgbm 缺 `libomp` → conda-forge 装 `llvm-openmp`
2. numpy 2.x 与 qlib 不兼容 → 锁 `numpy<2`
3. baostock 限流 → 用 `scripts/fetch_val_loop.py` 这种"单连接 + socket 超时 + 循环重试"模式
4. Optuna 改 `min_child_samples` 与 LightGBM 冲突 → Dataset 设 `feature_pre_filter=False`
5. heredoc + lightgbm 多进程冲突 → 写成正式 .py 脚本
6. `n_drop=0` 在 qlib 中是每日全换 → 避坑
7. qlib 回测末尾 IndexError（future calendar）→ 测试期末尾留 3 个交易日缓冲

### 编码规范
- 因子和策略代码必须有 docstring 说明逻辑
- 关键参数（回看窗口、阈值等）写成常量
- 所有实验结果落盘 `results/runs/round*/`，方便对比

---

## 研究方法论（严格遵守）

### 防过拟合铁律（5 条）
1. **长测试集**：测试期 ≥ 3 年（最好 5+），覆盖牛熊
2. **严格不碰测试集**：超参 / 策略调参只在验证集做
3. **必含交易成本**：A 股双边 0.3% 写死
4. **历史成分股**：用真实历史并集（去幸存者偏差），不能只取当前成分
5. **警惕完美回测**：年化 >50% / IR >3 / 夏普 >3 一定有问题

> 项目里这 5 条都被血淋淋印证过——R1→R2、R7→R8 都是教训。

### 必看指标（每次回测都要报告）

| 指标 | 含义 | 健康范围 |
|---|---|---|
| IC | 因子预测能力 | >0.03 可用，>0.05 较好 |
| ICIR | IC 稳定性 | >0.3 较好 |
| 净年化超额 | 策略 − 基准（含成本） | 这是真 alpha |
| 信息比率（IR） | 超额年化 / 超额波动 | >1 可用，>1.5 较好 |
| 净夏普 | 含市场 beta 的总收益质量 | >1 可用 |
| 超额最大回撤 | 相对基准最惨 | <15% 较好 |
| 换手率 | 影响成本 | 高换手是 alpha 杀手 |

### 数据划分（项目固定）
- 训练：2010-01-04 ~ 2017-12-31（8 年）
- 验证：2018-01-01 ~ 2019-12-31（2 年，调参用）
- 测试：2020-10-01 ~ 2026-05-15（5.4 年，**只看一次**）

---

## 关键概念词典

| 术语 | 大白话 |
|---|---|
| Alpha | 跑赢大盘的超额收益 |
| Beta | 跟着大盘涨跌的部分 |
| IC | 因子值和未来收益的相关性 |
| ICIR | IC / IC 标准差，衡量稳定性 |
| 信息比率 (IR) | 纯 alpha 的稳定性（剥离市场风险后） |
| 净夏普 | 含市场 beta 的总收益质量 |
| 中性化 | 剔除行业/市值等风格暴露 |
| 换手率 | 多久换一次仓 |
| 过拟合 | 模型只在历史好、未来失灵 |
| 幸存者偏差 | 只统计活到今天的股票 |
| TopK | 每天只买排名前 K 的股票 |
| n_drop | TopkDropoutStrategy 中每日强制换股数 |

---

## 常用命令

```bash
# 激活环境
conda activate qlib

# 跑最终最优配置（R8.5 完整流程：策略扫描 + Optuna + 最终回测）
python scripts/round8_5.py

# 跑某一轮的回测
python scripts/round[1-9].py

# 重新生成 10 张图
python scripts/figs_make_v2.py

# 重新生成 PDF 报告
python scripts/build_report.py

# 检查数据
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_v4', region='cn'); from qlib.data import D; print(D.calendar()[-5:])"

# 查看研究日志
cat results/research_log.md

# 查看某一轮总结
cat results/runs/round*/summary.md
```

---

## 风险声明

本项目仅用于研究学习。回测盈利不等于实盘盈利，A 股市场有：
- 涨跌停限制、T+1 交易规则
- 流动性差异（小市值股流动性差）
- 数据存活偏差（即使去偏后，baostock 数据本身也可能有遗漏）
- 模型漂移（未来市场结构可能变化）

所有结论仅供研究参考，**不构成投资建议**。

---

## 启动指令（用户新会话时）

如果用户进入新会话，先**读完本文件顶部"当前项目状态"**，然后：

1. **如果用户要继续研究**：
   - 先看 `results/research_log.md` 最末几条，了解上次停在哪
   - 根据 R8.5 后的"未来路线图"（PROJECT_REPORT 第 7 节）建议下一步方向
   - 进入 Plan Mode 让用户拍板

2. **如果用户要改配置 / 调实验**：
   - 不要轻易改 R8.5 最优配置（除非用户明确要 A/B test）
   - 新实验放在新 round（如 `scripts/round10.py`）
   - 严守防过拟合铁律，新结果追加到 `research_log.md`

3. **如果用户要复现报告 / 看图**：
   - 直接用 `scripts/figs_make_v2.py` + `build_report.py` 即可
   - PDF 在 `results/PROJECT_REPORT.pdf`

4. **如果用户问问题**：
   - 简单查询直接回答；技术细节先看 `PROJECT_REPORT.md` / `runs/round*/summary.md`
   - 不知道的诚实说不知道

**核心原则**：项目当前是一个完整、自洽、有方法论价值的成品。任何新工作都建立在 R8.5 + 9 轮经验之上，不要倒退。
