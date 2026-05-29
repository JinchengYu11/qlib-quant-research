# R14 工程版 long-short — 设计文档

**日期**：2026-05-30
**作者**：研究员（与 Claude 共同设计）
**状态**：Draft，待用户 review

---

## 1. 研究背景

| 轮 | 配置 | 关键指标 |
|---|---|---|
| R8.5 | CSI500 去偏 + Alpha158 + LGBM 默认超参 + TopK=30 long-only | IR **0.445**（净超额 +4.40%） |
| R12 | 上述 long leg + hypothetical short (signal=-pred, 假装做空 bottom K) | Sharpe **0.716**（突破天花板 60%） |

R12 突破的来源：market-neutral 把"被市场 beta 拖累的一半 alpha"拿回来。但 R12 的 short leg 是 **hypothetical**——把"反向选股能力"也计入了，A 股实盘融券约束下拿不到。

## 2. 研究问题

**核心**：把 R12 的 hypothetical short 替换成**真实可执行的 CSI500 期货对冲**后，能保留多少 alpha？预期落在 R8.5 (0.45) 与 R12 (0.72) 之间。

**次级**：
- beta-adjusted hedge 比静态 1x notional 改善多少？
- Finance 假设（basis drag / cash 利率）对结果的敏感度多大？

## 3. 数据

### 3.1 Long-side
- 复用现有 `~/.qlib/qlib_data/cn_data_v4` (CSI500 历史并集 1840 只)
- 复用 R8.5 模型预测值 `results/figures_data/R85_CSI500debiased_FINAL_pred.pkl`
- **不重跑** long leg 模型，保持跨轮可比

### 3.2 IC 期货
- 数据源：akshare `futures_main_sina` （主力连续合约 IC0）
- 落盘路径：`data_raw/futures/IC0_daily.csv`
- 列：`date, open, high, low, close, settle, volume`
- 时间范围：2015-01-01 ~ 2026-05-15（覆盖 R8.5 测试期 + 校验期）
- **采集脚本**：`scripts/collect_ic_futures.py`（一次性，预计 < 2 分钟，期货数据量小）
- akshare 限流：期货只一个时间序列，无 R13 那种 1838×4 的并发压力，单线程即可

### 3.3 Basis 时间序列（用于估 drag）
- `IC0_close - SH000905_close` → `basis_daily.csv`
- 用于估"主力连续合约"的隐含展期成本（research log 中报告平均贴水）
- baseline 用固定 5% 年化 basis drag；sensitivity 用 3%；实际数据用于校验 5% 的合理性

## 4. 时间框架

- **测试期**：`2020-10-01 ~ 2026-05-15`（与 R8.5 / R12 完全一致）
- **训练期**：不再训练，复用 R8.5 模型
- **不动测试集**：所有 finance 假设在 R12 完成时已 fixed，新参数只做 sensitivity scan，不调参

## 5. 架构

```
qlib量化研究/
├── scripts/
│   ├── collect_ic_futures.py        # 新建: akshare → IC0_daily.csv + basis_daily.csv
│   └── round14_engineering_ls.py    # 新建: 顶层组合脚本
├── engine/                           # 新建目录
│   ├── __init__.py
│   ├── ic_data.py                    # IC 数据加载 + 缺失日填充
│   ├── ic_engine.py                  # IC 头寸维护 (margin, daily PnL, basis drag)
│   └── account.py                    # 资金账户聚合 (cash + long + IC margin → NAV)
├── data_raw/futures/                 # 新建: IC 数据落盘
└── results/runs/round14/             # 新建: 结果输出
```

### 5.1 模块边界

#### `engine/ic_data.py`
- 接口：`load_ic_daily(start, end) -> pd.DataFrame[date, close, basis_5pct_drag_daily]`
- 职责：读 CSV、按交易日历对齐、缺失日 forward fill、计算 basis_drag 时间序列
- 不依赖：qlib、其他 engine 模块

#### `engine/ic_engine.py`
- 接口：
  ```python
  class ICEngine:
      def __init__(self, contract_multiplier=200, margin_rate=0.14,
                   trade_cost_bps=0.23, basis_drag_annual=0.05):
          ...
      def step(self, date, target_short_notional, ic_price) -> dict:
          """input: 今天想空多少 notional + 当日 IC 价格
             output: dict(delta_contracts, daily_pnl, margin_required,
                          trade_cost, basis_cost_today, position_contracts)
          """
  ```
- 内部状态：当前持仓张数、累积 PnL、entry price
- 不知道现货 portfolio，只服务于"今天给我空多少 notional"
- basis drag 按 `notional × basis_annual / 252` 每日摊销

#### `engine/account.py`
- 接口：
  ```python
  class Account:
      def __init__(self, initial_capital=1e8, cash_rate=0.0):
          ...
      def update(self, date, long_value, long_return_today,
                 ic_step_output) -> dict:
          """每日聚合一步, 返回 dict(nav, cash, long_value, ic_margin,
                                       ic_unrealized_pnl, margin_call)"""
  ```
- 记账：现金账户 / 多头股票市值 / IC 保证金占用 / IC 浮盈浮亏
- margin call 触发逻辑：`cash + ic_unrealized_pnl < ic_margin_required` → log warning，不强平（研究用）
- 每日更新 cash 利息（baseline 0%，sensitivity 1.5%）

#### `scripts/round14_engineering_ls.py`
- 调用 qlib backtest 跑 long leg（复用 R8.5 TopK=30 配置）
- 提取 daily `(long_value, long_return)`
- For each config in [C1, C2, C3]:
  - 初始化 ICEngine + Account
  - 按日 loop：
    1. 算今天的 target_short_notional（C1: =long_value；C2: =beta × long_value；C3: =long_value）
    2. `ic_step = ic_engine.step(date, target_notional, ic_price_today)`
    3. `account_state = account.update(date, long_value, long_ret, ic_step)`
  - 输出 daily NAV 序列
- 算 metrics + 可视化 + 落盘

### 5.2 跑测配置矩阵

| Config | Hedge ratio | Finance 假设 |
|---|---|---|
| **C1** (baseline) | 1x notional static | 保守 (14% margin / 0% cash / 5% basis / 万 0.23 手续费) |
| **C2** (beta-adjusted) | 60d rolling beta × long_value | 保守（同上） |
| **C3** (sensitivity) | 1x notional static | 乐观 (14% margin / **1.5% cash** / **3% basis** / 万 0.23) |

**Beta 估计**：60 个交易日窗口，long_leg daily returns vs SH000905 daily returns 简单 OLS slope。前 60 日 fallback 用 beta=1。

## 6. 评估

### 6.1 指标
- 净年化收益、年化波动、Sharpe、最大回撤、平均换手率（仅多头侧）
- **新指标**：margin 占用峰值、margin call 触发次数、basis drag 累积成本

### 6.2 横评
| 对比对象 | 数据来源 |
|---|---|
| CSI500 基准 | benchmark from qlib backtest |
| R8.5 long-only (净超额) | `results/runs/round8_5/` |
| R12 LS-100% K=30 (Sharpe 0.72) | `results/runs/round12/ls_LS-100pct_K_30.pkl` |
| R14 C1 / C2 / C3 | 本轮 |

### 6.3 必报结论
- C1 vs R12：alpha 衰减多少 = "hypothetical short → 真实期货对冲" 的代价
- C2 vs C1：beta-adjusted 是否改善
- C3 vs C1：finance 假设松紧对结果的影响范围

## 7. 输出

```
results/runs/round14/
├── summary.md                          # 跟 round12 同风格的轮次总结
├── engineering_ls_summary.json         # 所有 metrics dict
├── nav_C1_1x_conservative.pkl          # 每日 NAV 序列
├── nav_C2_beta_conservative.pkl
├── nav_C3_1x_optimistic.pkl
└── ic_state_C1.pkl                     # IC engine 每日状态 (debug 用)

results/figures/
└── fig14_engineering_ls.png            # 跟 fig13 同布局, 两块面板:
                                         #   上: 5 条 NAV 累计曲线
                                         #   下: Sharpe / IR 横评柱图

results/research_log.md                 # 追加 R14 章节
CLAUDE.md                                # 更新顶部"当前项目状态"表 (加 R14 行)
```

## 8. 不在范围内

明确**不做**的事，避免 scope creep：
- 实盘 broker 对接（CTP / vn.py）
- 实盘风控（margin call 时的实际平仓逻辑）
- IC 当季/次季/季季月策略（用主力连续，不自管换月）
- 持仓限制（CFFEX 对个人有手数上限，研究用 1 亿资金不会触及）
- 重训 long leg 模型 / 改 Alpha158
- 触碰任何 R13 财务因子框架（等 Wind 数据回来再开 R15）

## 9. 风险与待校验项

| 风险 | 缓解 |
|---|---|
| akshare IC0 数据有缺失日 | ic_data.py 用交易日历 forward fill，并 log 缺失天数 |
| basis drag 5% 假设偏离实际 | 用 IC0-SH000905 实际历史数据校验，研究 log 报告实际值 |
| 60d rolling beta 不稳 | 前 60 日 fallback=1.0；report beta 时序的均值/std |
| 浮盈/浮亏会拉爆 cash（margin call） | account.py 记账并 log；研究用不强平，但报告触发次数 |
| 第一版可能 backtest 数字有 bug | C1 1x notional 跑出来的 net return 应该 ≈ R12 LS-100% K=30 的 "long leg net excess" 部分，可对账 |

## 10. 估计代码量与工期

- `collect_ic_futures.py`：~50 行
- `engine/ic_data.py`：~80 行
- `engine/ic_engine.py`：~150 行
- `engine/account.py`：~120 行
- `round14_engineering_ls.py`：~250 行
- `summary.md`：人工写
- **合计**：~650 行，2-3 个工作日（含 debug + 对账）

## 11. 跟项目历史对账

| Checkpoint | 期望 |
|---|---|
| C1 long leg 跑出的年化收益 | ≈ R8.5 absolute return（不是超额）|
| C1 IC 端总 PnL ≈ -CSI500 净收益 × hedge_period | basis drag 之外应该跟 -CSI500 高度负相关 |
| C1 净 Sharpe | 介于 0.45 (R8.5 IR) 和 0.72 (R12) 之间 |
| C2 vs C1 | 微小改善（CSI500 选股的 beta 通常接近 1，beta-adjust 影响有限）|
| C3 vs C1 | 乐观 finance 提升 ~30-50bp 年化 |

如果 C1 跑出来 > 0.72 或 < 0.30，**先怀疑 bug，不要相信结果**。

---

**Spec 完。**
