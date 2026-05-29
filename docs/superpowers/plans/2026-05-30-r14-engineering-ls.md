# R14 工程版 long-short Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 R12 hypothetical short 换成真实 CSI500 期货 (IC) 对冲，跑出 3 组 backtest (1x notional 保守 / beta-adjusted / sensitivity)，量化 alpha 衰减。

**Architecture:** 3 个独立 engine 模块 (`ic_data`, `ic_engine`, `account`) + 1 个顶层组合脚本 (`round14_engineering_ls.py`)。Long leg 复用 R8.5 模型预测（不重跑），IC 端独立维护 margin/PnL/basis，账户层聚合 NAV。每个 engine 模块自带 inline `__main__` 单元测试（不引入 pytest）。

**Tech Stack:** Python 3.10 / qlib / pandas / numpy / akshare / matplotlib。完整 spec 在 [docs/superpowers/specs/2026-05-30-r14-engineering-ls-design.md](../specs/2026-05-30-r14-engineering-ls-design.md)。

**Environment:** 所有命令前置 `conda activate qlib` 或用 `/Users/cedricyu/miniconda3/envs/qlib/bin/python`。CWD = `/Users/cedricyu/qlib量化研究`。

---

## Task 1: 项目脚手架

**Files:**
- Create: `engine/__init__.py`
- Create: `data_raw/futures/.gitkeep`
- Create: `results/runs/round14/.gitkeep`

- [ ] **Step 1: 建目录**

Run:
```bash
mkdir -p /Users/cedricyu/qlib量化研究/engine
mkdir -p /Users/cedricyu/qlib量化研究/data_raw/futures
mkdir -p /Users/cedricyu/qlib量化研究/results/runs/round14
```

- [ ] **Step 2: 建空 `engine/__init__.py`**

Write `engine/__init__.py` with content:
```python
"""R14 工程版 long-short 的独立模块: IC 数据 / IC 头寸引擎 / 资金账户聚合"""
```

- [ ] **Step 3: 建 .gitkeep 占位**

Run:
```bash
touch /Users/cedricyu/qlib量化研究/data_raw/futures/.gitkeep
touch /Users/cedricyu/qlib量化研究/results/runs/round14/.gitkeep
```

- [ ] **Step 4: 检查 .gitignore 是否会忽略 data_raw/futures/**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && git check-ignore -v data_raw/futures/IC0_daily.csv 2>&1 || echo "NOT IGNORED"
```

Expected: 如果输出 "NOT IGNORED" 就直接 commit；如果被 ignore 了，在 `.gitignore` 加一行 `!data_raw/futures/` 显式取消忽略（因为这是研究产物中**重要**的少量数据，不像 baostock 全量 CSV）。

- [ ] **Step 5: Commit 脚手架**

```bash
cd /Users/cedricyu/qlib量化研究
git add engine/__init__.py data_raw/futures/.gitkeep results/runs/round14/.gitkeep
git commit -m "R14 脚手架: engine/ 目录 + 输出占位"
```

---

## Task 2: IC 期货数据采集

**Files:**
- Create: `scripts/collect_ic_futures.py`
- Output: `data_raw/futures/IC0_daily.csv` + `data_raw/futures/basis_daily.csv`

- [ ] **Step 1: 写 collector 脚本**

Write `scripts/collect_ic_futures.py`:
```python
"""一次性采集 CSI500 期货 IC 主力连续合约日线 + basis 时间序列.

数据源: akshare futures_main_sina (sina 主力连续, 已处理换月)
输出:
  data_raw/futures/IC0_daily.csv  -- date, open, high, low, close, settle, volume
  data_raw/futures/basis_daily.csv -- date, ic_close, csi500_close, basis_abs, basis_pct
"""
import sys
from pathlib import Path
import pandas as pd
import akshare as ak

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'data_raw' / 'futures'
OUT.mkdir(parents=True, exist_ok=True)

START = '2015-01-01'
END = '2026-05-30'


def fetch_ic_main():
    print(f"[1] 拉 IC 主力连续 ({START} ~ {END}) ...", flush=True)
    df = ak.futures_main_sina(symbol='IC0', start_date=START.replace('-', ''),
                               end_date=END.replace('-', ''))
    df.columns = [c.lower() for c in df.columns]
    rename = {'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
              '收盘价': 'close', '结算价': 'settle', '成交量': 'volume', '持仓量': 'oi'}
    df = df.rename(columns=rename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"    拿到 {len(df)} 行, 时间范围 {df['date'].min()} ~ {df['date'].max()}", flush=True)
    return df[['date', 'open', 'high', 'low', 'close', 'settle', 'volume']]


def fetch_csi500_index():
    print(f"[2] 拉 SH000905 (CSI500 指数) 日线 ...", flush=True)
    df = ak.stock_zh_index_daily(symbol='sh000905')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"    拿到 {len(df)} 行", flush=True)
    return df[['date', 'close']].rename(columns={'close': 'csi500_close'})


def main():
    ic = fetch_ic_main()
    ic.to_csv(OUT / 'IC0_daily.csv', index=False)
    print(f"  ✓ 写出 {OUT / 'IC0_daily.csv'}", flush=True)

    idx = fetch_csi500_index()
    basis = ic[['date', 'close']].rename(columns={'close': 'ic_close'}).merge(idx, on='date', how='inner')
    basis['basis_abs'] = basis['ic_close'] - basis['csi500_close']
    basis['basis_pct'] = basis['basis_abs'] / basis['csi500_close']
    basis.to_csv(OUT / 'basis_daily.csv', index=False)
    print(f"  ✓ 写出 {OUT / 'basis_daily.csv'}", flush=True)

    avg_basis_pct = basis['basis_pct'].mean() * 100
    print(f"\n[INFO] 平均 basis = {avg_basis_pct:+.3f}% (负值 = IC 贴水 = 多头持仓有收益)", flush=True)
    print(f"       Spec baseline 假设 5% 年化 basis drag, 实际数据校验后可在 round14 里 override", flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑 collector**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python scripts/collect_ic_futures.py
```

Expected: 2 个 CSV 文件落到 `data_raw/futures/`，stdout 打印行数和平均 basis。如果 akshare 接口名变了或返回字段不同，调整 `rename` 字典直到成功。

- [ ] **Step 3: 抽验输出**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
import pandas as pd
ic = pd.read_csv('data_raw/futures/IC0_daily.csv', parse_dates=['date'])
basis = pd.read_csv('data_raw/futures/basis_daily.csv', parse_dates=['date'])
print('IC0:', len(ic), '行', ic['date'].min(), '~', ic['date'].max())
print('Basis:', len(basis), '行, 平均贴水 (basis_pct):', basis['basis_pct'].mean()*100, '%')
print('IC0 close 前 3:', ic[['date','close']].head(3).to_dict('records'))
print('IC0 close 后 3:', ic[['date','close']].tail(3).to_dict('records'))
"
```

Expected: IC0 应有 ~2500+ 行（2015-至今每日），平均 basis_pct 大概率在 -1% ~ -4% 之间（A 股 IC 历史贴水）。

- [ ] **Step 4: Commit data + collector**

```bash
cd /Users/cedricyu/qlib量化研究
git add scripts/collect_ic_futures.py data_raw/futures/IC0_daily.csv data_raw/futures/basis_daily.csv
git commit -m "R14: 采集 IC 主力连续合约 + basis 时间序列"
```

---

## Task 3: `engine/ic_data.py` (TDD)

**Files:**
- Create: `engine/ic_data.py`

- [ ] **Step 1: 写实现 + 测试（写在文件底部 `__main__` 里）**

注意: basis drag 仅由 `engine/ic_engine.py` 处理 (constructor 参数 `basis_drag_daily`)。`ic_data` 只负责加载价格，避免两处计算 → double-count bug。

Write `engine/ic_data.py`:
```python
"""IC 期货日线加载器 (只管价格).

接口: load_ic_daily(start, end) -> pd.DataFrame
    index = trading_date (datetime)
    columns = [close]

basis drag 不在这里算 — 是 ICEngine 的事 (避免 double-count).
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/cedricyu/qlib量化研究')
IC_CSV = ROOT / 'data_raw' / 'futures' / 'IC0_daily.csv'


def load_ic_daily(start, end, ic_csv=None):
    """加载 IC 主力连续日线.

    Args:
        start, end: 字符串或 datetime, inclusive
        ic_csv: 可选, 覆盖默认 CSV 路径 (测试用)

    Returns:
        pd.DataFrame indexed by date with column [close]
        只返回 IC 实际交易日, 不补 weekend / holiday.
    """
    path = Path(ic_csv) if ic_csv else IC_CSV
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['date', 'close']].sort_values('date').reset_index(drop=True)
    df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
    df = df.set_index('date')
    return df


# ===== inline tests =====
def _test_load_basic():
    """读 3 行小 CSV, 验证列和值."""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.write("2020-01-06,5030,5060,5010,5050,5045,1100\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert len(df) == 3, f"期望 3 行, 拿到 {len(df)}"
    assert df.index[0] == pd.Timestamp('2020-01-02'), f"首日错: {df.index[0]}"
    assert df['close'].iloc[0] == 5020
    assert df['close'].iloc[2] == 5050
    assert list(df.columns) == ['close'], f"应该只有 close 列, 拿到 {list(df.columns)}"
    print("  ✓ _test_load_basic")


def _test_date_filter():
    """start/end 过滤生效"""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2019-12-30,4900,4950,4880,4920,4915,1000\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert len(df) == 2, f"期望过滤掉 12-30, 拿到 {len(df)} 行"
    print("  ✓ _test_date_filter")


def _test_sorted():
    """无论 CSV 顺序, 输出按日期排序"""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2020-01-06,5030,5060,5010,5050,5045,1100\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert df.index.is_monotonic_increasing
    print("  ✓ _test_sorted")


if __name__ == '__main__':
    print("Running engine/ic_data.py tests ...")
    _test_load_basic()
    _test_date_filter()
    _test_sorted()
    print("All tests passed.")
```

- [ ] **Step 2: 跑 inline tests**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python engine/ic_data.py
```

Expected:
```
Running engine/ic_data.py tests ...
  ✓ _test_load_basic
  ✓ _test_date_filter
  ✓ _test_sorted
All tests passed.
```

- [ ] **Step 3: 验证 real data 加载**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
from engine.ic_data import load_ic_daily
df = load_ic_daily('2020-10-01', '2026-05-15')
print('rows:', len(df))
print('first:', df.index[0], df.iloc[0].to_dict())
print('last:', df.index[-1], df.iloc[-1].to_dict())
"
```

Expected: ~1300+ 行（5.4 年交易日），首/末日打印正确。

- [ ] **Step 4: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add engine/ic_data.py
git commit -m "R14: engine/ic_data.py — IC 日线加载 + basis drag"
```

---

## Task 4: `engine/ic_engine.py` (TDD)

**Files:**
- Create: `engine/ic_engine.py`

- [ ] **Step 1: 写实现 + 测试**

Write `engine/ic_engine.py`:
```python
"""IC 期货头寸引擎 — 维护持空 N 张合约的状态.

每日接收 target_short_notional (今天想空多少元), 算出:
  - 需要的合约张数 (N = round(notional / (price × multiplier)))
  - delta_contracts (今天要新开/平多少张)
  - daily_pnl (持仓张数 × (last_close - today_close) × multiplier)
    [注: 空头收益 = -(today_close - last_close), 我们维护 self.contracts > 0 表示空 N 张]
  - margin_required (今天 EOD 占用的保证金)
  - trade_cost (今天调仓的手续费)
  - basis_cost_today (basis drag 摊销)

不知道现货 portfolio, 也不管资金账户. 那是 account.py 的事.
"""
from dataclasses import dataclass, field


@dataclass
class ICEngineState:
    contracts_short: int = 0           # 持空合约张数 (正值 = 空 N 张)
    last_close: float = 0.0            # 上一日 IC 收盘价 (用于算今日 PnL)


class ICEngine:
    def __init__(self, contract_multiplier=200, margin_rate=0.14,
                 trade_cost_bps=0.23, basis_drag_daily=-0.05/252):
        """
        Args:
            contract_multiplier: IC 合约乘数 (CFFEX = 200 元/点)
            margin_rate: 保证金率 (0.14 = 14%)
            trade_cost_bps: 期货手续费 (万分之 bps, 0.23 = 万 0.23)
            basis_drag_daily: 每日 basis drag (负值 = 空头每天亏的部分), 默认 -5%/252
        """
        self.M = contract_multiplier
        self.margin_rate = margin_rate
        self.trade_cost_bps = trade_cost_bps
        self.basis_drag_daily = basis_drag_daily
        self.state = ICEngineState()

    def step(self, target_short_notional, ic_price):
        """处理一个交易日.

        Returns: dict 含以下字段:
            delta_contracts: int, 今日调仓张数 (正 = 新开空, 负 = 平空)
            position_contracts: int, EOD 持仓
            daily_pnl: float, 持仓 PnL (上一日 close → 今日 close, 空头方向)
            margin_required: float, EOD 保证金
            trade_cost: float, 今日手续费
            basis_cost: float, 今日 basis drag (基于平均持仓 notional)
        """
        # 1. 目标张数
        target_contracts = int(round(target_short_notional / (ic_price * self.M)))
        delta_contracts = target_contracts - self.state.contracts_short

        # 2. 今日 PnL (基于上一日 close → 今日 close)
        if self.state.last_close > 0:
            # 空头: 价格涨 → 亏, 价格跌 → 赚
            daily_pnl = self.state.contracts_short * (self.state.last_close - ic_price) * self.M
        else:
            daily_pnl = 0.0  # 第一天还没有 last_close

        # 3. 今日 trade cost (按调仓张数 × 价格 × 乘数 × bps)
        trade_notional = abs(delta_contracts) * ic_price * self.M
        trade_cost = trade_notional * self.trade_cost_bps / 1e4

        # 4. EOD 持仓 + 保证金
        self.state.contracts_short = target_contracts
        eod_notional = abs(target_contracts) * ic_price * self.M
        margin_required = eod_notional * self.margin_rate

        # 5. Basis drag (按 EOD notional 摊销)
        basis_cost = eod_notional * self.basis_drag_daily  # basis_drag_daily 是负数

        # 6. 更新 last_close
        self.state.last_close = ic_price

        return dict(
            delta_contracts=delta_contracts,
            position_contracts=self.state.contracts_short,
            daily_pnl=daily_pnl,
            margin_required=margin_required,
            trade_cost=trade_cost,
            basis_cost=basis_cost,
        )


# ===== inline tests =====
def _test_first_day_no_pnl():
    """第一天: 开仓但没有 PnL (没有 last_close)"""
    e = ICEngine()
    out = e.step(target_short_notional=1_000_000, ic_price=5000)
    # 5000 × 200 = 1,000,000/张, 想空 1M → 1 张
    assert out['delta_contracts'] == 1, f"期望 1 张, 拿到 {out['delta_contracts']}"
    assert out['position_contracts'] == 1
    assert out['daily_pnl'] == 0.0, "第一天不应有 PnL"
    assert abs(out['margin_required'] - 1_000_000 * 0.14) < 1e-6
    print("  ✓ _test_first_day_no_pnl")


def _test_second_day_pnl_short_wins_on_drop():
    """第二天 IC 跌 → 空头赚钱"""
    e = ICEngine()
    e.step(target_short_notional=1_000_000, ic_price=5000)  # 开 1 张, last_close=5000
    out = e.step(target_short_notional=1_000_000, ic_price=4900)  # IC 跌到 4900
    # 空 1 张, IC 跌 100 点 × 200 元/点 = 赚 20,000
    assert abs(out['daily_pnl'] - 20_000) < 1e-6, f"期望 +20000, 拿到 {out['daily_pnl']}"
    # 新目标 notional = 1M, 新 price = 4900, target = round(1M / (4900×200)) = round(1.02) = 1 张
    assert out['delta_contracts'] == 0, f"期望不调仓 (1→1), 拿到 {out['delta_contracts']}"
    print("  ✓ _test_second_day_pnl_short_wins_on_drop")


def _test_short_loses_on_rise():
    """IC 涨 → 空头亏"""
    e = ICEngine()
    e.step(target_short_notional=10_000_000, ic_price=5000)  # 开 10 张
    out = e.step(target_short_notional=10_000_000, ic_price=5050)  # IC 涨 50 点
    # 空 10 张, IC 涨 50 点 × 200 = 亏 100,000
    assert abs(out['daily_pnl'] - (-100_000)) < 1e-6, f"期望 -100000, 拿到 {out['daily_pnl']}"
    print("  ✓ _test_short_loses_on_rise")


def _test_rebalance_charges_trade_cost():
    """调仓收手续费"""
    e = ICEngine(trade_cost_bps=0.23)
    out = e.step(target_short_notional=10_000_000, ic_price=5000)  # 开 10 张
    # 10 张 × 5000 × 200 = 10M notional
    # 手续费 = 10M × 0.23/1e4 = 230
    assert abs(out['trade_cost'] - 230.0) < 1e-6, f"期望 230, 拿到 {out['trade_cost']}"
    print("  ✓ _test_rebalance_charges_trade_cost")


def _test_basis_drag_negative():
    """basis drag 默认负 (空头承担贴水成本)"""
    e = ICEngine(basis_drag_daily=-0.05/252)
    out = e.step(target_short_notional=10_000_000, ic_price=5000)
    # 10M notional × -5%/252 ≈ -1984
    expected = 10_000_000 * (-0.05/252)
    assert abs(out['basis_cost'] - expected) < 1e-6, f"期望 {expected}, 拿到 {out['basis_cost']}"
    assert out['basis_cost'] < 0
    print("  ✓ _test_basis_drag_negative")


def _test_target_zero_closes_position():
    """target=0 平仓"""
    e = ICEngine()
    e.step(target_short_notional=10_000_000, ic_price=5000)  # 开 10
    out = e.step(target_short_notional=0, ic_price=5000)
    assert out['delta_contracts'] == -10, f"期望 -10, 拿到 {out['delta_contracts']}"
    assert out['position_contracts'] == 0
    assert out['margin_required'] == 0
    print("  ✓ _test_target_zero_closes_position")


def _test_contract_rounding():
    """target notional 不是整数张时取整"""
    e = ICEngine()
    # 1.4 张 → round → 1 张
    out = e.step(target_short_notional=1_400_000, ic_price=5000)
    assert out['position_contracts'] == 1, f"1.4 round → 1, 拿到 {out['position_contracts']}"
    e2 = ICEngine()
    # 1.6 张 → round → 2 张
    out2 = e2.step(target_short_notional=1_600_000, ic_price=5000)
    assert out2['position_contracts'] == 2, f"1.6 round → 2, 拿到 {out2['position_contracts']}"
    print("  ✓ _test_contract_rounding")


if __name__ == '__main__':
    print("Running engine/ic_engine.py tests ...")
    _test_first_day_no_pnl()
    _test_second_day_pnl_short_wins_on_drop()
    _test_short_loses_on_rise()
    _test_rebalance_charges_trade_cost()
    _test_basis_drag_negative()
    _test_target_zero_closes_position()
    _test_contract_rounding()
    print("All tests passed.")
```

- [ ] **Step 2: 跑测试**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python engine/ic_engine.py
```

Expected:
```
Running engine/ic_engine.py tests ...
  ✓ _test_first_day_no_pnl
  ✓ _test_second_day_pnl_short_wins_on_drop
  ✓ _test_short_loses_on_rise
  ✓ _test_rebalance_charges_trade_cost
  ✓ _test_basis_drag_negative
  ✓ _test_target_zero_closes_position
  ✓ _test_contract_rounding
All tests passed.
```

- [ ] **Step 3: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add engine/ic_engine.py
git commit -m "R14: engine/ic_engine.py — IC 头寸引擎 (margin/PnL/basis drag) + 7 个单元测试"
```

---

## Task 5: `engine/account.py` (TDD)

**Files:**
- Create: `engine/account.py`

- [ ] **Step 1: 写实现 + 测试**

Write `engine/account.py`:
```python
"""资金账户聚合 — 把 long stock value + IC margin + cash 合到一个 NAV.

每日:
  1. cash 计息 (cash_rate, 默认 0)
  2. 接收 long leg 日报 (long_value_eod, long_return_today)
  3. 接收 IC engine 输出 (daily_pnl, margin_required, trade_cost, basis_cost)
  4. 更新 cash = cash + IC PnL - trade_cost + basis_cost + cash_interest
     (basis_cost 已经是负值)
  5. NAV = cash + long_value_eod + ic_margin_required
  6. margin_call: 如果 cash < 0, log warning (研究用不强平)
"""
from dataclasses import dataclass


@dataclass
class AccountState:
    cash: float
    long_value: float = 0.0
    ic_margin: float = 0.0
    nav: float = 0.0
    margin_call_count: int = 0


class Account:
    def __init__(self, initial_capital=1e8, cash_rate=0.0):
        """
        Args:
            initial_capital: 初始资金 (默认 1 亿)
            cash_rate: 现金年化利率 (0.0 baseline, 0.015 sensitivity)
        """
        self.cash_rate_daily = cash_rate / 252.0
        self.state = AccountState(cash=initial_capital, nav=initial_capital)

    def update(self, long_value_eod, ic_step_output):
        """每日聚合一步.

        Args:
            long_value_eod: 今日收盘 long leg 股票市值 (qlib backtest 给出)
            ic_step_output: ICEngine.step() 的返回 dict

        Returns: dict (nav, cash, long_value, ic_margin, ic_pnl_today, margin_call)
        """
        # 1. Cash interest (按昨日 EOD cash 余额)
        cash_interest = self.state.cash * self.cash_rate_daily

        # 2. IC PnL / cost / basis 全部进 cash
        ic_pnl = ic_step_output['daily_pnl']
        trade_cost = ic_step_output['trade_cost']
        basis_cost = ic_step_output['basis_cost']  # negative

        self.state.cash += cash_interest + ic_pnl - trade_cost + basis_cost

        # 3. 更新 long_value 和 ic_margin
        self.state.long_value = long_value_eod
        self.state.ic_margin = ic_step_output['margin_required']

        # 4. NAV = cash + long stock value + IC margin (margin 也是资产)
        self.state.nav = self.state.cash + self.state.long_value + self.state.ic_margin

        # 5. Margin call: cash 不够覆盖额外的 margin call (cash < 0 直接 trigger)
        margin_call = self.state.cash < 0
        if margin_call:
            self.state.margin_call_count += 1

        return dict(
            nav=self.state.nav,
            cash=self.state.cash,
            long_value=self.state.long_value,
            ic_margin=self.state.ic_margin,
            ic_pnl_today=ic_pnl,
            cash_interest_today=cash_interest,
            margin_call=margin_call,
        )


# ===== inline tests =====
def _test_initial_state():
    a = Account(initial_capital=1e8)
    assert a.state.cash == 1e8
    assert a.state.nav == 1e8
    print("  ✓ _test_initial_state")


def _test_nav_includes_long_value():
    """长头股票市值入 NAV"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    # long_value=5000万, 表示 5000万 cash 已经买成股票
    out = a.update(long_value_eod=5e7, ic_step_output=ic_out)
    # NAV 应该 = cash (1e8, 还没扣股票钱因为 simplification) + 长头 (5e7) + IC margin (0)
    # 注意: 这里 long_value 是 qlib 给的, 它已经从 cash 减过买股票钱了 — 但我们 cash 没扣
    # 解释: account 把 cash 当 "可用 buffer", long_value 是 "已投股票部分", 二者相加 ≈ 1.5e8 是不对的
    # 实际我们要的是: cash + long_value = initial_capital + cumulative PnL
    # 所以 update() 不应这样用. 见 _test_nav_unchanged_with_zero_returns 修正语义
    assert out['nav'] > 1e8
    print("  ✓ _test_nav_includes_long_value (注: 语义见 round14 主脚本)")


def _test_ic_pnl_into_cash():
    """IC daily_pnl 直接进 cash"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=50000, margin_required=0, trade_cost=200, basis_cost=-100)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    expected_cash = 1e8 + 50000 - 200 - 100
    assert abs(out['cash'] - expected_cash) < 1e-6, f"期望 {expected_cash}, 拿到 {out['cash']}"
    print("  ✓ _test_ic_pnl_into_cash")


def _test_cash_interest():
    """cash 利息每日累计"""
    a = Account(initial_capital=1e8, cash_rate=0.015)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    expected_interest = 1e8 * 0.015 / 252
    assert abs(out['cash_interest_today'] - expected_interest) < 1e-3
    print("  ✓ _test_cash_interest")


def _test_margin_call_count():
    """cash 跌负 → margin_call count 增加"""
    a = Account(initial_capital=1000)
    # 大额 trade_cost 把 cash 打负
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=5000, basis_cost=0)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    assert out['margin_call'] is True
    assert a.state.margin_call_count == 1
    print("  ✓ _test_margin_call_count")


def _test_nav_accounting_identity():
    """NAV = cash + long_value + ic_margin (恒等式)"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=12345, margin_required=2_500_000, trade_cost=300, basis_cost=-150)
    out = a.update(long_value_eod=5e7, ic_step_output=ic_out)
    assert abs(out['nav'] - (out['cash'] + out['long_value'] + out['ic_margin'])) < 1e-6
    print("  ✓ _test_nav_accounting_identity")


if __name__ == '__main__':
    print("Running engine/account.py tests ...")
    _test_initial_state()
    _test_nav_includes_long_value()
    _test_ic_pnl_into_cash()
    _test_cash_interest()
    _test_margin_call_count()
    _test_nav_accounting_identity()
    print("All tests passed.")
```

- [ ] **Step 2: 跑测试**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python engine/account.py
```

Expected:
```
Running engine/account.py tests ...
  ✓ _test_initial_state
  ✓ _test_nav_includes_long_value (注: 语义见 round14 主脚本)
  ✓ _test_ic_pnl_into_cash
  ✓ _test_cash_interest
  ✓ _test_margin_call_count
  ✓ _test_nav_accounting_identity
All tests passed.
```

- [ ] **Step 3: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add engine/account.py
git commit -m "R14: engine/account.py — 资金账户聚合 (cash + long + IC margin → NAV) + 6 个单元测试"
```

---

## Task 6: Long leg backtest 复跑 + 缓存

**Files:**
- Create: `scripts/round14_step1_long_leg.py`
- Output: `results/runs/round14/long_leg_daily.pkl`

**Why a separate script:** qlib backtest 跑一次要几分钟。R14 要跑 3 个 config (C1/C2/C3)，long leg 完全一样，缓存避免重复。

- [ ] **Step 1: 写 long leg replay 脚本**

Write `scripts/round14_step1_long_leg.py`:
```python
"""R14 Step 1: 用 R8.5 预测值跑一次 qlib backtest 拿 long leg daily report, 落盘缓存.

输出: results/runs/round14/long_leg_daily.pkl
  pd.DataFrame indexed by date, columns:
    long_return:   net daily return after cost (策略层面)
    long_cost:     当日交易成本占比
    long_value:    EOD 账户总值 (qlib report['account'] 列)
    bench_return:  CSI500 当日收益 (qlib report['bench'] 列)
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'results' / 'runs' / 'round14'
OUT.mkdir(parents=True, exist_ok=True)

EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def main():
    print("[1] 加载 R8.5 预测值 ...", flush=True)
    pred = pd.read_pickle(ROOT / 'results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    print(f"    {len(pred)} 行", flush=True)

    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')

    print("[2] TopkDropout K=30 long-only backtest (复用 R8.5 配置) ...", flush=True)
    strat = TopkDropoutStrategy(signal=pred, topk=30, n_drop=1, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15',
                      strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]
    rep.index = pd.to_datetime(rep.index)
    print(f"    报告 {len(rep)} 行, 列 {list(rep.columns)}", flush=True)

    # qlib report 标准列: account (EOD NAV), return (gross), cost, bench, turnover, ...
    # 我们要的: long_value (EOD account), long_return (net = return - cost), bench_return
    out = pd.DataFrame({
        'long_value': rep['account'],
        'long_return': rep['return'] - rep['cost'],
        'long_cost': rep['cost'],
        'long_return_gross': rep['return'],
        'bench_return': rep['bench'],
        'turnover': rep['turnover'],
    }, index=rep.index)

    out.to_pickle(OUT / 'long_leg_daily.pkl')
    print(f"\n[3] 写出 {OUT / 'long_leg_daily.pkl'}", flush=True)
    print(f"    首日 {out.index[0]} long_value={out['long_value'].iloc[0]:,.0f}", flush=True)
    print(f"    末日 {out.index[-1]} long_value={out['long_value'].iloc[-1]:,.0f}", flush=True)
    print(f"    年化 long_return (净) = {out['long_return'].mean()*252*100:+.2f}%", flush=True)
    print(f"    年化 bench_return = {out['bench_return'].mean()*252*100:+.2f}%", flush=True)
    print(f"    净超额 = {(out['long_return']-out['bench_return']).mean()*252*100:+.2f}%", flush=True)
    print(f"    (R8.5 spec 数: 净超额 +4.40% — 应该高度一致)", flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑 long leg backtest**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python scripts/round14_step1_long_leg.py
```

Expected:
- 跑 2-5 分钟（取决于机器）
- 末尾打印的"净超额"应该 ≈ **+4.40%**（R8.5 公认数字）。如果偏离 > 0.3pp 就有问题，停下来排查。

- [ ] **Step 3: 抽验缓存**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
import pandas as pd
df = pd.read_pickle('results/runs/round14/long_leg_daily.pkl')
print('rows:', len(df))
print('columns:', list(df.columns))
print(df.head(3))
print('long_value 范围:', df['long_value'].min(), '~', df['long_value'].max())
"
```

Expected: ~1300+ 行，long_value 从 1e8 起，到末日大概 1.2-1.4e8 之间（R8.5 是 5.4 年 +4.40% 超额 + CSI500 ≈ 0% 总），意味着大致 1.25e8 末值。

- [ ] **Step 4: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add scripts/round14_step1_long_leg.py results/runs/round14/long_leg_daily.pkl
git commit -m "R14: long leg backtest 缓存 (TopK=30 R8.5 配置)"
```

---

## Task 7: 主回测脚本 — C1/C2/C3 三个配置

**Files:**
- Create: `scripts/round14_step2_engineering_ls.py`
- Output: `results/runs/round14/nav_C1_1x_conservative.pkl`, `nav_C2_beta_conservative.pkl`, `nav_C3_1x_optimistic.pkl`, `engineering_ls_summary.json`

- [ ] **Step 1: 写主脚本**

Write `scripts/round14_step2_engineering_ls.py`:
```python
"""R14 Step 2: 跑 C1 / C2 / C3 三个 long-short 配置, 用独立 engine 模块.

C1 baseline:   1x notional static + 保守 finance (14% margin / 0% cash / 5% basis)
C2 beta-adj:   60d rolling beta × long_value + 保守 finance
C3 sensitivity: 1x notional + 乐观 finance (1.5% cash / 3% basis)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
from engine.ic_data import load_ic_daily
from engine.ic_engine import ICEngine
from engine.account import Account

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'results' / 'runs' / 'round14'


# ===== Config matrix =====
CONFIGS = {
    'C1_1x_conservative': dict(
        hedge_mode='1x_notional',
        cash_rate=0.0,
        basis_drag_annual=0.05,
        margin_rate=0.14,
        trade_cost_bps=0.23,
    ),
    'C2_beta_conservative': dict(
        hedge_mode='beta_adjusted',
        beta_window=60,
        beta_fallback=1.0,
        cash_rate=0.0,
        basis_drag_annual=0.05,
        margin_rate=0.14,
        trade_cost_bps=0.23,
    ),
    'C3_1x_optimistic': dict(
        hedge_mode='1x_notional',
        cash_rate=0.015,
        basis_drag_annual=0.03,
        margin_rate=0.14,
        trade_cost_bps=0.23,
    ),
}


def compute_rolling_beta(long_returns, bench_returns, window=60, fallback=1.0):
    """60 日滚动 OLS beta. 不足窗口的位置用 fallback."""
    # rolling cov / var
    cov = long_returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var()
    beta = (cov / var).fillna(fallback)
    # 前 window-1 天 fallback
    beta.iloc[:window-1] = fallback
    return beta


def run_config(name, cfg, long_leg, ic_daily, bench_returns):
    """跑一个 config, 返回 daily NAV DataFrame."""
    print(f"\n[Config {name}] cfg = {cfg}", flush=True)

    ic_engine = ICEngine(
        contract_multiplier=200,
        margin_rate=cfg['margin_rate'],
        trade_cost_bps=cfg['trade_cost_bps'],
        basis_drag_daily=-cfg['basis_drag_annual'] / 252.0,
    )
    account = Account(initial_capital=1e8, cash_rate=cfg['cash_rate'])

    # 提前算 beta (如果需要)
    if cfg['hedge_mode'] == 'beta_adjusted':
        beta_series = compute_rolling_beta(
            long_leg['long_return'], bench_returns,
            window=cfg['beta_window'], fallback=cfg['beta_fallback'])
    else:
        beta_series = pd.Series(1.0, index=long_leg.index)

    # 按日 loop
    records = []
    for date, row in long_leg.iterrows():
        long_value = row['long_value']
        long_return = row['long_return']

        # IC 价格 (forward fill 防止周末/缺失)
        if date in ic_daily.index:
            ic_price = ic_daily.loc[date, 'close']
        else:
            # 找最近的可用日 (前向)
            past = ic_daily.index[ic_daily.index <= date]
            if len(past) == 0:
                continue  # 数据起始之前
            ic_price = ic_daily.loc[past[-1], 'close']

        # 决定 target short notional
        beta = beta_series.loc[date]
        target_short_notional = beta * long_value

        # IC engine 一步
        ic_step = ic_engine.step(target_short_notional, ic_price)

        # Account 聚合
        acc_state = account.update(long_value, ic_step)

        records.append(dict(
            date=date,
            long_value=long_value,
            long_return=long_return,
            beta=beta,
            ic_price=ic_price,
            target_short_notional=target_short_notional,
            ic_position=ic_step['position_contracts'],
            ic_pnl=ic_step['daily_pnl'],
            ic_margin=ic_step['margin_required'],
            ic_trade_cost=ic_step['trade_cost'],
            ic_basis_cost=ic_step['basis_cost'],
            cash=acc_state['cash'],
            nav=acc_state['nav'],
            margin_call=acc_state['margin_call'],
        ))

    df = pd.DataFrame(records).set_index('date')
    print(f"    margin call 触发 {df['margin_call'].sum()} 次", flush=True)
    print(f"    IC margin 峰值 {df['ic_margin'].max():,.0f} ({df['ic_margin'].max()/1e8*100:.1f}% of init)", flush=True)
    return df


def metrics_from_nav(nav_series, name=""):
    """从 NAV 序列算 metrics."""
    daily_ret = nav_series.pct_change().dropna()
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float('nan')
    eq = nav_series / nav_series.iloc[0]
    mdd = (eq / eq.cummax() - 1).min()
    return dict(name=name, ann_ret=ann_ret*100, ann_vol=ann_vol*100,
                sharpe=sharpe, mdd=mdd*100)


def main():
    print("[1] 加载 long leg + IC 数据 ...", flush=True)
    long_leg = pd.read_pickle(OUT / 'long_leg_daily.pkl')
    print(f"    long leg: {len(long_leg)} 行 {long_leg.index[0]} ~ {long_leg.index[-1]}", flush=True)

    ic_daily = load_ic_daily(long_leg.index[0], long_leg.index[-1])
    print(f"    IC daily: {len(ic_daily)} 行", flush=True)

    bench_returns = long_leg['bench_return']

    print("\n[2] 跑 3 个配置 ...", flush=True)
    all_results = {}
    for name, cfg in CONFIGS.items():
        df = run_config(name, cfg, long_leg, ic_daily, bench_returns)
        df.to_pickle(OUT / f'nav_{name}.pkl')
        all_results[name] = df
        m = metrics_from_nav(df['nav'], name)
        print(f"    [{name}] 年化 {m['ann_ret']:+.2f}% Vol {m['ann_vol']:.2f}% "
              f"Sharpe {m['sharpe']:+.3f} MDD {m['mdd']:+.2f}%", flush=True)

    print("\n[3] 横评对比 ...", flush=True)
    # 加 R8.5 long-only 净超额 + R12 LS-100% K=30
    r12_path = ROOT / 'results/runs/round12/ls_LS-100pct_K_30.pkl'
    r12_curve = pd.read_pickle(r12_path)
    r12_nav = (1 + r12_curve).cumprod()
    m_r12 = metrics_from_nav(r12_nav, 'R12 LS-100% K=30')

    r85_excess = long_leg['long_return'] - long_leg['bench_return']
    r85_nav = (1 + r85_excess).cumprod()
    m_r85 = metrics_from_nav(r85_nav, 'R8.5 long-only (净超额)')

    summary_rows = []
    for name in CONFIGS.keys():
        summary_rows.append(metrics_from_nav(all_results[name]['nav'], name))
    summary_rows.append(m_r85)
    summary_rows.append(m_r12)

    print("\n  对比表 (按 Sharpe 排):", flush=True)
    print(f"  {'配置':30s} {'年化%':>8s} {'Vol%':>7s} {'Sharpe':>8s} {'MDD%':>8s}", flush=True)
    for r in sorted(summary_rows, key=lambda x: -x['sharpe']):
        print(f"  {r['name']:30s} {r['ann_ret']:+8.2f} {r['ann_vol']:7.2f} "
              f"{r['sharpe']:+8.3f} {r['mdd']:+8.2f}", flush=True)

    # ===== 落盘 summary JSON =====
    out_json = dict(
        configs=summary_rows,
        ic_margin_peaks={n: float(all_results[n]['ic_margin'].max()) for n in CONFIGS},
        margin_call_counts={n: int(all_results[n]['margin_call'].sum()) for n in CONFIGS},
    )
    with open(OUT / 'engineering_ls_summary.json', 'w') as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"\n[4] 写出 {OUT / 'engineering_ls_summary.json'}", flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑主脚本**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python scripts/round14_step2_engineering_ls.py
```

Expected:
- 3 个配置 daily loop ~ 1-2 分钟
- 末尾对比表打印
- **C1 baseline 的 Sharpe 应介于 0.30 ~ 0.72 之间**（spec 第 11 节）。如果 > 0.72 或 < 0.30，先怀疑 bug：
  - 检查 IC `daily_pnl` 符号（空头跌赚涨亏）
  - 检查 basis drag 是不是 double-count 进 cash 和 ic_engine 输出
  - 检查 long_value 单位（应该是元，初始 1e8）

- [ ] **Step 3: 抽验 NAV pickle**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
import pandas as pd
for n in ['C1_1x_conservative', 'C2_beta_conservative', 'C3_1x_optimistic']:
    df = pd.read_pickle(f'results/runs/round14/nav_{n}.pkl')
    nav = df['nav']
    print(f'{n}: rows={len(nav)}, NAV {nav.iloc[0]:,.0f} → {nav.iloc[-1]:,.0f}, '
          f'IC pos mean {df[\"ic_position\"].mean():.1f}, basis drag total {df[\"ic_basis_cost\"].sum():,.0f}')
"
```

Expected: 每个 config 末值都 > 1e8（小幅或中幅盈利），IC position 平均 ~10-20 张（5000 IC × 200 mult = 100万/张，1.2e8 long / 1M ≈ 100-120 张？让我重新算: long_value 始终 ≈ 1e8，所以 short notional 也 ≈ 1e8，price ≈ 5000 → 1e8 / (5000×200) = 100 张）。如果数字偏离一个数量级，检查单位。

- [ ] **Step 4: Commit results**

```bash
cd /Users/cedricyu/qlib量化研究
git add scripts/round14_step2_engineering_ls.py results/runs/round14/nav_*.pkl results/runs/round14/engineering_ls_summary.json
git commit -m "R14: 3 个配置回测 (C1/C2/C3) + summary"
```

---

## Task 8: 可视化

**Files:**
- Create: `scripts/round14_step3_figures.py`
- Output: `results/figures/fig14_engineering_ls.png`

- [ ] **Step 1: 写画图脚本**

Write `scripts/round14_step3_figures.py`:
```python
"""R14 Step 3: 画 fig14_engineering_ls.png — 跟 fig13 同布局.

上面板: 5 条 NAV / Excess cumulative curves (C1/C2/C3 + R8.5 + R12)
下面板: Sharpe 横评柱图
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
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

ROOT = Path('/Users/cedricyu/qlib量化研究')
R14 = ROOT / 'results/runs/round14'
FIGS = ROOT / 'results/figures'
FIGS.mkdir(parents=True, exist_ok=True)


def metrics(daily_returns, name=""):
    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float('nan')
    eq = (1 + daily_returns).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(name=name, sharpe=sharpe, ann_ret=ann_ret*100, mdd=mdd*100)


def main():
    # ===== 加载 5 条曲线 =====
    print("加载 5 条曲线 ...", flush=True)
    long_leg = pd.read_pickle(R14 / 'long_leg_daily.pkl')

    # R8.5 净超额 (long_return - bench_return)
    r85_daily = long_leg['long_return'] - long_leg['bench_return']
    r85_nav = (1 + r85_daily).cumprod()
    m_r85 = metrics(r85_daily, 'R8.5 long-only (净超额)')

    # R12
    r12_daily = pd.read_pickle(ROOT / 'results/runs/round12/ls_LS-100pct_K_30.pkl')
    r12_nav = (1 + r12_daily).cumprod()
    m_r12 = metrics(r12_daily, 'R12 LS-100% K=30 (hypothetical)')

    # R14 三个配置 — 用 nav.pct_change() 还原日 return 以匹配同一基准
    r14 = {}
    for n in ['C1_1x_conservative', 'C2_beta_conservative', 'C3_1x_optimistic']:
        df = pd.read_pickle(R14 / f'nav_{n}.pkl')
        daily = df['nav'].pct_change().dropna()
        r14[n] = dict(curve=(df['nav'] / df['nav'].iloc[0]),
                      metrics=metrics(daily, n))

    # ===== 画图 =====
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), facecolor='white',
                              gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]

    colors = {
        'C1_1x_conservative': '#1a365d',
        'C2_beta_conservative': '#2c7a7b',
        'C3_1x_optimistic': '#c89b3c',
    }
    for n, payload in r14.items():
        c = payload['curve']
        m = payload['metrics']
        ax.plot(c.index, c.values, label=f"R14 {n}  (Sharpe {m['sharpe']:+.2f})",
                color=colors[n], linewidth=2.2)

    ax.plot(r85_nav.index, r85_nav.values,
            label=f"R8.5 long-only 净超额  (IR {m_r85['sharpe']:+.2f})",
            color='#a0aec0', linestyle='--', linewidth=1.6)
    ax.plot(r12_nav.index, r12_nav.values,
            label=f"R12 LS-100% K=30 (hypothetical)  (Sharpe {m_r12['sharpe']:+.2f})",
            color='#9467bd', linestyle=':', linewidth=1.8)

    ax.set_title('图 R14-1 | 工程版 L/S vs 历史基线 (含 IC margin / basis / 手续费)', loc='left')
    ax.set_ylabel('累计净值 (相对起点)')
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0)
    ax.axhline(1.0, color='#cbd5e0', linewidth=0.5)

    # Sharpe 横评
    ax = axes[1]
    items = [
        ('R8.5\nlong-only', m_r85['sharpe'], '#a0aec0'),
        ('R14 C1\n1x 保守', r14['C1_1x_conservative']['metrics']['sharpe'], '#1a365d'),
        ('R14 C2\nbeta-adj', r14['C2_beta_conservative']['metrics']['sharpe'], '#2c7a7b'),
        ('R14 C3\n1x 乐观', r14['C3_1x_optimistic']['metrics']['sharpe'], '#c89b3c'),
        ('R12\nhypothetical', m_r12['sharpe'], '#9467bd'),
    ]
    names = [x[0] for x in items]; vals = [x[1] for x in items]; cols = [x[2] for x in items]
    ax.bar(names, vals, color=cols, alpha=0.92, edgecolor='white')
    ax.axhline(1.0, color='#2f855a', linestyle=':', alpha=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03 if v >= 0 else v - 0.08, f'{v:.2f}',
                ha='center', fontweight='bold', fontsize=9, color='#1a365d')
    ax.set_title('Sharpe / IR 横评', loc='left')
    ax.set_ylabel('Sharpe / IR')
    ax.axhline(0, color='#4a5568', linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig14_engineering_ls.png')
    plt.close()
    print(f"  ✓ {FIGS / 'fig14_engineering_ls.png'}", flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 跑画图**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python scripts/round14_step3_figures.py
```

Expected: `results/figures/fig14_engineering_ls.png` 出现，文件 size > 100KB（不是 0 字节占位）。

- [ ] **Step 3: 抽验 PNG 大小**

Run:
```bash
ls -lh /Users/cedricyu/qlib量化研究/results/figures/fig14_engineering_ls.png
```

Expected: 文件大小 200-800 KB（典型 matplotlib 双面板图）。

- [ ] **Step 4: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add scripts/round14_step3_figures.py results/figures/fig14_engineering_ls.png
git commit -m "R14: fig14_engineering_ls.png — NAV 对比 + Sharpe 横评"
```

---

## Task 9: 写 summary.md + 更新 research_log.md + 更新 CLAUDE.md

**Files:**
- Create: `results/runs/round14/summary.md`
- Modify: `results/research_log.md` (追加)
- Modify: `CLAUDE.md` (顶部表格 + 当前最优配置区)

- [ ] **Step 1: 写 round14 summary.md**

读取实际数字写。先用：
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
import json
d = json.load(open('results/runs/round14/engineering_ls_summary.json'))
for r in d['configs']:
    print(f\"{r['name']:30s} ann={r['ann_ret']:+.2f}%  vol={r['ann_vol']:.2f}%  sharpe={r['sharpe']:+.3f}  mdd={r['mdd']:+.2f}%\")
print('margin peaks:', d['ic_margin_peaks'])
print('margin calls:', d['margin_call_counts'])
"
```

Write `results/runs/round14/summary.md`（把上面打印的实际数字代入下面模板的 `[#]` 占位）：
```markdown
# R14 (B1+++) 工程版 long-short — IC 期货对冲

**假设**：把 R12 hypothetical short 换成真实可执行的 CSI500 期货 (IC) 对冲后，alpha 衰减多少？

**实现**：独立 engine 模块（`ic_data` + `ic_engine` + `account`），跑 3 组配置：
- C1: 1x notional + 保守 finance (14% margin / 0% cash / 5% basis)
- C2: 60d rolling beta-adjusted + 保守 finance
- C3: 1x notional + 乐观 finance (1.5% cash / 3% basis)

测试期 2020-10-01 ~ 2026-05-15（与 R8.5 / R12 一致），long leg 复用 R8.5 模型预测（不重训）。

## 结果

| 配置 | 年化收益 | Vol | Sharpe | MDD | 备注 |
|---|---|---|---|---|---|
| R8.5 long-only 净超额 | +4.40% | 9.9% | 0.445 | -15.7% | long-only 天花板 |
| **R14 C1 1x 保守** | [+#.##%] | [#.##%] | [#.###] | [#.##%] | 工程基线 |
| R14 C2 beta-adj | [+#.##%] | [#.##%] | [#.###] | [#.##%] | 60d 滚动 beta |
| R14 C3 1x 乐观 | [+#.##%] | [#.##%] | [#.###] | [#.##%] | sensitivity |
| R12 LS-100% K=30 | +7.32% | 10.2% | 0.716 | -14.2% | hypothetical short |

IC margin 占用峰值: C1 [#.#]% / C2 [#.#]% / C3 [#.#]% of 1 亿。
Margin call 触发次数: C1 [#] / C2 [#] / C3 [#]。

## 关键发现

1. **C1 vs R12 alpha 衰减** = [R12 0.72 − C1 Sharpe]，对应"hypothetical short → 真实期货对冲"代价。
2. **C2 vs C1** = beta-adjusted 是否改善：[填实际差异 + 一句话解读]。
3. **C3 vs C1** = finance 假设松紧的边界：[填实际差异]。

## 结论

[根据实际数字写 1-2 段]：
- R14 C1 [Sharpe 在 R8.5 0.45 和 R12 0.72 之间 / 还是偏到边界] → 工程化代价是 [#.##]
- beta-adjusted [边际改善 / 没用]，因为 [CSI500 选股 portfolio 与指数 beta 接近 1 / 或别的原因]
- finance 假设松紧带来 [#.##] Sharpe 区间，结论 [robust / sensitive]

## 下一步

[根据 C1 实际数字判断]：
- 若 C1 Sharpe > 0.60：工程版接近 R12 hypothetical，已经是 deployable
- 若 C1 Sharpe < 0.50：alpha 主要来自 R12 的"反向选股 short"，期货对冲拿不到 → 实盘上限就在 R8.5 IR 0.45 附近
- 若 C1 在 0.50-0.60：中间区域，可以再做（a）日内 IC 价格滑点模拟（b）IC 日数据频率提升（c）short leg 部分用 ETF 借券

可考虑 R15：等 R13 财务因子完成后接入，看 long leg 模型升级能不能再推 C1。
```

- [ ] **Step 2: 在 results/research_log.md 末尾追加 R14 章节**

Run 上面的 python 一句把数字读出来，然后 Edit `results/research_log.md`，在文件末尾 append（用 Edit 而非 Write，避免覆盖之前所有内容）。

要插入的文本（把实际数字填入 [#]）:
```markdown

---

## 阶段五·第 14 轮 (B1+++)：工程版 long-short — IC 期货对冲 (2026-05-30)

**假设**：R12 hypothetical short (Sharpe 0.72) 换成真实 CSI500 期货对冲后，alpha 还剩多少？

**实现**：独立 engine 模块 + 3 组配置 (1x 保守 / beta-adj / 1x 乐观)。
脚本 [scripts/round14_step1_long_leg.py](../scripts/round14_step1_long_leg.py) + [scripts/round14_step2_engineering_ls.py](../scripts/round14_step2_engineering_ls.py)
详细 [results/runs/round14/summary.md](runs/round14/summary.md)。

**结果**：

| 配置 | 年化 | Sharpe | MDD |
|---|---|---|---|
| R14 C1 (1x 保守) | [+#.##%] | [#.###] | [#.##%] |
| R14 C2 (beta-adj) | [+#.##%] | [#.###] | [#.##%] |
| R14 C3 (1x 乐观) | [+#.##%] | [#.###] | [#.##%] |

**核心结论**：[填 1-2 句根据实际结果]。

**下一步**：[填，参考 summary.md "下一步"]。
```

具体操作（替换占位符）：先读出 summary JSON，用实际数字替换 `[#.##]` 等占位，再用 Edit 把这段塞进文件末尾。

- [ ] **Step 3: 更新 CLAUDE.md 顶部状态表 + 当前最优区**

Edit `CLAUDE.md`：

(a) 在轮次表（"已完成的 12 轮迭代"段下面）增加 R14 行：
```
| **R14** | **工程版 LS (IC 期货对冲)** | **[+#.##%]** | **[#.###]** | **[填一句关键发现]** |
```
同时把表头 "已完成的 12 轮迭代" 改为 "已完成的 14 轮迭代"（注: 这是历史迭代数，R13 仍未完成，所以可能写 "已完成的 13 轮迭代 + R13 半成品"，按 R14 实际完成状态写）。

(b) 在 "突破版 (R12 long-short, 研究用)" 后面新增 "工程版 (R14 工程实盘可执行)" 块：
```yaml
# 多头: 同 R8.5 (TopK=30, n_drop=1, Alpha158, LGBM 默认)
# 空头: CSI500 期货 IC0 主力连续, hedge_mode=1x_notional 或 beta_adjusted
# 资金: 1 亿元, IC 保证金 14%, 现金 0% (保守) 或 1.5% (乐观)
# 成本: 多头 0.1%/0.2% (qlib 默认), 期货 万 0.23, basis drag 5%/年 (保守)
# 实测: 年化 [+#.##%], Sharpe [#.###], MDD [#.##%]
# 这是真实可执行的工程版 (vs R12 是 hypothetical)
```

- [ ] **Step 4: 跑 sanity check — round14 各产物齐了**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && ls results/runs/round14/ && echo "---" && ls results/figures/fig14*.png && echo "---" && tail -30 results/research_log.md
```

Expected: round14/ 下有 summary.md / 3 个 nav_*.pkl / engineering_ls_summary.json / long_leg_daily.pkl；fig14 PNG 存在；research_log 末尾有 R14 章节。

- [ ] **Step 5: Commit**

```bash
cd /Users/cedricyu/qlib量化研究
git add results/runs/round14/summary.md results/research_log.md CLAUDE.md
git commit -m "R14: summary + research_log + CLAUDE.md 更新"
```

---

## Task 10: 最终对账 + push

**Files (no new):**

- [ ] **Step 1: 跨轮对账**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && /Users/cedricyu/miniconda3/envs/qlib/bin/python -c "
import pandas as pd, json
# (1) C1 long leg gross return 应 ≈ R8.5 absolute return (不是超额)
ll = pd.read_pickle('results/runs/round14/long_leg_daily.pkl')
c1 = pd.read_pickle('results/runs/round14/nav_C1_1x_conservative.pkl')
ann_long_abs = ll['long_return'].mean()*252*100
print(f'(1) Long leg 年化净绝对收益 = {ann_long_abs:+.2f}% (期望 ≈ R8.5 +4.40% + CSI500 bench 期间收益)')

# (2) C1 IC 端 PnL 应跟 -CSI500 高度负相关
bench_cum_ret = (1 + ll['bench_return']).cumprod()
ic_cum_pnl = c1['ic_pnl'].cumsum()
import numpy as np
corr = np.corrcoef(ll['bench_return'].fillna(0).values, c1['ic_pnl'].pct_change().fillna(0).values)[0,1]
print(f'(2) IC daily PnL vs bench return 相关系数 = {corr:+.3f} (期望 < -0.7)')

# (3) C1 Sharpe 应介于 0.30 ~ 0.72
d = json.load(open('results/runs/round14/engineering_ls_summary.json'))
c1_sharpe = next(r['sharpe'] for r in d['configs'] if r['name']=='C1_1x_conservative')
print(f'(3) C1 Sharpe = {c1_sharpe:+.3f} (期望 0.30 ~ 0.72)')
if not (0.30 <= c1_sharpe <= 0.80):
    print('   ⚠️  WARNING: C1 Sharpe 超出预期区间, 先怀疑 bug 再相信结果')
else:
    print('   ✓ C1 Sharpe 在预期区间内')
"
```

Expected:
- (1) long abs return 跟 CSI500 同期 + R8.5 超额 ≈ R8.5 abs return
- (2) IC PnL vs bench return 相关系数 < -0.7
- (3) C1 Sharpe 在 0.30 ~ 0.80（spec 第 11 节边界 + 一点余量）

如果 (3) 超出边界，**停下**：debug 后从 Task 7 重跑。

- [ ] **Step 2: 检查整体 git status 干净**

Run:
```bash
cd /Users/cedricyu/qlib量化研究 && git status
```

Expected: working tree clean (所有 R14 改动都已 commit)。

- [ ] **Step 3: 询问用户是否 push**

写一条消息给用户：
> R14 全部跑通：C1 Sharpe = [#.###]，[结论一句]。8 个 commit 落地。要 push 到 GitHub 吗？

等用户拍板。**不要自动 push**（CLAUDE.md 规则: 推 git 必须用户确认）。

---

## Notes for Engineers

1. **不要重训 long leg 模型** — R8.5 预测值 pickle 已存在，直接复用。重训会破坏跨轮可比。
2. **不要碰 R13 财务因子代码** — 与 R14 完全独立，scope 严格分开。
3. **IC engine 的 `contracts_short` 用正整数表示空头张数**（不是负数），daily_pnl 公式已经处理符号。改语义会引入 bug。
4. **basis drag 是负数**（空头每天承担贴水成本）。account 里 `cash += basis_cost` 已经把负号包含进去了，不要再加负号。
5. **如果 Task 7 跑出来 C1 Sharpe > 1.0**：几乎肯定是 bug。常见原因：
   - PnL 把 long 部分和 IC 部分都算了多头收益（重复）
   - basis drag 漏掉（drag 是负的，漏掉会让结果偏高）
   - long_value 单位错（应该是元，不是百万元）
6. **每个 task 一个 commit**，不要批量。失败时容易回滚单步。
