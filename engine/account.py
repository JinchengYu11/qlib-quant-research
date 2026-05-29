"""资金账户聚合 — 把 long stock value + IC margin + cash 合到一个 NAV.

语义 (修正版, 避免 double-count):
  - 初始: cash = initial_capital, long_value = 0, ic_margin = 0, NAV = initial_capital
  - 首次 update(long_value > 0): 视为"用 cash 买入 long stocks", cash -= long_value
    (qlib 已经把整个资金跑成 long-only 组合, long_value_eod 实际是组合 NAV)
  - 后续每天:
      cash += cash_interest + ic_pnl - trade_cost + basis_cost
      long_value = long_value_eod  (qlib 给, 市场起伏不流过 cash)
      ic_margin = ic_step.margin_required
      NAV = cash + long_value + ic_margin  (margin 也是我们的钱, 锁住但属于我们)
  - margin_call: cash < 0 → 没钱付保证金的额外占用 (研究用 log, 不强平)
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
        self.cash_rate_daily = cash_rate / 252.0
        self.state = AccountState(cash=initial_capital, nav=initial_capital)
        self._first_allocation_done = False

    def update(self, long_value_eod, ic_step_output):
        if not self._first_allocation_done and long_value_eod > 0:
            self.state.cash -= long_value_eod
            self._first_allocation_done = True

        cash_interest = self.state.cash * self.cash_rate_daily

        ic_pnl = ic_step_output['daily_pnl']
        trade_cost = ic_step_output['trade_cost']
        basis_cost = ic_step_output['basis_cost']

        self.state.cash += cash_interest + ic_pnl - trade_cost + basis_cost

        self.state.long_value = long_value_eod
        self.state.ic_margin = ic_step_output['margin_required']

        self.state.nav = self.state.cash + self.state.long_value + self.state.ic_margin

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


def _test_initial_state():
    a = Account(initial_capital=1e8)
    assert a.state.cash == 1e8
    assert a.state.nav == 1e8
    print("  ✓ _test_initial_state")


def _test_first_allocation_preserves_nav():
    """首次 long_value=5000万 → cash 扣 5000万, NAV 不变保持 1 亿"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=5e7, ic_step_output=ic_out)
    assert abs(out['cash'] - 5e7) < 1e-6, f"cash 应扣到 5e7, 拿到 {out['cash']}"
    assert abs(out['nav'] - 1e8) < 1e-6, f"NAV 应保持 1e8, 拿到 {out['nav']}"
    print("  ✓ _test_first_allocation_preserves_nav")


def _test_second_day_market_move_no_cash_change():
    """第二天 long_value 涨到 5500万 (市场起伏), cash 不变"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    a.update(long_value_eod=5e7, ic_step_output=ic_out)
    out = a.update(long_value_eod=5.5e7, ic_step_output=ic_out)
    assert abs(out['cash'] - 5e7) < 1e-6, f"cash 应保持 5e7 (市场涨不流过 cash), 拿到 {out['cash']}"
    assert abs(out['nav'] - 1.05e8) < 1e-6, f"NAV 应是 1.05e8, 拿到 {out['nav']}"
    print("  ✓ _test_second_day_market_move_no_cash_change")


def _test_ic_pnl_into_cash():
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=50000, margin_required=0, trade_cost=200, basis_cost=-100)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    expected_cash = 1e8 + 50000 - 200 - 100
    assert abs(out['cash'] - expected_cash) < 1e-6, f"期望 {expected_cash}, 拿到 {out['cash']}"
    print("  ✓ _test_ic_pnl_into_cash")


def _test_cash_interest_on_buffer():
    """初始 1e8, long=5000万, cash buffer = 5000万, 利息按 5000万 算"""
    a = Account(initial_capital=1e8, cash_rate=0.015)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=5e7, ic_step_output=ic_out)
    expected_interest = 5e7 * 0.015 / 252
    assert abs(out['cash_interest_today'] - expected_interest) < 1e-3, \
        f"期望 {expected_interest}, 拿到 {out['cash_interest_today']}"
    print("  ✓ _test_cash_interest_on_buffer")


def _test_margin_call_count():
    """cash 跌负 → margin_call count 增加"""
    a = Account(initial_capital=1000)
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


def _test_full_allocation_yields_zero_cash():
    """如果首日 long_value = initial_capital, 那 cash 应为 0"""
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=1e8, ic_step_output=ic_out)
    assert abs(out['cash']) < 1e-6, f"cash 应为 0, 拿到 {out['cash']}"
    assert abs(out['nav'] - 1e8) < 1e-6
    print("  ✓ _test_full_allocation_yields_zero_cash")


if __name__ == '__main__':
    print("Running engine/account.py tests ...")
    _test_initial_state()
    _test_first_allocation_preserves_nav()
    _test_second_day_market_move_no_cash_change()
    _test_ic_pnl_into_cash()
    _test_cash_interest_on_buffer()
    _test_margin_call_count()
    _test_nav_accounting_identity()
    _test_full_allocation_yields_zero_cash()
    print("All tests passed.")
