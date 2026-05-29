"""资金账户聚合 — 把 long stock value + IC margin + cash 合到一个 NAV.

每日:
  1. cash 计息 (cash_rate, 默认 0)
  2. 接收 long leg 日报 (long_value_eod, long_return_today)
  3. 接收 IC engine 输出 (daily_pnl, margin_required, trade_cost, basis_cost)
  4. 更新 cash = cash + IC PnL - trade_cost + basis_cost + cash_interest
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
        self.cash_rate_daily = cash_rate / 252.0
        self.state = AccountState(cash=initial_capital, nav=initial_capital)

    def update(self, long_value_eod, ic_step_output):
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


def _test_nav_includes_long_value():
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=5e7, ic_step_output=ic_out)
    assert out['nav'] > 1e8
    print("  ✓ _test_nav_includes_long_value")


def _test_ic_pnl_into_cash():
    a = Account(initial_capital=1e8)
    ic_out = dict(daily_pnl=50000, margin_required=0, trade_cost=200, basis_cost=-100)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    expected_cash = 1e8 + 50000 - 200 - 100
    assert abs(out['cash'] - expected_cash) < 1e-6, f"期望 {expected_cash}, 拿到 {out['cash']}"
    print("  ✓ _test_ic_pnl_into_cash")


def _test_cash_interest():
    a = Account(initial_capital=1e8, cash_rate=0.015)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=0, basis_cost=0)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    expected_interest = 1e8 * 0.015 / 252
    assert abs(out['cash_interest_today'] - expected_interest) < 1e-3
    print("  ✓ _test_cash_interest")


def _test_margin_call_count():
    a = Account(initial_capital=1000)
    ic_out = dict(daily_pnl=0, margin_required=0, trade_cost=5000, basis_cost=0)
    out = a.update(long_value_eod=0, ic_step_output=ic_out)
    assert out['margin_call'] is True
    assert a.state.margin_call_count == 1
    print("  ✓ _test_margin_call_count")


def _test_nav_accounting_identity():
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
