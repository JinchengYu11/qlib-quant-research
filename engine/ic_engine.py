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
from dataclasses import dataclass


@dataclass
class ICEngineState:
    contracts_short: int = 0
    last_close: float = 0.0


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
        """处理一个交易日."""
        target_contracts = int(round(target_short_notional / (ic_price * self.M)))
        delta_contracts = target_contracts - self.state.contracts_short

        if self.state.last_close > 0:
            daily_pnl = self.state.contracts_short * (self.state.last_close - ic_price) * self.M
        else:
            daily_pnl = 0.0

        trade_notional = abs(delta_contracts) * ic_price * self.M
        trade_cost = trade_notional * self.trade_cost_bps / 1e4

        self.state.contracts_short = target_contracts
        eod_notional = abs(target_contracts) * ic_price * self.M
        margin_required = eod_notional * self.margin_rate

        basis_cost = eod_notional * self.basis_drag_daily

        self.state.last_close = ic_price

        return dict(
            delta_contracts=delta_contracts,
            position_contracts=self.state.contracts_short,
            daily_pnl=daily_pnl,
            margin_required=margin_required,
            trade_cost=trade_cost,
            basis_cost=basis_cost,
        )


def _test_first_day_no_pnl():
    e = ICEngine()
    out = e.step(target_short_notional=1_000_000, ic_price=5000)
    assert out['delta_contracts'] == 1, f"期望 1 张, 拿到 {out['delta_contracts']}"
    assert out['position_contracts'] == 1
    assert out['daily_pnl'] == 0.0, "第一天不应有 PnL"
    assert abs(out['margin_required'] - 1_000_000 * 0.14) < 1e-6
    print("  ✓ _test_first_day_no_pnl")


def _test_second_day_pnl_short_wins_on_drop():
    e = ICEngine()
    e.step(target_short_notional=1_000_000, ic_price=5000)
    out = e.step(target_short_notional=1_000_000, ic_price=4900)
    assert abs(out['daily_pnl'] - 20_000) < 1e-6, f"期望 +20000, 拿到 {out['daily_pnl']}"
    assert out['delta_contracts'] == 0, f"期望不调仓, 拿到 {out['delta_contracts']}"
    print("  ✓ _test_second_day_pnl_short_wins_on_drop")


def _test_short_loses_on_rise():
    e = ICEngine()
    e.step(target_short_notional=10_000_000, ic_price=5000)
    out = e.step(target_short_notional=10_000_000, ic_price=5050)
    assert abs(out['daily_pnl'] - (-100_000)) < 1e-6, f"期望 -100000, 拿到 {out['daily_pnl']}"
    print("  ✓ _test_short_loses_on_rise")


def _test_rebalance_charges_trade_cost():
    e = ICEngine(trade_cost_bps=0.23)
    out = e.step(target_short_notional=10_000_000, ic_price=5000)
    assert abs(out['trade_cost'] - 230.0) < 1e-6, f"期望 230, 拿到 {out['trade_cost']}"
    print("  ✓ _test_rebalance_charges_trade_cost")


def _test_basis_drag_negative():
    e = ICEngine(basis_drag_daily=-0.05/252)
    out = e.step(target_short_notional=10_000_000, ic_price=5000)
    expected = 10_000_000 * (-0.05/252)
    assert abs(out['basis_cost'] - expected) < 1e-6, f"期望 {expected}, 拿到 {out['basis_cost']}"
    assert out['basis_cost'] < 0
    print("  ✓ _test_basis_drag_negative")


def _test_target_zero_closes_position():
    e = ICEngine()
    e.step(target_short_notional=10_000_000, ic_price=5000)
    out = e.step(target_short_notional=0, ic_price=5000)
    assert out['delta_contracts'] == -10, f"期望 -10, 拿到 {out['delta_contracts']}"
    assert out['position_contracts'] == 0
    assert out['margin_required'] == 0
    print("  ✓ _test_target_zero_closes_position")


def _test_contract_rounding():
    e = ICEngine()
    out = e.step(target_short_notional=1_400_000, ic_price=5000)
    assert out['position_contracts'] == 1, f"1.4 round → 1, 拿到 {out['position_contracts']}"
    e2 = ICEngine()
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
