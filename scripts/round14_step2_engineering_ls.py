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
    cov = long_returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var()
    beta = (cov / var).fillna(fallback)
    beta.iloc[:window-1] = fallback
    return beta


def run_config(name, cfg, long_leg, ic_daily, bench_returns):
    print(f"\n[Config {name}] cfg = {cfg}", flush=True)

    ic_engine = ICEngine(
        contract_multiplier=200,
        margin_rate=cfg['margin_rate'],
        trade_cost_bps=cfg['trade_cost_bps'],
        basis_drag_daily=-cfg['basis_drag_annual'] / 252.0,
    )
    account = Account(initial_capital=1e8, cash_rate=cfg['cash_rate'])

    if cfg['hedge_mode'] == 'beta_adjusted':
        beta_series = compute_rolling_beta(
            long_leg['long_return'], bench_returns,
            window=cfg['beta_window'], fallback=cfg['beta_fallback'])
    else:
        beta_series = pd.Series(1.0, index=long_leg.index)

    records = []
    for date, row in long_leg.iterrows():
        long_value = row['long_value']
        long_return = row['long_return']

        if date in ic_daily.index:
            ic_price = ic_daily.loc[date, 'close']
        else:
            past = ic_daily.index[ic_daily.index <= date]
            if len(past) == 0:
                continue
            ic_price = ic_daily.loc[past[-1], 'close']

        beta = beta_series.loc[date]
        target_short_notional = beta * long_value

        ic_step = ic_engine.step(target_short_notional, ic_price)
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
