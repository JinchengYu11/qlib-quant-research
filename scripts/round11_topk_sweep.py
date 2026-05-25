"""第11轮 (B1+): topk 扫描，验证 R10 发现的"覆盖 Q4+Q5 更稳"。
复用 R8.5 预测值，跑 topk × n_drop 网格。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
import qlib
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor

OUT = Path('/Users/cedricyu/qlib量化研究/results/runs/round11')
OUT.mkdir(parents=True, exist_ok=True)
EXCHANGE = dict(limit_threshold=0.095, deal_price='close',
                open_cost=0.001, close_cost=0.002, min_cost=5)


def ann(x): return x.mean() * 252 * 100
def vol(x): return x.std() * np.sqrt(252) * 100
def mdd(x):
    eq=(1+x).cumprod(); return (eq/eq.cummax()-1).min()*100


def run_bt(pred, topk, n_drop):
    strat = TopkDropoutStrategy(signal=pred, topk=topk, n_drop=n_drop, hold_thresh=1)
    ex = SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True)
    pmd, _ = backtest(start_time='2020-10-01', end_time='2026-05-15', strategy=strat, executor=ex,
                      benchmark='SH000905', account=1e8, exchange_kwargs=EXCHANGE)
    rep = pmd['1day'][0]; rep.index = pd.to_datetime(rep.index)
    ret, bench, cost = rep['return'], rep['bench'], rep['cost']
    net = ret - cost; ex_net = net - bench
    return dict(net_ann=ann(net), excess_ann=ann(ex_net), excess_ir=ann(ex_net)/vol(ex_net),
                excess_vol=vol(ex_net), net_sharpe=ann(net)/vol(net),
                net_mdd=mdd(net), excess_mdd=mdd(ex_net),
                turnover=rep['turnover'].mean()*100, cost_ann=ann(cost))


if __name__ == '__main__':
    qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
    pred = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
    print(f"已加载 R8.5 预测值, {len(pred)} 行", flush=True)
    print("\n=== topk × n_drop 网格扫描 (固定 hold=1, 基准 SH000905) ===\n", flush=True)

    topks = [30, 50, 60, 80, 100, 120, 150, 200]
    n_drops = [1, 2]

    rows = []
    print(f"{'topk':>6} {'n_drop':>7}  {'净超额':>8} {'IR':>7} {'净夏普':>8} {'换手':>7} {'成本':>7} {'超额回撤':>9}", flush=True)
    print("-" * 70, flush=True)
    for n_drop in n_drops:
        for topk in topks:
            m = run_bt(pred, topk, n_drop)
            m['topk'] = topk; m['n_drop'] = n_drop
            rows.append(m)
            mark = " ★" if (topk == 30 and n_drop == 1) else ""
            print(f"{topk:>6} {n_drop:>7}  {m['excess_ann']:>+7.2f}% {m['excess_ir']:>+7.3f} "
                  f"{m['net_sharpe']:>8.3f} {m['turnover']:>6.1f}% {m['cost_ann']:>6.2f}% "
                  f"{m['excess_mdd']:>+8.1f}%{mark}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'topk_sweep.csv', index=False)

    # 找最优
    best_ir = df.loc[df['excess_ir'].idxmax()]
    best_excess = df.loc[df['excess_ann'].idxmax()]
    best_sharpe = df.loc[df['net_sharpe'].idxmax()]
    print(f"\n=== 最优配置 ===", flush=True)
    print(f"按 IR:     topk={int(best_ir['topk'])}, n_drop={int(best_ir['n_drop'])}, IR={best_ir['excess_ir']:.3f}, 净超额={best_ir['excess_ann']:+.2f}%", flush=True)
    print(f"按 净超额: topk={int(best_excess['topk'])}, n_drop={int(best_excess['n_drop'])}, IR={best_excess['excess_ir']:.3f}, 净超额={best_excess['excess_ann']:+.2f}%", flush=True)
    print(f"按 净夏普: topk={int(best_sharpe['topk'])}, n_drop={int(best_sharpe['n_drop'])}, 净夏普={best_sharpe['net_sharpe']:.3f}, IR={best_sharpe['excess_ir']:.3f}", flush=True)

    print(f"\n=== 对比 R8.5 基线 (topk=30, n_drop=1) ===", flush=True)
    base = df[(df['topk']==30)&(df['n_drop']==1)].iloc[0]
    print(f"  基线:   净超额 {base['excess_ann']:+.2f}%, IR {base['excess_ir']:.3f}, 净夏普 {base['net_sharpe']:.3f}", flush=True)
    print(f"  最佳IR: 净超额 {best_ir['excess_ann']:+.2f}%, IR {best_ir['excess_ir']:.3f} (Δ {best_ir['excess_ir']-base['excess_ir']:+.3f})", flush=True)

    json.dump(dict(baseline=base.to_dict(), best_ir=best_ir.to_dict(),
                   best_excess=best_excess.to_dict(), best_sharpe=best_sharpe.to_dict(),
                   all_rows=df.to_dict('records')),
              open(OUT / 'sweep_result.json', 'w'), indent=1, default=str)
    print(f"\n[DONE] 结果存 results/runs/round11/", flush=True)
