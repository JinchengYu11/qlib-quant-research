"""R13 财务数据采集 — 东方财富版.

数据源: akshare ak.stock_financial_abstract(symbol='600000')
  返回 80 指标 × ~110 季度的宽表, 日期回溯到 1996.

策略: 单线程 + 1.5s/股 节流, 避免触发东财风控.
  - 1838 股 × 1.5s ≈ 46 分钟 (含 API 调用 ~0.75s + sleep ~0.75s)
  - 支持断点续传 (已下载文件 skip)
  - 失败重试 3 次 + 长 backoff

输出: data_raw/financials_em/{ts_code}.csv
  - 格式: 选项, 指标, 20260331, 20251231, ..., 19961231 (宽表)
  - 跟 stock_financial_analysis_indicator (新浪) 同格式
  - 可直接用 factors/financial_pit_ak.py 处理

注意:
  - akshare 在不同版本可能改 symbol 格式. 当前要求纯 6 位数字 (无 SH/SZ 前缀).
  - 与之前 stock_financial_analysis_indicator (新浪源, 现已废) 不同, 东财源稳定.
"""
import sys, time, random
from pathlib import Path
import akshare as ak

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'data_raw' / 'financials_em'
OUT.mkdir(parents=True, exist_ok=True)

CODES_FILE = ROOT / 'data_raw' / 'meta' / 'wind_codes_for_query.csv'

SLEEP_MIN = 0.8
SLEEP_MAX = 1.6
RETRY_MAX = 3
RETRY_BACKOFF_S = [30, 60, 120]  # 失败递增等待


def code_to_6digit(ts_code):
    """600000.SH → 600000, 000001.SZ → 000001"""
    return ts_code.split('.')[0]


def load_codes():
    raw = CODES_FILE.read_text().strip()
    return raw.split(',')


def collect_one(symbol):
    """单股采集, 抛异常表示失败 (调用方决定 retry)."""
    df = ak.stock_financial_abstract(symbol=symbol)
    if df is None or len(df) == 0:
        raise RuntimeError("empty response")
    return df


def main():
    codes = load_codes()
    print(f"[Plan] 共 {len(codes)} 只股票, 预计 ~{len(codes) * 1.2 / 60:.0f} 分钟", flush=True)

    # 断点续传: 已存在的 skip
    todo = []
    for c in codes:
        sym = code_to_6digit(c)
        if not (OUT / f"{c}.csv").exists():
            todo.append((c, sym))
    skipped = len(codes) - len(todo)
    print(f"[Plan] 已完成 {skipped} 只, 待采 {len(todo)} 只", flush=True)

    success = 0
    failed_codes = []
    t_start = time.time()
    for i, (ts_code, sym) in enumerate(todo):
        for retry in range(RETRY_MAX):
            try:
                df = collect_one(sym)
                df.to_csv(OUT / f"{ts_code}.csv", index=False, encoding='utf-8-sig')
                success += 1
                break
            except Exception as e:
                msg = str(e)[:80]
                if retry < RETRY_MAX - 1:
                    wait = RETRY_BACKOFF_S[retry]
                    print(f"[{i+1}/{len(todo)}] {ts_code} 失败 retry {retry+1}: {msg} -> wait {wait}s", flush=True)
                    time.sleep(wait)
                else:
                    print(f"[{i+1}/{len(todo)}] {ts_code} GIVE UP after {RETRY_MAX} retries: {msg}", flush=True)
                    failed_codes.append(ts_code)

        # 限速
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # 进度
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed * 60
            remaining = (len(todo) - i - 1) / max(rate / 60, 0.01) / 60
            print(f"  [{i+1}/{len(todo)}] success={success} fail={len(failed_codes)} | "
                  f"{rate:.1f} 股/min, 剩余 ~{remaining:.0f}min", flush=True)

    # 落盘 failed list
    if failed_codes:
        (OUT / 'failed_codes.txt').write_text('\n'.join(failed_codes))
        print(f"\n失败 {len(failed_codes)} 只, 见 {OUT / 'failed_codes.txt'}", flush=True)

    total_files = len(list(OUT.glob('*.csv')))
    print(f"\n[Done] 已落盘 {total_files} 个 CSV, 失败 {len(failed_codes)} 个", flush=True)


if __name__ == '__main__':
    main()
