"""
补采 790 只股票的日频估值指标：peTTM / pbMRQ / psTTM / pcfNcfTTM。
复用 baostock_collector 的并行 + 重连逻辑。输出到 data_raw/baostock_valuation/。
"""
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import baostock as bs

PROJ = Path(__file__).resolve().parent.parent
VAL_DIR = PROJ / "data_raw" / "baostock_valuation"
CSV_DIR = PROJ / "data_raw" / "baostock_csv"
START, END = "2010-01-01", "2026-05-19"


def _conn_broken(msg):
    if not msg: return False
    s = str(msg).lower()
    return ("broken pipe" in s) or ("接收数据异常" in str(msg)) or ("connection" in s) or ("reset" in s)


def fetch_val(code, retries=6):
    fields = "date,code,peTTM,pbMRQ,psTTM,pcfNcfTTM"
    for attempt in range(retries):
        try:
            rs = bs.query_history_k_data_plus(code, fields, start_date=START, end_date=END,
                                              frequency="d", adjustflag="3")
            if rs.error_code == "0":
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                return pd.DataFrame(rows, columns=rs.fields)
            if _conn_broken(rs.error_msg):
                try: bs.logout()
                except Exception: pass
                time.sleep(1.0 + attempt); bs.login()
            else:
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if _conn_broken(str(e)):
                try: bs.logout()
                except Exception: pass
                time.sleep(1.0 + attempt); bs.login()
            else:
                time.sleep(0.5 * (attempt + 1))
    print(f"  [WARN] {code} 估值拉取失败", flush=True)
    return pd.DataFrame()


def worker_init():
    bs.login()


def worker(code):
    out = VAL_DIR / f"{code}.csv"
    if out.exists() and out.stat().st_size > 500:
        return (code, "skip")
    df = fetch_val(code)
    if df.empty:
        return (code, "fail")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return (code, "ok")


def main():
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    # 股票代码来自已采集的 OHLCV CSV（排除指数）
    codes = sorted(f.stem for f in CSV_DIR.glob("*.csv") if f.stem != "sh.000300")
    print(f"待采估值 {len(codes)} 只", flush=True)

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ok = skip = 0
    fail = []
    t0 = time.time()
    with mp.Pool(workers, initializer=worker_init) as pool:
        for i, (code, st) in enumerate(pool.imap_unordered(worker, codes, chunksize=2)):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail.append(code)
            if (i + 1) % 50 == 0 or (i + 1) == len(codes):
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(codes) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(codes)}] ok={ok} skip={skip} fail={len(fail)} "
                      f"| {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
    print(f"\n[DONE] ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        (PROJ / "data_raw" / "meta" / "val_failed.txt").write_text("\n".join(fail))
        print(f"失败 {len(fail)} 只 -> meta/val_failed.txt", flush=True)


if __name__ == "__main__":
    main()
