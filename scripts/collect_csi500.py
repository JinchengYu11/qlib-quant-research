"""
采集当前 CSI500 成分股的 OHLCV（快速验证用，有幸存者偏差）。
复用 baostock_collector 的并行 + 重连逻辑，写入 data_raw/baostock_csv/。
当前成分列表存到 meta/csi500_current.txt。
"""
import socket
socket.setdefaulttimeout(30)
import sys
import time
import multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baostock_collector import RAW_DIR, META_DIR, fetch_stock, save_csv, worker_init
import baostock as bs

START, END = "2010-01-01", "2026-05-19"


def worker(args):
    code, csv_path = args
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 1000:
        return (code, "skip")
    df = fetch_stock(code, START, END)
    if df.empty:
        return (code, "fail")
    save_csv(df, csv_path)
    return (code, "ok")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    bs.login()
    rs = bs.query_zz500_stocks(date="2026-05-15")
    codes = []
    while rs.next():
        codes.append(rs.get_row_data()[1])
    bs.logout()
    print(f"CSI500 当前成分 {len(codes)} 只", flush=True)
    (META_DIR).mkdir(parents=True, exist_ok=True)
    (META_DIR / "csi500_current.txt").write_text("\n".join(sorted(codes)))

    tasks = [(c, str(RAW_DIR / f"{c}.csv")) for c in codes]
    ok = skip = 0
    fail = []
    t0 = time.time()
    with mp.Pool(workers, initializer=worker_init) as pool:
        for i, (code, st) in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail.append(code)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] ok={ok} skip={skip} fail={len(fail)} "
                      f"| {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
    print(f"\n[DONE] ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        (META_DIR / "csi500_failed.txt").write_text("\n".join(fail))


if __name__ == "__main__":
    main()
