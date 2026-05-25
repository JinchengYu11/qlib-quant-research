"""采集 CSI500 历史并集中尚未拿到的股票 OHLCV。复用并行 + 重连基础设施。"""
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
    todo = [c.strip() for c in (META_DIR / "csi500_todo.txt").read_text().splitlines() if c.strip()]
    print(f"待采 {len(todo)} 只 CSI500 新历史股", flush=True)
    tasks = [(c, str(RAW_DIR / f"{c}.csv")) for c in todo]

    ok = skip = 0
    fail = []
    t0 = time.time()
    with mp.Pool(workers, initializer=worker_init) as pool:
        for i, (code, st) in enumerate(pool.imap_unordered(worker, tasks, chunksize=2)):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail.append(code)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] ok={ok} skip={skip} fail={len(fail)} "
                      f"| {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
    print(f"\n[DONE] ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        (META_DIR / "csi500_union_failed.txt").write_text("\n".join(fail))
        print(f"失败列表 -> meta/csi500_union_failed.txt", flush=True)


if __name__ == "__main__":
    main()
