"""用 akshare 重采财务数据 (替代 baostock).
单股 1 次 API call 拉全 80 指标 × 109 季度. 比 baostock 快 70 倍.

PIT (点位安全) 处理: akshare 没有 pubDate 字段, 用 A 股法定披露截止日:
  Q1 (3/31) → 5/1 起可用     (一季报法定 4/30 前披露)
  Q2 (6/30) → 9/1 起可用     (中报法定 8/31 前)
  Q3 (9/30) → 11/1 起可用    (三季报法定 10/31 前)
  Q4 (12/31) → 次年 5/1 起可用  (年报法定 4/30 前)

保守 PIT: 宁可错过 alpha, 绝不数据泄漏.
"""
import socket; socket.setdefaulttimeout(30)
import sys, time, multiprocessing as mp
from pathlib import Path
import pandas as pd
import akshare as ak

PROJ = Path("/Users/cedricyu/qlib量化研究")
CSV_DIR = PROJ / "data_raw" / "baostock_csv"
OUT_DIR = PROJ / "data_raw" / "financials_ak"
META_DIR = PROJ / "data_raw" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_one(code_with_prefix):
    """code_with_prefix: 如 'sh.600000', akshare 用 600000 这种 6 位."""
    code_short = code_with_prefix.split(".")[1]
    out_csv = OUT_DIR / f"{code_with_prefix}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 5000:
        return (code_with_prefix, "skip")
    try:
        df = ak.stock_financial_abstract(symbol=code_short)
        if df is None or df.empty:
            return (code_with_prefix, "empty")
        df.to_csv(out_csv, index=False)
        return (code_with_prefix, "ok")
    except Exception as e:
        return (code_with_prefix, f"fail:{type(e).__name__}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    codes = sorted(f.stem for f in CSV_DIR.glob("*.csv")
                   if f.stem.startswith(("sh.", "sz.")) and not f.stem.startswith("sh.000"))
    print(f"待采 {len(codes)} 只股票, 使用 {workers} workers (akshare T2)", flush=True)
    ok = skip = 0; fail = []
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, (code, st) in enumerate(pool.imap_unordered(fetch_one, codes, chunksize=2)):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail.append((code, st))
            if (i + 1) % 50 == 0 or (i + 1) == len(codes):
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(codes) - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  [{i+1}/{len(codes)}] ok={ok} skip={skip} fail={len(fail)} | "
                      f"{rate:.2f}/s | ETA {eta:.1f} min", flush=True)
    print(f"\n[DONE] ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        (META_DIR / "financials_ak_failed.txt").write_text(
            "\n".join(f"{c}\t{s}" for c, s in fail))


if __name__ == "__main__":
    main()
