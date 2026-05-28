"""采集 baostock 季度财务数据 (R13 用)。

3 个接口:
  - query_profit_data:  roeAvg / npMargin / gpMargin / epsTTM
  - query_growth_data:  YOYNI / YOYPNI / YOYEquity
  - query_dupont_data:  dupontROE / dupontAssetTurn

每只股票每季度 1 次调用 × 3 个接口 = 3 次, 共 1840 股 × 16 年 × 4 季 × 3 ≈ 35 万次.
用 4 worker 并行 + 容错重连. 输出: data_raw/financials/{code}.csv (合并 3 接口字段).
"""
import socket; socket.setdefaulttimeout(30)
import sys, time, json, os
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import baostock as bs

PROJ = Path("/Users/cedricyu/qlib量化研究")
CSV_DIR = PROJ / "data_raw" / "baostock_csv"
OUT_DIR = PROJ / "data_raw" / "financials"
META_DIR = PROJ / "data_raw" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 拉 2009-2026 年, 覆盖训练 2010-2017 + 验证 2018-2019 + 测试 2020-2026
YEARS = list(range(2009, 2027))
QUARTERS = [1, 2, 3, 4]


def _conn_bad(msg):
    s = str(msg).lower()
    return any(k in s for k in ["broken pipe", "接收数据异常", "connection", "reset"])


def fetch_one_quarter(code, year, quarter, retries=4):
    """拉单股单季度的 3 个接口, 返回合并字典."""
    result = {"code": code, "year": year, "quarter": quarter}
    apis = [
        ("profit", lambda: bs.query_profit_data(code=code, year=year, quarter=quarter)),
        ("growth", lambda: bs.query_growth_data(code=code, year=year, quarter=quarter)),
        ("dupont", lambda: bs.query_dupont_data(code=code, year=year, quarter=quarter)),
    ]
    for tag, fn in apis:
        for att in range(retries):
            try:
                rs = fn()
                if rs.error_code == "0":
                    if rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        for k, v in row.items():
                            if k in ("code",):
                                continue
                            result[f"{tag}_{k}"] = v
                    break
                if _conn_bad(rs.error_msg):
                    try: bs.logout()
                    except Exception: pass
                    time.sleep(1 + att); bs.login()
                else:
                    time.sleep(0.3 * (att + 1))
            except Exception as e:
                if _conn_bad(str(e)):
                    try: bs.logout()
                    except Exception: pass
                    time.sleep(1 + att); bs.login()
                else:
                    time.sleep(0.3 * (att + 1))
    return result


def worker_init():
    bs.login()


def worker(code):
    """拉一只股票全部年份季度的财务数据, 写一个 CSV."""
    out_csv = OUT_DIR / f"{code}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 1000:
        return (code, "skip")
    rows = []
    for y in YEARS:
        for q in QUARTERS:
            r = fetch_one_quarter(code, y, q)
            rows.append(r)
            time.sleep(0.03)  # 节流
    df = pd.DataFrame(rows)
    # 只保留有 pubDate 的行 (有真实财报数据的)
    if "profit_pubDate" in df.columns:
        keep = df["profit_pubDate"].astype(str).str.len() > 4
        df = df[keep]
    if df.empty:
        return (code, "empty")
    df.to_csv(out_csv, index=False)
    return (code, "ok")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    # 用历史并集股票列表 (包含 CSI300 + CSI500 历史成分)
    codes = sorted(f.stem for f in CSV_DIR.glob("*.csv") if f.stem.startswith(("sh.", "sz.")) and not f.stem.startswith("sh.000"))
    print(f"待采 {len(codes)} 只股票的财务数据, 每只 {len(YEARS)}×{len(QUARTERS)}={len(YEARS)*len(QUARTERS)} 个季度 × 3 接口", flush=True)
    print(f"总 API 调用量: {len(codes) * len(YEARS) * len(QUARTERS) * 3:,} 次", flush=True)
    print(f"使用 {workers} workers", flush=True)

    ok = skip = 0; fail = []
    t0 = time.time()
    with mp.Pool(workers, initializer=worker_init) as pool:
        for i, (code, st) in enumerate(pool.imap_unordered(worker, codes, chunksize=1)):
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else: fail.append((code, st))
            if (i + 1) % 25 == 0 or (i + 1) == len(codes):
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(codes) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(codes)}] ok={ok} skip={skip} fail={len(fail)} | "
                      f"{rate:.2f}/s | ETA {eta/60:.1f} min", flush=True)
    print(f"\n[DONE] ok={ok} skip={skip} fail={len(fail)}", flush=True)
    if fail:
        (META_DIR / "financials_failed.txt").write_text("\n".join(c for c, _ in fail))


if __name__ == "__main__":
    main()
