"""单线程兜底补拉：读取 failed_codes.txt，逐只重试，慢但稳。"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baostock_collector import RAW_DIR, META_DIR, fetch_stock, save_csv
import baostock as bs

failed_file = META_DIR / "failed_codes.txt"
codes = [line.strip() for line in failed_file.read_text().splitlines() if line.strip()]
print(f"待补 {len(codes)} 只", flush=True)

rs = bs.login()
print(f"login: {rs.error_msg}", flush=True)

ok, fail = 0, []
t0 = time.time()
for i, code in enumerate(codes):
    csv = RAW_DIR / f"{code}.csv"
    if csv.exists() and csv.stat().st_size > 1000:
        ok += 1
        continue
    df = fetch_stock(code, "2010-01-01", "2026-05-19", retries=5)
    if df.empty:
        fail.append(code)
    else:
        save_csv(df, csv)
        ok += 1
    if (i + 1) % 20 == 0 or (i + 1) == len(codes):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(codes) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(codes)}] ok={ok} fail={len(fail)} | {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
    time.sleep(1.0)  # 单线程慢节奏

bs.logout()
print(f"\n[DONE] 成功 {ok}，失败 {len(fail)} / 总 {len(codes)}", flush=True)
if fail:
    out = META_DIR / "still_failed.txt"
    out.write_text("\n".join(fail))
    print(f"剩余失败 -> {out}", flush=True)
