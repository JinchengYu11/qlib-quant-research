"""单线程循环兜底：补齐 CSI500 OHLCV 采集失败的股票。"""
import socket
socket.setdefaulttimeout(30)
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baostock_collector import RAW_DIR, META_DIR, fetch_stock, save_csv
import baostock as bs

START, END = "2010-01-01", "2026-05-19"
codes = [c.strip() for c in (META_DIR / "csi500_failed.txt").read_text().splitlines() if c.strip()]
print(f"待补 {len(codes)} 只", flush=True)

for p in range(1, 7):
    todo = [c for c in codes if not (RAW_DIR / f"{c}.csv").exists()
            or (RAW_DIR / f"{c}.csv").stat().st_size <= 1000]
    print(f"\n=== 第 {p} 轮：待补 {len(todo)} ===", flush=True)
    if not todo:
        print("全部完成", flush=True)
        break
    bs.login()
    ok, fail = 0, 0
    for i, code in enumerate(todo):
        df = fetch_stock(code, START, END)
        if df.empty:
            fail += 1
        else:
            save_csv(df, RAW_DIR / f"{code}.csv")
            ok += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(todo):
            print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail}", flush=True)
        time.sleep(2.5)
    bs.logout()
    if fail > 0 and p < 6:
        time.sleep(30)

remain = [c for c in codes if not (RAW_DIR / f"{c}.csv").exists()
          or (RAW_DIR / f"{c}.csv").stat().st_size <= 1000]
print(f"\n[DONE] CSI500 OHLCV 补齐完成，仍缺 {len(remain)}", flush=True)
