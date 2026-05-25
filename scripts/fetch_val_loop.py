"""循环兜底补齐估值数据：单连接、慢节奏、多轮重试。
每轮只处理仍缺失的股票，轮间冷却，让 baostock 限流窗口自然滑过。
关键：设 socket 全局超时，避免 baostock 网络调用无限卡死。"""
import socket
socket.setdefaulttimeout(30)  # 任何网络调用 30s 无响应即抛异常，由重试逻辑接管
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from valuation_collector import VAL_DIR, CSV_DIR, fetch_val
import baostock as bs

VAL_DIR.mkdir(parents=True, exist_ok=True)
all_codes = sorted(f.stem for f in CSV_DIR.glob("*.csv") if f.stem != "sh.000300")

MAX_PASSES = 8
SLEEP = 3.0  # 单连接慢节奏

for p in range(1, MAX_PASSES + 1):
    todo = [c for c in all_codes
            if not (VAL_DIR / f"{c}.csv").exists() or (VAL_DIR / f"{c}.csv").stat().st_size <= 500]
    print(f"\n=== 第 {p}/{MAX_PASSES} 轮：待补 {len(todo)} 只 ===", flush=True)
    if not todo:
        print("全部完成！", flush=True)
        break
    bs.login()
    ok, fail = 0, 0
    t0 = time.time()
    for i, code in enumerate(todo):
        df = fetch_val(code)
        if df.empty:
            fail += 1
        else:
            df.to_csv(VAL_DIR / f"{code}.csv", index=False)
            ok += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail} | {rate:.2f}/s", flush=True)
        time.sleep(SLEEP)
    bs.logout()
    print(f"  第 {p} 轮: ok={ok} fail={fail}", flush=True)
    if fail > 0 and p < MAX_PASSES:
        print(f"  冷却 30s 后重试残余 ...", flush=True)
        time.sleep(30)

remain = [c for c in all_codes
          if not (VAL_DIR / f"{c}.csv").exists() or (VAL_DIR / f"{c}.csv").stat().st_size <= 500]
total = len(all_codes)
print(f"\n[DONE] 估值覆盖 {total - len(remain)}/{total} ({(total-len(remain))/total*100:.1f}%)，仍缺 {len(remain)}", flush=True)
if remain:
    (Path(__file__).resolve().parent.parent / "data_raw" / "meta" / "val_still_missing.txt").write_text("\n".join(remain))
