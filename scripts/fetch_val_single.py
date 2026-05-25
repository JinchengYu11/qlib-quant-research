"""单线程补齐估值数据：逐只拉取，慢但稳（baostock 单连接不触发限流）。"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from valuation_collector import VAL_DIR, CSV_DIR, fetch_val
import baostock as bs

VAL_DIR.mkdir(parents=True, exist_ok=True)
# 分片：python fetch_val_single.py <shard_idx> <shard_total>
shard_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
shard_total = int(sys.argv[2]) if len(sys.argv) > 2 else 1
codes = sorted(f.stem for f in CSV_DIR.glob("*.csv") if f.stem != "sh.000300")
todo = [c for c in codes if not (VAL_DIR / f"{c}.csv").exists()
        or (VAL_DIR / f"{c}.csv").stat().st_size <= 500]
todo = [c for i, c in enumerate(todo) if i % shard_total == shard_idx]
print(f"[shard {shard_idx}/{shard_total}] 待补 {len(todo)} 只", flush=True)

bs.login()
ok, fail = 0, []
t0 = time.time()
for i, code in enumerate(todo):
    df = fetch_val(code)
    if df.empty:
        fail.append(code)
    else:
        df.to_csv(VAL_DIR / f"{code}.csv", index=False)
        ok += 1
    if (i + 1) % 25 == 0 or (i + 1) == len(todo):
        el = time.time() - t0
        rate = (i + 1) / el if el > 0 else 0
        eta = (len(todo) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(todo)}] ok={ok} fail={len(fail)} | {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
    time.sleep(0.8)
bs.logout()
print(f"\n[DONE] ok={ok} fail={len(fail)} / 待补 {len(todo)}", flush=True)
if fail:
    print(f"仍失败: {fail}", flush=True)
