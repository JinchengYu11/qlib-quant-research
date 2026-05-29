"""akshare 财务数据 - 单线程 + 智能 backoff (避免 JSONDecodeError 反爬).

策略:
  - 单线程 (1 connection)
  - 每股 1.5s 间隔
  - JSONDecodeError → 等 30-120s 指数 backoff, 重试 3 次
  - 失败的写入 retry 列表, 下次跑
"""
import socket; socket.setdefaulttimeout(30)
import sys, time, json
from pathlib import Path
import pandas as pd
import akshare as ak

PROJ = Path("/Users/cedricyu/qlib量化研究")
CSV_DIR = PROJ / "data_raw" / "baostock_csv"
OUT_DIR = PROJ / "data_raw" / "financials_ak"
META_DIR = PROJ / "data_raw" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL = 1.2  # 每次调用间隔 (秒)
RETRY_BACKOFF = [10, 30, 90]  # 失败时 backoff (秒)


def fetch_one(code_short, retries=3):
    for att in range(retries + 1):
        try:
            df = ak.stock_financial_abstract(symbol=code_short)
            if df is None or df.empty:
                return None
            return df
        except json.JSONDecodeError:
            if att < len(RETRY_BACKOFF):
                wait = RETRY_BACKOFF[att]
                print(f"    JSONDecodeError, backoff {wait}s ...", flush=True)
                time.sleep(wait)
            else:
                return "JSONDecodeError"
        except Exception as e:
            if att < len(RETRY_BACKOFF):
                wait = RETRY_BACKOFF[att]
                time.sleep(wait)
            else:
                return f"{type(e).__name__}: {str(e)[:60]}"
    return "all retries failed"


def main():
    codes = sorted(f.stem for f in CSV_DIR.glob("*.csv")
                   if f.stem.startswith(("sh.", "sz.")) and not f.stem.startswith("sh.000"))
    # 排除已有
    todo = [c for c in codes if not (OUT_DIR / f"{c}.csv").exists()
            or (OUT_DIR / f"{c}.csv").stat().st_size <= 5000]
    print(f"总 {len(codes)}, 已有 {len(codes)-len(todo)}, 待采 {len(todo)}, 单线程", flush=True)

    ok = 0; fail = []
    t0 = time.time()
    for i, code in enumerate(todo):
        code_short = code.split(".")[1]
        result = fetch_one(code_short)
        if isinstance(result, pd.DataFrame):
            result.to_csv(OUT_DIR / f"{code}.csv", index=False)
            ok += 1
        else:
            fail.append((code, result or "empty"))
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(todo)}] ok={ok} fail={len(fail)} | "
                  f"{rate:.2f}/s | ETA {eta:.1f} min", flush=True)
        time.sleep(INTERVAL)

    print(f"\n[DONE] ok={ok} fail={len(fail)} / 待采 {len(todo)}", flush=True)
    if fail:
        (META_DIR / "financials_ak_failed.txt").write_text(
            "\n".join(f"{c}\t{r}" for c, r in fail))


if __name__ == "__main__":
    main()
