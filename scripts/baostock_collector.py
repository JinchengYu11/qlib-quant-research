"""
baostock A股数据采集器
- 拉取 2010-01-01 到指定结束日的日频数据
- 第一阶段（pipeline 验证）：拉历史 CSI300 成分股并集 + SH000300 指数
- 数据写到 RAW_DIR 下，按股票一个 CSV
- 复权方式：qfq (adjustflag=2, 前复权)，factor 列置 1（在 qlib 端使用）

用法：
    python scripts/baostock_collector.py --end 2026-05-19 --universe csi300
"""
import argparse
import os
import sys
import time
import json
import multiprocessing as mp
import pandas as pd
import baostock as bs
from datetime import date, timedelta
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent.parent / "data_raw" / "baostock_csv"
META_DIR = Path(__file__).resolve().parent.parent / "data_raw" / "meta"


def login_or_die():
    rs = bs.login()
    if rs.error_code != "0":
        print(f"[FATAL] baostock login 失败: {rs.error_msg}", file=sys.stderr)
        sys.exit(1)


def query_history_csi300(start: str, end: str) -> dict:
    """按季度采样 CSI300 成分股，返回 {code: [(start_in, end_in), ...]} 持仓区间映射。"""
    snapshots = []
    cur = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    # 季度末快照（3-31, 6-30, 9-30, 12-31）
    while cur <= end_ts:
        snap = cur + pd.offsets.QuarterEnd(0)
        if snap > end_ts:
            snap = end_ts
        snapshots.append(snap.strftime("%Y-%m-%d"))
        cur = (snap + pd.Timedelta(days=1))

    snapshots = sorted(set(snapshots))
    print(f"[CSI300] 将查询 {len(snapshots)} 个季度快照: {snapshots[0]} ~ {snapshots[-1]}")

    code_periods = {}  # code -> list of [start, end] in CSI300
    for i, snap_date in enumerate(snapshots):
        rs = bs.query_hs300_stocks(date=snap_date)
        if rs.error_code != "0":
            print(f"  [{i+1}/{len(snapshots)}] {snap_date} 失败: {rs.error_msg}")
            continue
        codes_this = set()
        while rs.next():
            row = rs.get_row_data()
            codes_this.add(row[1])  # code
        print(f"  [{i+1}/{len(snapshots)}] {snap_date}: {len(codes_this)} 只")
        for c in codes_this:
            if c not in code_periods:
                code_periods[c] = []
            # 把当前快照日期记为该股的"出现日"
            code_periods[c].append(snap_date)
        time.sleep(0.05)
    return code_periods


def compact_periods(snapshot_dates: list, all_snapshots: list) -> list:
    """把离散快照日期合并为连续区间 [(start, end), ...]。区间以快照覆盖到的下一快照日前一日为止。"""
    snapshot_dates = sorted(set(snapshot_dates))
    all_snapshots = sorted(set(all_snapshots))
    # 用 all_snapshots 做时间轴，确定每个 snapshot_dates 项之后到下一全局快照的覆盖
    runs = []
    cur_start = None
    cur_end = None
    snap_set = set(snapshot_dates)
    for i, d in enumerate(all_snapshots):
        # 该快照日的覆盖区间：[d, 下一快照日的前一日]
        if i + 1 < len(all_snapshots):
            next_d = pd.Timestamp(all_snapshots[i + 1]) - pd.Timedelta(days=1)
            next_d = next_d.strftime("%Y-%m-%d")
        else:
            next_d = d  # 末尾就到末尾
        if d in snap_set:
            if cur_start is None:
                cur_start = d
                cur_end = next_d
            else:
                cur_end = next_d
        else:
            if cur_start is not None:
                runs.append((cur_start, cur_end))
                cur_start = None
                cur_end = None
    if cur_start is not None:
        runs.append((cur_start, cur_end))
    return runs


def _is_conn_broken(err_msg: str) -> bool:
    """识别 baostock 连接级别异常。"""
    if not err_msg:
        return False
    s = str(err_msg).lower()
    return ("broken pipe" in s) or ("接收数据异常" in str(err_msg)) or ("connection" in s) or ("reset" in s)


def fetch_stock(code: str, start: str, end: str, retries: int = 5) -> pd.DataFrame:
    """拉一只股票的日频 OHLCV (qfq 前复权)。返回 DataFrame。连接异常会自动重登重试。"""
    fields = "date,code,open,high,low,close,volume,amount,tradestatus"
    for attempt in range(retries):
        try:
            rs = bs.query_history_k_data_plus(
                code, fields,
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
            if rs.error_code == "0":
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                return pd.DataFrame(rows, columns=rs.fields)
            # 连接级错误 → 重登
            if _is_conn_broken(rs.error_msg):
                try: bs.logout()
                except Exception: pass
                time.sleep(1.0 + attempt)
                bs.login()
            else:
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            # 任何 socket/io 异常 → 重登
            if _is_conn_broken(str(e)):
                try: bs.logout()
                except Exception: pass
                time.sleep(1.0 + attempt)
                bs.login()
            else:
                time.sleep(0.5 * (attempt + 1))
    print(f"  [WARN] {code} 拉取失败", flush=True)
    return pd.DataFrame()


# === 并行 worker ===
_worker_logged_in = False

def worker_init():
    """每个进程初始化时登录 baostock。"""
    global _worker_logged_in
    rs = bs.login()
    _worker_logged_in = (rs.error_code == "0")
    if not _worker_logged_in:
        print(f"[WORKER {os.getpid()}] login 失败: {rs.error_msg}", flush=True)


def worker_fetch(args):
    """worker 任务：拉一只股票，写 CSV。返回 (code, status, rows)。"""
    code, start, end, csv_path = args
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 1000:
        return (code, "skipped", 0)
    df = fetch_stock(code, start, end)
    if df.empty:
        return (code, "failed", 0)
    save_csv(df, csv_path)
    return (code, "ok", len(df))


def fetch_index(code: str, start: str, end: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="3",  # 指数不复权
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def save_csv(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--universe", default="csi300", choices=["csi300", "all"])
    p.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    login_or_die()
    try:
        if args.universe == "csi300":
            csi_periods_cache = META_DIR / "csi300_periods.json"
            if csi_periods_cache.exists():
                with open(csi_periods_cache) as f:
                    csi300_periods = json.load(f)
                codes_to_fetch = sorted(csi300_periods.keys())
                print(f"[CACHE] 已用缓存 csi300_periods.json: {len(codes_to_fetch)} 只股票", flush=True)
            else:
                # 历史 CSI300 成分股并集
                code_snapshots = query_history_csi300(args.start, args.end)
                print(f"\n[UNION] 历史 CSI300 涉及股票 {len(code_snapshots)} 只", flush=True)

                # 全部快照日期
                all_snaps = sorted({s for snaps in code_snapshots.values() for s in snaps})

                # 转为时间区间
                csi300_periods = {c: compact_periods(snaps, all_snaps) for c, snaps in code_snapshots.items()}
                with open(csi_periods_cache, "w") as f:
                    json.dump(csi300_periods, f, indent=2, ensure_ascii=False)
                print(f"[META] csi300 持仓区间已保存到 {csi_periods_cache}", flush=True)

                codes_to_fetch = sorted(code_snapshots.keys())
        else:
            # query_all_stock 拿当日全市场（仅当前股票池，不含历史退市股，第二阶段处理）
            raise NotImplementedError("--universe all 留到第二阶段")

        # 拉指数 SH000300
        print(f"\n[INDEX] 拉取 sh.000300 {args.start} ~ {args.end}")
        idx_df = fetch_index("sh.000300", args.start, args.end)
        if not idx_df.empty:
            save_csv(idx_df, RAW_DIR / "sh.000300.csv")
            print(f"  -> {len(idx_df)} 行")

        # 拉每只股票（并行）
        n = args.workers
        print(f"\n[STOCKS] 并行 {n} workers 拉取 {len(codes_to_fetch)} 只股票 ({args.start} ~ {args.end})", flush=True)
        tasks = [(code, args.start, args.end, str(RAW_DIR / f"{code}.csv")) for code in codes_to_fetch]
        # 主进程不需要 baostock 会话（worker 各自登录），先 logout
        bs.logout()

        failed = []
        ok_count = 0
        skipped = 0
        t0 = time.time()
        with mp.Pool(processes=n, initializer=worker_init) as pool:
            for idx, (code, status, rows) in enumerate(pool.imap_unordered(worker_fetch, tasks, chunksize=2)):
                if status == "ok":
                    ok_count += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append(code)
                if (idx + 1) % 20 == 0 or (idx + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - idx - 1) / rate if rate > 0 else 0
                    print(f"  [{idx+1}/{len(tasks)}] ok={ok_count} skip={skipped} fail={len(failed)} | {rate:.2f}/s | ETA {eta:.0f}s", flush=True)
        print(f"\n[DONE] 成功 {ok_count}，跳过 {skipped}，失败 {len(failed)} / 总 {len(tasks)}", flush=True)
        # 主进程重新登录以便 finally 的 logout 不报错
        bs.login()
        if failed:
            with open(META_DIR / "failed_codes.txt", "w") as f:
                f.write("\n".join(failed))
            print(f"  失败列表 -> {META_DIR / 'failed_codes.txt'}")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
