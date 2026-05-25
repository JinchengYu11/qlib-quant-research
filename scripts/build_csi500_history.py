"""
拉 CSI500 历史季度快照、构建并集与 periods.json（仿 csi300 的处理）。
快速：只是 ~64 次成分股查询。
"""
import socket
socket.setdefaulttimeout(30)
import json
import time
import sys
from pathlib import Path
import pandas as pd
import baostock as bs

sys.path.insert(0, str(Path(__file__).parent))
from baostock_collector import META_DIR, RAW_DIR, compact_periods


def main():
    bs.login()
    # 季度末日期 2010-Q1 ~ 2025-Q4
    snaps = pd.date_range("2010-03-31", "2025-12-31", freq="QE").strftime("%Y-%m-%d").tolist()
    print(f"将查询 {len(snaps)} 个季度快照", flush=True)

    code_snapshots = {}
    valid_snaps = []
    for i, d in enumerate(snaps):
        rs = bs.query_zz500_stocks(date=d)
        if rs.error_code != "0":
            print(f"  [{i+1}/{len(snaps)}] {d}: query 失败 {rs.error_msg}", flush=True)
            continue
        cnt = 0
        codes_this = set()
        while rs.next():
            codes_this.add(rs.get_row_data()[1])
            cnt += 1
        if cnt == 0:
            print(f"  [{i+1}/{len(snaps)}] {d}: 0 只（baostock 可能无该日数据）", flush=True)
            continue
        for c in codes_this:
            code_snapshots.setdefault(c, []).append(d)
        valid_snaps.append(d)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(snaps)}] {d}: {cnt} 只，累计并集 {len(code_snapshots)}", flush=True)
        time.sleep(0.1)
    bs.logout()

    print(f"\n[UNION] CSI500 历史并集: {len(code_snapshots)} 只股票 (跨 {len(valid_snaps)} 个有效快照)", flush=True)

    # 转区间
    periods = {c: compact_periods(snaps, valid_snaps) for c, snaps in code_snapshots.items()}
    out = META_DIR / "csi500_periods.json"
    json.dump(periods, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[META] 历史持仓区间 → {out}", flush=True)

    # 识别尚未采集的股票
    have = {f.stem for f in RAW_DIR.glob("*.csv")}
    union = set(code_snapshots.keys())
    todo = sorted(union - have)
    print(f"\n[TODO] CSI500 并集 {len(union)} 只 - 已有 {len(union & have)} = 待采 {len(todo)} 只", flush=True)
    (META_DIR / "csi500_todo.txt").write_text("\n".join(todo))


if __name__ == "__main__":
    main()
