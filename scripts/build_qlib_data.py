"""
将 baostock CSV → qlib 二进制格式。

qlib 二进制格式（每日频率）：
  cn_data/
    calendars/day.txt           交易日列表，YYYY-MM-DD 每行
    instruments/all.txt          所有股票：code\tstart_date\tend_date
    instruments/csi300.txt       CSI300 持仓区间（可多段）
    features/<CODE>/<field>.day.bin
      bin 格式: 前 4 字节 float32 = 该股第一个交易日在 calendar 中的索引
                后 N*4 字节 float32 = 数据数组，按交易日顺序

约定:
  - 价格列已是 qfq 前复权 (baostock adjustflag=2)，所以 factor=1
  - vwap = (high+low+close)/3 (近似，因 baostock qfq 下 amount/volume 比例与 qfq 价格不同尺度)
  - tradestatus != "1" 的日子置 NaN（停牌/异常）

用法:
    python scripts/build_qlib_data.py --output ~/.qlib/qlib_data/cn_data_v2
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parent.parent
CSV_DIR = PROJ / "data_raw" / "baostock_csv"
VAL_DIR = PROJ / "data_raw" / "baostock_valuation"
META_DIR = PROJ / "data_raw" / "meta"

# 估值字段：raw 比率 → qlib 字段名
VAL_FIELDS = {"peTTM": "pettm", "pbMRQ": "pbmrq", "psTTM": "psttm", "pcfNcfTTM": "pcfttm"}

FIELDS_OUT = ["open", "high", "low", "close", "volume", "factor", "vwap",
              "pettm", "pbmrq", "psttm", "pcfttm"]


def code_b2q(code: str) -> str:
    """sh.600000 → SH600000"""
    return code.replace(".", "").upper()


def code_q2b(code: str) -> str:
    return code[:2].lower() + "." + code[2:]


def normalize_one_stock(df: pd.DataFrame, val_df: pd.DataFrame = None) -> pd.DataFrame:
    """处理一只股票的 baostock CSV，返回带 FIELDS_OUT 的 DataFrame，index 为 date 字符串。
    val_df: 可选的估值 CSV（date,code,peTTM,pbMRQ,psTTM,pcfNcfTTM）。"""
    df = df.copy()
    # 类型转换
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 停牌：tradestatus != 1
    if "tradestatus" in df.columns:
        df["tradestatus"] = df["tradestatus"].astype(str)
        halted = df["tradestatus"] != "1"
        df.loc[halted, ["open", "high", "low", "close"]] = np.nan
        # 停牌日 volume 通常为 0，保留
    # 派生字段
    df["factor"] = 1.0  # qfq 已调
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    # date 索引
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    # 合并估值字段
    for qname in VAL_FIELDS.values():
        df[qname] = np.nan
    if val_df is not None and not val_df.empty:
        v = val_df.copy()
        v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
        v = v.drop_duplicates("date").set_index("date")
        for raw, qname in VAL_FIELDS.items():
            if raw in v.columns:
                series = pd.to_numeric(v[raw], errors="coerce")
                # 估值比率为 0 视为缺失（避免 1/0）
                series = series.replace(0.0, np.nan)
                df[qname] = series.reindex(df.index)
    return df[FIELDS_OUT]


def write_bin(filepath: Path, start_idx: int, arr: np.ndarray):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arr = arr.astype("<f4", copy=False)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<f", float(start_idx)))
        f.write(arr.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="qlib 数据输出目录")
    args = p.parse_args()

    out = Path(args.output).expanduser()
    (out / "calendars").mkdir(parents=True, exist_ok=True)
    (out / "instruments").mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[FATAL] {CSV_DIR} 下没有 CSV，先跑 collector", file=sys.stderr)
        sys.exit(1)
    print(f"[STAGE] 读取 {len(csv_files)} 个 CSV")

    # 1) 先读所有 CSV，找出全部交易日（calendar）
    stocks = {}  # qlib_code -> DataFrame
    all_dates = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype={"code": str})
        except Exception as e:
            print(f"  [WARN] {f.name} 读取失败: {e}")
            continue
        if df.empty:
            continue
        # baostock code 在 csv 内
        bcode = df["code"].iloc[0]
        qcode = code_b2q(bcode)
        # 估值 CSV（同名）
        val_df = None
        val_path = VAL_DIR / f"{bcode}.csv"
        if val_path.exists():
            try:
                val_df = pd.read_csv(val_path, dtype={"code": str})
            except Exception:
                val_df = None
        norm = normalize_one_stock(df, val_df)
        if norm.empty:
            continue
        stocks[qcode] = norm
        all_dates.update(norm.index.tolist())

    # 指数 SH000300 的日期一定要进 calendar（它代表"市场交易日"）
    calendar = sorted(all_dates)
    print(f"[CALENDAR] {len(calendar)} 个交易日: {calendar[0]} ~ {calendar[-1]}")

    with open(out / "calendars" / "day.txt", "w") as f:
        f.write("\n".join(calendar) + "\n")

    cal_idx = {d: i for i, d in enumerate(calendar)}

    # 2) 每只股票写 bin
    print(f"[STAGE] 写 bin: {len(stocks)} 只股票 × {len(FIELDS_OUT)} 字段")
    inst_ranges = {}  # qcode -> (start_date, end_date) for instruments/all.txt
    for i, (qcode, df) in enumerate(sorted(stocks.items())):
        # 该股第一个/最后一个交易日在 calendar 中的索引
        first_d = df.index[0]
        last_d = df.index[-1]
        first_i = cal_idx[first_d]
        last_i = cal_idx[last_d]
        # 用 calendar 在 [first_i, last_i] 区间作为该股的轴
        sub_cal = calendar[first_i : last_i + 1]
        # 把 df 对齐到这个轴上
        df_aligned = df.reindex(sub_cal)
        for field in FIELDS_OUT:
            arr = df_aligned[field].to_numpy(dtype="float32")
            write_bin(out / "features" / qcode.lower() / f"{field}.day.bin", first_i, arr)
        inst_ranges[qcode] = (first_d, last_d)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(stocks)}]")

    # 3) instruments/all.txt
    with open(out / "instruments" / "all.txt", "w") as f:
        for qcode, (s, e) in sorted(inst_ranges.items()):
            f.write(f"{qcode}\t{s}\t{e}\n")
    print(f"[INSTRUMENTS] all.txt: {len(inst_ranges)} 行")

    # 4) instruments/csi300.txt 用历史持仓区间
    csi_periods_path = META_DIR / "csi300_periods.json"
    if csi_periods_path.exists():
        with open(csi_periods_path) as f:
            csi_periods = json.load(f)
        lines = []
        for bcode, periods in sorted(csi_periods.items()):
            qcode = code_b2q(bcode)
            if qcode not in inst_ranges:
                continue
            # 每个 (start, end) 区间一行；与该股的实际数据区间求交
            stock_s, stock_e = inst_ranges[qcode]
            for s, e in periods:
                s_clip = max(s, stock_s)
                e_clip = min(e, stock_e)
                if s_clip <= e_clip:
                    lines.append(f"{qcode}\t{s_clip}\t{e_clip}")
        with open(out / "instruments" / "csi300.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INSTRUMENTS] csi300.txt: {len(lines)} 行")
    else:
        print("[WARN] 未找到 csi300_periods.json，跳过 csi300.txt")

    # 5) instruments/csi500.txt
    #    优先用历史持仓区间 (csi500_periods.json) - 去幸存者偏差
    #    次选当前成分 (csi500_current.txt) - 含幸存者偏差，仅供快速验证
    csi500_periods_path = META_DIR / "csi500_periods.json"
    csi500_current_path = META_DIR / "csi500_current.txt"
    if csi500_periods_path.exists():
        with open(csi500_periods_path) as f:
            csi500_periods = json.load(f)
        lines = []
        for bcode, periods in sorted(csi500_periods.items()):
            qcode = code_b2q(bcode)
            if qcode not in inst_ranges:
                continue
            stock_s, stock_e = inst_ranges[qcode]
            for s, e in periods:
                s_clip = max(s, stock_s)
                e_clip = min(e, stock_e)
                if s_clip <= e_clip:
                    lines.append(f"{qcode}\t{s_clip}\t{e_clip}")
        with open(out / "instruments" / "csi500.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INSTRUMENTS] csi500.txt: {len(lines)} 行 (历史持仓区间，已去幸存者偏差)")
    elif csi500_current_path.exists():
        codes_500 = [c.strip() for c in csi500_current_path.read_text().splitlines() if c.strip()]
        lines = []
        for bcode in sorted(codes_500):
            qcode = code_b2q(bcode)
            if qcode in inst_ranges:
                s, e = inst_ranges[qcode]
                lines.append(f"{qcode}\t{s}\t{e}")
        with open(out / "instruments" / "csi500.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INSTRUMENTS] csi500.txt: {len(lines)} 行 (当前成分，⚠️有幸存者偏差)")
    else:
        print("[INFO] 未找到 csi500 元数据，跳过 csi500.txt")

    print(f"\n[DONE] qlib 数据已写到 {out}")


if __name__ == "__main__":
    main()
