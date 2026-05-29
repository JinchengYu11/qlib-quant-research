"""akshare 财务数据的 PIT 处理 + qlib bin 输出.

输入: data_raw/financials_ak/{code}.csv (akshare T2 格式)
     行 = 80 指标, 列 = ['选项','指标', 20260331, 20251231, ...] 日期降序

输出: 在 cn_data_v4 上, 给每只股票额外加 N 个财务字段的 bin 文件

PIT 规则 (法定披露截止日 = 最迟可用日):
  statDate 3/31 → pubDate 5/1
  statDate 6/30 → pubDate 9/1
  statDate 9/30 → pubDate 11/1
  statDate 12/31 → pubDate 次年 5/1
"""
import struct
from pathlib import Path
import numpy as np
import pandas as pd


# 我们提取的 10 个核心指标 (akshare 指标名 → qlib 字段名)
INDICATOR_MAP = {
    "净资产收益率(ROE)":      "roe",       # 盈利能力
    "销售净利率":              "npm",       # 净利率
    "毛利率":                  "gpm",       # 毛利率
    "总资产报酬率(ROA)":       "roa",       # 资产回报
    "基本每股收益":            "eps",       # 每股盈利
    "资产负债率":              "debt_ratio", # 杠杆
    "期间费用率":              "exp_ratio",  # 费用控制
    "归母净利润":              "ni",         # 净利润 (绝对值, 算同比用)
    "营业总收入":              "rev",        # 营收 (算同比用)
    "经营现金流量净额":         "ocf",        # 经营现金流 (质量因子)
}
QLIB_FIELDS = list(INDICATOR_MAP.values())


def statdate_to_pubdate(stat_str):
    """statDate 20231231 格式 → pubDate datetime (法定披露截止日)."""
    s = str(stat_str)
    if len(s) != 8 or not s.isdigit():
        return None
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    if m == 3 and d == 31:    # Q1
        return pd.Timestamp(y, 5, 1)
    elif m == 6 and d == 30:  # Q2 中报
        return pd.Timestamp(y, 9, 1)
    elif m == 9 and d == 30:  # Q3
        return pd.Timestamp(y, 11, 1)
    elif m == 12 and d == 31: # Q4 年报
        return pd.Timestamp(y + 1, 5, 1)
    return None


def load_akshare_csv(csv_path):
    """读取 akshare T2 CSV, 返回排序好的 (pubDate, dict_of_field_values) 列表.

    输入格式 (akshare T2):
      选项,指标,20260331,20251231,...,19961231
      常用指标,归母净利润,1.79e10,5.00e10,...,6.29e8
      常用指标,营业总收入,...
      ...
    """
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty or "指标" not in df.columns:
        return []

    # 找日期列 (8 位数字)
    date_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    if not date_cols:
        return []

    # 找我们要的指标行
    needed_rows = {}  # indicator_name → row series (subset to date cols)
    seen = set()
    for _, row in df.iterrows():
        ind = row["指标"]
        if ind in INDICATOR_MAP and ind not in seen:
            needed_rows[INDICATOR_MAP[ind]] = row[date_cols]
            seen.add(ind)

    # 按 pubDate 组织: 每个 pubDate → {field: value}
    pit_records = {}  # pubDate → dict
    for date_col in date_cols:
        pub = statdate_to_pubdate(date_col)
        if pub is None:
            continue
        vals = {}
        for field in QLIB_FIELDS:
            if field in needed_rows:
                v = needed_rows[field].get(date_col)
                try:
                    vals[field] = float(v) if v not in ("", "nan", None) and pd.notna(v) else np.nan
                except (ValueError, TypeError):
                    vals[field] = np.nan
        pit_records[pub] = vals

    # 排序
    return sorted(pit_records.items(), key=lambda x: x[0])


def align_to_calendar(records, calendar):
    """records: [(pubDate, dict), ...] 排序好; calendar: list of date str.
    每日 = 截止该日 (含) 已"可用"的最新一份财报.
    """
    out = pd.DataFrame(np.nan, index=calendar, columns=QLIB_FIELDS)
    if not records:
        return out
    cal_dt = pd.to_datetime(calendar)
    rec_idx = 0
    cur = {f: np.nan for f in QLIB_FIELDS}
    n = len(records)
    for i, d in enumerate(cal_dt):
        while rec_idx < n and records[rec_idx][0] <= d:
            for k, v in records[rec_idx][1].items():
                if not pd.isna(v):
                    cur[k] = v
            rec_idx += 1
        for k, v in cur.items():
            out.iat[i, out.columns.get_loc(k)] = v
    return out


def code_b2q(c):
    return c.replace(".", "").upper()


def write_bin(filepath, start_idx, arr):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arr = arr.astype("<f4", copy=False)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<f", float(start_idx)))
        f.write(arr.tobytes())


if __name__ == "__main__":
    PROJ = Path("/Users/cedricyu/qlib量化研究")
    FIN_DIR = PROJ / "data_raw" / "financials_ak"
    QLIB_DIR = Path("/Users/cedricyu/.qlib/qlib_data/cn_data_v4")

    cal = (QLIB_DIR / "calendars" / "day.txt").read_text().strip().split("\n")
    fin_files = sorted(FIN_DIR.glob("*.csv"))
    print(f"读取 {len(fin_files)} 只股票的 akshare 财务 CSV", flush=True)
    print(f"目标: {QLIB_DIR}, 字段 {QLIB_FIELDS}", flush=True)

    ok = empty = skipped = 0
    for i, csv_p in enumerate(fin_files):
        bcode = csv_p.stem
        qcode = code_b2q(bcode).lower()
        out_dir = QLIB_DIR / "features" / qcode
        if not out_dir.exists():
            skipped += 1
            continue
        records = load_akshare_csv(csv_p)
        if not records:
            empty += 1
            continue
        close_bin = out_dir / "close.day.bin"
        if not close_bin.exists():
            skipped += 1
            continue
        with open(close_bin, "rb") as f:
            start_idx = int(struct.unpack("<f", f.read(4))[0])
            n_days = (close_bin.stat().st_size - 4) // 4
        sub_cal = cal[start_idx : start_idx + n_days]
        aligned = align_to_calendar(records, sub_cal)
        for field in QLIB_FIELDS:
            arr = aligned[field].to_numpy(dtype="float32")
            write_bin(out_dir / f"{field}.day.bin", start_idx, arr)
        ok += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(fin_files)}] ok={ok} empty={empty} skipped={skipped}", flush=True)
    print(f"\n[DONE] ok={ok} empty={empty} skipped(no qlib data)={skipped}", flush=True)
