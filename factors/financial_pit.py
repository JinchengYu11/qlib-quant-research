"""财务因子点位安全 (Point-in-Time) 处理 + qlib bin 输出.

核心思路:
  - 每只股票每个交易日, 取"该日已披露的最新一份财报"的字段
  - 避免用未来才知道的数据 (避免数据泄漏)

输入:  data_raw/financials/{code}.csv (每行一个季度的财务数据)
输出:  在原 qlib 数据集 (cn_data_v4) 上, 给每只股票额外加 N 个财务字段的 bin 文件
       字段日序列已对齐: 每日 = 截止该日已公布的最新财报值

字段命名 (qlib field 名小写):
  roe      = roeAvg (净资产收益率)
  npm      = npMargin (净利率)
  gpm      = gpMargin (毛利率, 银行业空)
  eps_ttm  = epsTTM (每股盈利 TTM)
  yoy_ni   = YOYNI (净利润同比)
  yoy_pni  = YOYPNI (归母净利同比)
  yoy_eq   = YOYEquity (股东权益同比)
  d_roe    = dupontROE (杜邦 ROE)
  d_turn   = dupontAssetTurn (资产周转率)
"""
import json, struct
from pathlib import Path
import numpy as np
import pandas as pd


FIELDS_MAP = {
    "profit_roeAvg": "roe",
    "profit_npMargin": "npm",
    "profit_gpMargin": "gpm",
    "profit_epsTTM": "eps_ttm",
    "growth_YOYNI": "yoy_ni",
    "growth_YOYPNI": "yoy_pni",
    "growth_YOYEquity": "yoy_eq",
    "dupont_dupontROE": "d_roe",
    "dupont_dupontAssetTurn": "d_turn",
}
QLIB_FIELDS = list(FIELDS_MAP.values())  # 输出字段


def code_b2q(code):  # sh.600000 → SH600000
    return code.replace(".", "").upper()


def load_financial_csv(csv_path):
    """读取单股财务 CSV, 返回 (pubDate, dict_of_fields) 排序列表."""
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty or "profit_pubDate" not in df.columns:
        return []
    # 关键: pubDate 决定"何时可用"
    df = df.dropna(subset=["profit_pubDate"]).copy()
    df = df[df["profit_pubDate"].str.len() > 4]
    # 按 pubDate 升序
    df = df.sort_values("profit_pubDate")
    records = []
    for _, row in df.iterrows():
        pub = row["profit_pubDate"]
        vals = {}
        for raw_col, q_col in FIELDS_MAP.items():
            if raw_col in df.columns:
                v = row.get(raw_col, "")
                if pd.notna(v) and str(v).strip() and str(v).strip() != "nan":
                    try:
                        vals[q_col] = float(v)
                    except ValueError:
                        vals[q_col] = np.nan
                else:
                    vals[q_col] = np.nan
        records.append((pub, vals))
    return records


def align_to_calendar(records, calendar):
    """把财务记录 (pubDate, vals) 按交易日对齐, 返回 DataFrame (index=calendar, columns=QLIB_FIELDS).
    每个交易日的值 = 截止该日 (含) 已公布的最新一份财报的值; 之前 = NaN.
    """
    out = pd.DataFrame(np.nan, index=calendar, columns=QLIB_FIELDS)
    if not records:
        return out
    cal_dates = pd.to_datetime(calendar)
    cur = {f: np.nan for f in QLIB_FIELDS}
    rec_idx = 0
    records_sorted = [(pd.to_datetime(p), v) for p, v in records]
    records_sorted.sort(key=lambda x: x[0])
    n_rec = len(records_sorted)
    for i, d in enumerate(cal_dates):
        # 把所有 pubDate <= d 的记录吃进当前状态
        while rec_idx < n_rec and records_sorted[rec_idx][0] <= d:
            for k, v in records_sorted[rec_idx][1].items():
                if not pd.isna(v):
                    cur[k] = v
            rec_idx += 1
        # 写入当前快照
        for k, v in cur.items():
            out.iat[i, out.columns.get_loc(k)] = v
    return out


def write_bin(filepath, start_idx, arr):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arr = arr.astype("<f4", copy=False)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<f", float(start_idx)))
        f.write(arr.tobytes())


if __name__ == "__main__":
    import sys
    PROJ = Path("/Users/cedricyu/qlib量化研究")
    FIN_DIR = PROJ / "data_raw" / "financials"
    QLIB_DIR = Path("/Users/cedricyu/.qlib/qlib_data/cn_data_v4")
    # 读 calendar
    cal = (QLIB_DIR / "calendars" / "day.txt").read_text().strip().split("\n")
    cal_dates_str = cal  # 字符串 ["2010-01-04", ...]
    cal_dt = pd.to_datetime(cal_dates_str)

    fin_files = sorted(FIN_DIR.glob("*.csv"))
    print(f"读取 {len(fin_files)} 只股票的财务 CSV", flush=True)
    print(f"目标 qlib 数据: {QLIB_DIR}, calendar 共 {len(cal)} 天", flush=True)

    for i, csv_p in enumerate(fin_files):
        bcode = csv_p.stem
        qcode = code_b2q(bcode).lower()
        out_dir = QLIB_DIR / "features" / qcode
        if not out_dir.exists():
            # 该股不在 qlib 数据集里, 跳过
            continue
        records = load_financial_csv(csv_p)
        if not records:
            continue
        # 找该股票的 bin 数据范围 (用 close.day.bin 的 start_idx)
        close_bin = out_dir / "close.day.bin"
        if not close_bin.exists():
            continue
        with open(close_bin, "rb") as f:
            start_idx = int(struct.unpack("<f", f.read(4))[0])
            n = (close_bin.stat().st_size - 4) // 4
        sub_cal = cal_dates_str[start_idx : start_idx + n]
        aligned = align_to_calendar(records, sub_cal)
        # 写每个字段的 bin
        for field in QLIB_FIELDS:
            arr = aligned[field].to_numpy(dtype="float32")
            write_bin(out_dir / f"{field}.day.bin", start_idx, arr)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(fin_files)}] 已写 {qcode.upper()}", flush=True)
    print("[DONE] 财务字段已写入 cn_data_v4", flush=True)
