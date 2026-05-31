"""把东财财务 CSV (data_raw/financials_em/) 写入 qlib cn_data_v4 bins.

复用 factors/financial_pit_ak.py 的 helper (相同的宽表格式 + 相同的 10 指标 + PIT 规则).
区别仅在:
  - 输入目录: financials_em/ (vs financials_ak/)
  - 文件名格式: 600000.SH.csv (Wind/Tushare ts_code) vs sh.600000.csv (baostock)

PIT 规则 (跟 financial_pit_ak.py 一样, 用法定披露截止日):
  statDate 3/31  → 5/1   (Q1)
  statDate 6/30  → 9/1   (Q2 中报)
  statDate 9/30  → 11/1  (Q3)
  statDate 12/31 → 次年 5/1 (Q4 年报)
"""
import sys, struct
from pathlib import Path
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')
from factors.financial_pit_ak import (
    load_akshare_csv, align_to_calendar, write_bin, QLIB_FIELDS
)

PROJ = Path('/Users/cedricyu/qlib量化研究')
FIN_DIR = PROJ / 'data_raw' / 'financials_em'
QLIB_DIR = Path('/Users/cedricyu/.qlib/qlib_data/cn_data_v4')


def ts_code_to_qlib(ts_code):
    """600000.SH → sh600000, 000001.SZ → sz000001 (qlib bin 文件夹名)"""
    parts = ts_code.split('.')
    if len(parts) != 2:
        return None
    num, exch = parts
    return f"{exch.lower()}{num}"


def main():
    cal = (QLIB_DIR / 'calendars' / 'day.txt').read_text().strip().split('\n')
    fin_files = sorted(FIN_DIR.glob('*.csv'))
    # 过滤 failed_codes.txt 这种非 CSV 数据
    fin_files = [p for p in fin_files if '.' in p.stem and len(p.stem.split('.')) == 2]
    print(f"读取 {len(fin_files)} 只股票的东财财务 CSV", flush=True)
    print(f"目标: {QLIB_DIR}, 字段 {QLIB_FIELDS}", flush=True)

    ok = empty = skipped = 0
    for i, csv_p in enumerate(fin_files):
        ts_code = csv_p.stem  # "600000.SH"
        qcode = ts_code_to_qlib(ts_code)
        if qcode is None:
            skipped += 1
            continue
        out_dir = QLIB_DIR / 'features' / qcode
        if not out_dir.exists():
            skipped += 1
            continue

        records = load_akshare_csv(csv_p)
        if not records:
            empty += 1
            continue

        close_bin = out_dir / 'close.day.bin'
        if not close_bin.exists():
            skipped += 1
            continue
        with open(close_bin, 'rb') as f:
            start_idx = int(struct.unpack('<f', f.read(4))[0])
            n_days = (close_bin.stat().st_size - 4) // 4
        sub_cal = cal[start_idx: start_idx + n_days]
        aligned = align_to_calendar(records, sub_cal)
        for field in QLIB_FIELDS:
            arr = aligned[field].to_numpy(dtype='float32')
            write_bin(out_dir / f'{field}.day.bin', start_idx, arr)
        ok += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(fin_files)}] ok={ok} empty={empty} skipped={skipped}", flush=True)

    print(f"\n[DONE] ok={ok} empty={empty} skipped(no qlib data)={skipped}", flush=True)
    print(f"字段 bin 已写入 {QLIB_DIR}/features/<code>/{{{','.join(QLIB_FIELDS)}}}.day.bin", flush=True)


if __name__ == '__main__':
    main()
