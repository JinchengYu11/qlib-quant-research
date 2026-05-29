"""IC 期货日线加载器 (只管价格).

接口: load_ic_daily(start, end) -> pd.DataFrame
    index = trading_date (datetime)
    columns = [close]

basis drag 不在这里算 — 是 ICEngine 的事 (避免 double-count).
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/cedricyu/qlib量化研究')
IC_CSV = ROOT / 'data_raw' / 'futures' / 'IC0_daily.csv'


def load_ic_daily(start, end, ic_csv=None):
    """加载 IC 主力连续日线.

    Args:
        start, end: 字符串或 datetime, inclusive
        ic_csv: 可选, 覆盖默认 CSV 路径 (测试用)

    Returns:
        pd.DataFrame indexed by date with column [close]
        只返回 IC 实际交易日, 不补 weekend / holiday.
    """
    path = Path(ic_csv) if ic_csv else IC_CSV
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[['date', 'close']].sort_values('date').reset_index(drop=True)
    df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]
    df = df.set_index('date')
    return df


def _test_load_basic():
    """读 3 行小 CSV, 验证列和值."""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.write("2020-01-06,5030,5060,5010,5050,5045,1100\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert len(df) == 3, f"期望 3 行, 拿到 {len(df)}"
    assert df.index[0] == pd.Timestamp('2020-01-02'), f"首日错: {df.index[0]}"
    assert df['close'].iloc[0] == 5020
    assert df['close'].iloc[2] == 5050
    assert list(df.columns) == ['close'], f"应该只有 close 列, 拿到 {list(df.columns)}"
    print("  ✓ _test_load_basic")


def _test_date_filter():
    """start/end 过滤生效"""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2019-12-30,4900,4950,4880,4920,4915,1000\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert len(df) == 2, f"期望过滤掉 12-30, 拿到 {len(df)} 行"
    print("  ✓ _test_date_filter")


def _test_sorted():
    """无论 CSV 顺序, 输出按日期排序"""
    import tempfile, os
    csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
    csv.write("date,open,high,low,close,settle,volume\n")
    csv.write("2020-01-06,5030,5060,5010,5050,5045,1100\n")
    csv.write("2020-01-02,5000,5050,4980,5020,5015,1000\n")
    csv.write("2020-01-03,5020,5040,5000,5030,5025,1200\n")
    csv.close()
    df = load_ic_daily('2020-01-01', '2020-01-31', ic_csv=csv.name)
    os.unlink(csv.name)
    assert df.index.is_monotonic_increasing
    print("  ✓ _test_sorted")


if __name__ == '__main__':
    print("Running engine/ic_data.py tests ...")
    _test_load_basic()
    _test_date_filter()
    _test_sorted()
    print("All tests passed.")
