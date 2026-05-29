"""一次性采集 CSI500 期货 IC 主力连续合约日线 + basis 时间序列.

数据源: akshare futures_main_sina (sina 主力连续, 已处理换月)
输出:
  data_raw/futures/IC0_daily.csv  -- date, open, high, low, close, settle, volume
  data_raw/futures/basis_daily.csv -- date, ic_close, csi500_close, basis_abs, basis_pct

实测列名 (akshare 返回中文):
  日期 -> date
  开盘价 -> open
  最高价 -> high
  最低价 -> low
  收盘价 -> close
  成交量 -> volume
  持仓量 -> oi
  动态结算价 -> settle
"""
import sys
from pathlib import Path
import pandas as pd
import akshare as ak

ROOT = Path('/Users/cedricyu/qlib量化研究')
OUT = ROOT / 'data_raw' / 'futures'
OUT.mkdir(parents=True, exist_ok=True)

START = '2015-01-01'
END = '2026-05-30'


def fetch_ic_main():
    print(f"[1] 拉 IC 主力连续 ({START} ~ {END}) ...", flush=True)
    df = ak.futures_main_sina(symbol='IC0', start_date=START.replace('-', ''),
                               end_date=END.replace('-', ''))
    print(f"    原始列名: {list(df.columns)}", flush=True)

    # akshare 返回中文列名
    rename = {
        '日期': 'date',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '动态结算价': 'settle',
        '成交量': 'volume',
        '持仓量': 'oi',
    }
    # 也处理可能的英文列名 (以防 akshare 版本变化)
    rename_en = {
        'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'settle': 'settle', 'volume': 'volume', 'oi': 'oi',
    }
    df = df.rename(columns={**rename, **rename_en})

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"    拿到 {len(df)} 行, 时间范围 {df['date'].min().date()} ~ {df['date'].max().date()}", flush=True)

    # 只保留目标列 (settle 可能全为 0，仍保留方便后续判断)
    cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'settle', 'volume'] if c in df.columns]
    return df[cols]


def fetch_csi500_index():
    print(f"[2] 拉 SH000905 (CSI500 指数) 日线 ...", flush=True)
    df = ak.stock_zh_index_daily(symbol='sh000905')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"    拿到 {len(df)} 行", flush=True)
    return df[['date', 'close']].rename(columns={'close': 'csi500_close'})


def main():
    ic = fetch_ic_main()
    ic.to_csv(OUT / 'IC0_daily.csv', index=False)
    print(f"  ✓ 写出 {OUT / 'IC0_daily.csv'}", flush=True)

    idx = fetch_csi500_index()

    # 合并计算 basis
    basis = ic[['date', 'close']].rename(columns={'close': 'ic_close'}).merge(idx, on='date', how='inner')
    basis['basis_abs'] = basis['ic_close'] - basis['csi500_close']
    basis['basis_pct'] = basis['basis_abs'] / basis['csi500_close']
    basis.to_csv(OUT / 'basis_daily.csv', index=False)
    print(f"  ✓ 写出 {OUT / 'basis_daily.csv'}", flush=True)

    avg_basis_pct = basis['basis_pct'].mean() * 100
    print(f"\n[INFO] 平均 basis = {avg_basis_pct:+.3f}% (负值 = IC 贴水 = 多头持仓有收益)", flush=True)
    print(f"       Spec baseline 假设 5% 年化 basis drag, 实际数据校验后可在 round14 里 override", flush=True)


if __name__ == '__main__':
    main()
