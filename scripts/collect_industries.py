"""一次性拉所有 A 股的行业归属（baostock 证监会分类），保存到 meta/industries.json。"""
import json, sys
from pathlib import Path
import baostock as bs
import pandas as pd

META = Path('/Users/cedricyu/qlib量化研究/data_raw/meta')
META.mkdir(parents=True, exist_ok=True)

bs.login()
rs = bs.query_stock_industry()
data = []
while rs.next():
    data.append(rs.get_row_data())
df = pd.DataFrame(data, columns=rs.fields)
print(f'拉到 {len(df)} 只股票的行业信息')

# baostock code "sh.600000" → qlib code "SH600000"
def b2q(c): return c.replace('.', '').upper()

# 处理：industry 空的标 "UNKNOWN"，取前 3 字符（如 "C39"）作为细分大类
def industry_key(s):
    if not s or not s.strip():
        return 'UNKNOWN'
    return s[:3]  # 如 'C39', 'J66'

mapping = {b2q(row['code']): industry_key(row['industry']) for _, row in df.iterrows()}
counts = pd.Series(mapping.values()).value_counts()
print(f'\n细分大类数: {counts.shape[0]}')
print(f'每类股票数 (top 10):')
print(counts.head(10))
print(f'每类股票数 (bottom 10):')
print(counts.tail(10))

with open(META / 'industries.json', 'w') as f:
    json.dump(mapping, f, ensure_ascii=False)
print(f'\n[DONE] 已存到 {META}/industries.json')
bs.logout()
