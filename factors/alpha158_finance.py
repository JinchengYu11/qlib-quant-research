"""Alpha158Finance: Alpha158 (158 量价) + 10 个核心财务因子 (akshare 版).

财务字段 (来自 akshare 季报, 已 PIT 处理用法定披露日, 见 financial_pit_ak.py):
  $roe        净资产收益率 (盈利)
  $roa        总资产报酬率 (盈利)
  $npm        销售净利率 (利润率)
  $gpm        毛利率 (利润率)
  $eps        基本每股收益 (规模/盈利)
  $debt_ratio 资产负债率 (杠杆)
  $exp_ratio  期间费用率 (成本控制)
  $ni         归母净利润 (绝对盈利)
  $rev        营业总收入 (规模)
  $ocf        经营现金流量净额 (质量)

设计哲学: 保持简单, 不做高度衍生; 让 LGBM 自己组合.
"""
from qlib.contrib.data.handler import Alpha158


FINANCE_FIELDS = [
    "$roe", "$roa", "$npm", "$gpm", "$eps",
    "$debt_ratio", "$exp_ratio", "$ni", "$rev", "$ocf",
]
FINANCE_NAMES = [
    "ROE", "ROA", "NPM", "GPM", "EPS",
    "DEBT_RATIO", "EXP_RATIO", "NI", "REV", "OCF",
]


class Alpha158Finance(Alpha158):
    """Alpha158 + 10 财务因子."""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return fields + FINANCE_FIELDS, names + FINANCE_NAMES
