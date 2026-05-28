"""Alpha158Finance: Alpha158 (158 量价) + 9 个核心财务因子.

财务字段 (来自 baostock 季报, 已做点位安全处理, 见 financial_pit.py):
  $roe      净资产收益率
  $npm      净利率
  $gpm      毛利率
  $yoy_ni   净利润同比
  $yoy_pni  归母净利同比
  $yoy_eq   股东权益同比
  $d_roe    杜邦 ROE
  $d_turn   资产周转率
  $eps_ttm  EPS TTM

设计哲学: 保持简单, 不做高度衍生 (避免过拟合, 让 LGBM 自己组合).
"""
from qlib.contrib.data.handler import Alpha158


FINANCE_FIELDS = [
    "$roe",
    "$npm",
    "$gpm",
    "$eps_ttm",
    "$yoy_ni",
    "$yoy_pni",
    "$yoy_eq",
    "$d_roe",
    "$d_turn",
]
FINANCE_NAMES = ["ROE", "NPM", "GPM", "EPS_TTM",
                 "YOY_NI", "YOY_PNI", "YOY_EQ",
                 "D_ROE", "D_TURN"]


class Alpha158Finance(Alpha158):
    """Alpha158 + 9 个财务因子."""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return fields + FINANCE_FIELDS, names + FINANCE_NAMES
