"""
Alpha158Plus：在 Alpha158（158 个量价因子）基础上，追加估值类基本面因子。

估值因子全部来自 baostock 日频指标，点位安全（按当日已披露财报算 TTM）：
  $pettm  市盈率TTM   $pbmrq 市净率   $psttm 市销率TTM   $pcfttm 市现率TTM

追加 10 个因子：
  - 4 个估值"收益率"水平：EP/BP/SP/CFP（= 1/估值比率，越高越便宜）
  - 3 个估值动量：近 3/12 个月估值变化（PE 下降 = 变便宜）
  - 3 个估值均值回归：当前估值 / 自身 250 日均值（<1 = 比历史便宜）

设计动机：Alpha158 纯量价，与"股票贵不贵"无关；估值因子与之天然正交，
是 A 股最经得起检验的异象之一，最可能带来增量 alpha。
"""
from qlib.contrib.data.handler import Alpha158


VALUE_FIELDS = [
    "1/$pettm",                       # EP   盈利收益率
    "1/$pbmrq",                       # BP   账面收益率
    "1/$psttm",                       # SP   销售收益率
    "1/$pcfttm",                      # CFP  现金流收益率
    "Ref($pettm, 60)/$pettm",         # EP_M3   近 3 月 PE 下降幅度
    "Ref($pbmrq, 60)/$pbmrq",         # BP_M3
    "Ref($pettm, 250)/$pettm",        # EP_M12  近 12 月 PE 下降幅度
    "$pettm/Mean($pettm, 250)",       # PE_REL  当前 PE / 自身 1 年均值
    "$pbmrq/Mean($pbmrq, 250)",       # PB_REL
    "$psttm/Mean($psttm, 250)",       # PS_REL
]
VALUE_NAMES = ["EP", "BP", "SP", "CFP", "EP_M3", "BP_M3", "EP_M12",
               "PE_REL", "PB_REL", "PS_REL"]


class Alpha158Plus(Alpha158):
    """Alpha158 + 10 个估值类基本面因子。"""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return fields + VALUE_FIELDS, names + VALUE_NAMES
