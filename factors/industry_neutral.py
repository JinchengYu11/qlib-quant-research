"""行业中性化 Processor。
对每个交易日，每个因子值减去"该股所属行业当日均值"，得到行业中性后的残差。
"""
import json
from pathlib import Path
import pandas as pd

from qlib.data.dataset.processor import Processor, get_group_columns


class IndustryNeutral(Processor):
    """横截面行业中性化：对每天每个因子，减去同行业内的横截面均值。"""

    def __init__(self, industry_map_path: str, fields_group: str = "feature"):
        self.industry_map_path = industry_map_path
        with open(industry_map_path) as f:
            self.industry_map = json.load(f)
        self.fields_group = fields_group

    def fit(self, df=None):
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 取 instrument level
        inst_idx = df.index.get_level_values("instrument")
        # 对应行业（未知归为 UNKNOWN）
        industries = pd.Index(
            [self.industry_map.get(c, "UNKNOWN") for c in inst_idx],
            name="industry",
        )

        cols = get_group_columns(df, self.fields_group)
        if len(cols) == 0:
            return df

        subdf = df[cols]
        # 按 (datetime, industry) 分组，算 transform 均值
        grouper = [df.index.get_level_values("datetime"), industries]
        means = subdf.groupby(grouper).transform("mean")
        df[cols] = subdf - means
        return df

    def __repr__(self):
        return f"IndustryNeutral(map={Path(self.industry_map_path).name}, group={self.fields_group})"
