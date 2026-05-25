"""检查 Ridge 和 LightGBM 预测相关性 + 等权集成 IC。"""
import warnings; warnings.filterwarnings("ignore")
import sys, pandas as pd, numpy as np
sys.path.insert(0, '/Users/cedricyu/qlib量化研究')

ridge = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_Ridge_pred.pkl')
lgb = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')
df = pd.concat([ridge.rename('ridge'), lgb.rename('lgb')], axis=1).dropna()
print(f'对齐样本数: {len(df)}')
print(f'横截面 Pearson 相关: {df.corr().iloc[0,1]:.4f}')
ric = df.groupby(level=0).apply(lambda g: g['ridge'].corr(g['lgb'], method='spearman'))
print(f'日内 Rank 相关 (avg): {ric.mean():.4f}  (中位 {ric.median():.4f})')

# 等权融合（先按日横截面 z-score 归一化）
def daily_z(s):
    return s.groupby(level=0).apply(lambda g: (g - g.mean()) / (g.std() if g.std() > 0 else 1))

ridge_z = daily_z(df['ridge']).droplevel(0)
lgb_z = daily_z(df['lgb']).droplevel(0)
ens = (ridge_z + lgb_z) / 2

# 取 label
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
qlib.init(provider_uri='/Users/cedricyu/.qlib/qlib_data/cn_data_v4', region='cn')
dh = dict(start_time='2010-01-04', end_time='2026-05-19',
          fit_start_time='2010-01-04', fit_end_time='2017-12-31', instruments='csi500',
          infer_processors=[{'class':'RobustZScoreNorm','kwargs':{'fields_group':'feature','clip_outlier':True}},
                            {'class':'Fillna','kwargs':{'fields_group':'feature'}}],
          learn_processors=[{'class':'DropnaLabel'},
                            {'class':'CSRankNorm','kwargs':{'fields_group':'label'}}],
          label=['Ref($close, -2) / Ref($close, -1) - 1'])
if __name__ == '__main__':
    ds = DatasetH(handler=Alpha158(**dh), segments={
        'train':('2010-01-04','2017-12-31'),'valid':('2018-01-01','2019-12-31'),
        'test':('2020-10-01','2026-05-15')})
    te_label = ds.prepare('test', col_set='label', data_key=DataHandlerLP.DK_L).iloc[:,0]

    def eval_ic(pred, label):
        d = pd.concat([pred.rename('p'), label.rename('y')], axis=1).dropna()
        ic = d.groupby(level=0).apply(lambda g: g['p'].corr(g['y']))
        ric = d.groupby(level=0).apply(lambda g: g['p'].corr(g['y'], method='spearman'))
        return ic.mean(), ric.mean(), ic.mean()/ic.std(), ric.mean()/ric.std()

    print('\n=== 各模型预测信号质量 ===')
    for name, pred in [('LightGBM', lgb), ('Ridge', ridge), ('Equal-weight 集成', ens)]:
        ic, ric, icir, ricir = eval_ic(pred, te_label)
        print(f'  {name:20s} IC={ic:.4f}  RankIC={ric:.4f}  ICIR={icir:.3f}  RankICIR={ricir:.3f}')

    # 存等权集成预测供后续回测
    ens.to_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_EQens_pred.pkl')
    print('\n[DONE] 集成预测已存')
