"""诊断：LGB top 30 与 Ridge top 30 的重合度。"""
import pandas as pd, numpy as np

ridge = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_Ridge_pred.pkl')
lgb = pd.read_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R85_CSI500debiased_FINAL_pred.pkl')

# 对齐索引
df = pd.concat([ridge.rename('ridge'), lgb.rename('lgb')], axis=1).dropna()

# 每天比较 top 30 重合度
def overlap(g, k=30):
    if len(g) < k: return np.nan
    top_r = set(g.nlargest(k, 'ridge').index)
    top_l = set(g.nlargest(k, 'lgb').index)
    return len(top_r & top_l) / k

overlaps = df.groupby(level=0).apply(overlap)
print(f'LGB top 30 与 Ridge top 30 平均重合: {overlaps.mean()*100:.1f}%')
print(f'中位重合: {overlaps.median()*100:.1f}%   最低: {overlaps.min()*100:.1f}%   最高: {overlaps.max()*100:.1f}%')

# 各位数比较
for k in [10, 30, 50, 100]:
    def ov(g, k=k):
        if len(g) < k: return np.nan
        return len(set(g.nlargest(k, 'ridge').index) & set(g.nlargest(k, 'lgb').index)) / k
    print(f'  top {k:3d} 平均重合: {df.groupby(level=0).apply(ov).mean()*100:.1f}%')

# 加权融合（按 IC 加权，Ridge 0.0366, LGB 0.0328）
w_r = 0.0366 / (0.0366 + 0.0328)
w_l = 0.0328 / (0.0366 + 0.0328)
print(f'\nIC 加权: Ridge {w_r:.3f}, LGB {w_l:.3f}')
def zd(s): return s.groupby(level=0).apply(lambda g: (g - g.mean())/(g.std() if g.std()>0 else 1)).droplevel(0)
ens = w_r * zd(df['ridge']) + w_l * zd(df['lgb'])
ens.to_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_ICw_pred.pkl')

# 70/30 偏 LGB（因为 LGB 单独 IR 更高 0.445 vs 0.313）
ens2 = 0.3 * zd(df['ridge']) + 0.7 * zd(df['lgb'])
ens2.to_pickle('/Users/cedricyu/qlib量化研究/results/figures_data/R9_LGB_Ridge_70_30_pred.pkl')
print('已存 IC 加权 + 70/30 偏 LGB 两个版本预测，待回测')
