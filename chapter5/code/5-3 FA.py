import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
df = pd.read_csv("../data/character.csv", index_col=0).reset_index(drop=True)
#1.数据处理，去掉无效字段与空值
df.drop(["gender","education","age"],axis=1,inplace=True)
df.dropna(inplace=True)
#2.进行适用性检测
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(f'p值：{p_value}')
kmo_all,kmo_model=calculate_kmo(df)
print(f'KMO：{kmo_model}')
#3.调用因子分析算法拟合数据
fa = FactorAnalyzer(25,rotation='varimax')
fa.fit(df)
#4.求解特征值ev、特征向量v
ev,v=fa.get_eigenvalues()
#5.通过绘制碎石图确定因子个数
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(6, 5))
plt.plot(range(1, df.shape[1] + 1), ev, marker='o', color='#8389E0')
plt.xlabel("因子个数")
plt.ylabel("特征值")
plt.grid()
plt.tight_layout()
plt.show()
#6.选择6个因子构建模型，指定矩阵旋转方式为varimax，实现方差最大化
fa_six = fa = FactorAnalyzer(6,rotation='varimax')
fa_six.fit(df)
#7.求解因子载荷矩阵
fa_six.loadings_
#8.将原始数据用6个因子进行描述
pd.DataFrame(fa_six.loadings_,index=df.columns)
df1=pd.DataFrame(fa_six.transform(df))
df2=pd.DataFrame(fa.get_factor_variance(),index=['variance','proportional_variance','cumulative_variances'], columns=[f"factor{x}" for x in range(1,7)])
print(round(df1,3))
print(round(df2,3))