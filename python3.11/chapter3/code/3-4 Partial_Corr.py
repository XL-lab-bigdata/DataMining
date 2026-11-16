from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import orth
# 1. 读取数据集并处理缺失值
data = pd.read_csv('../data/Annual Statistical Indicators Data.csv').dropna()
data.columns = ['地区', '年份'] + ['A'+str(i) for i in range(1, 12)]  # 重命名列名

# 2. 对指标中心化和标准化，计算每对指标的偏相关系数
indicators = data.iloc[:, 2:]
varying = indicators.nunique().gt(1)
scales = indicators.std(ddof=0)
standardized = (indicators - indicators.mean()) / scales.replace(0, 1)
standardized.loc[:, ~varying] = 0.0
partial_corr_matrix = pd.DataFrame(np.nan, index=indicators.columns, columns=indicators.columns)
for variable in varying.index[varying]:
    partial_corr_matrix.loc[variable, variable] = 1.0

for var1, var2 in combinations(indicators.columns, 2):
    controls = standardized.drop(columns=[var1, var2]).to_numpy()
    targets = standardized[[var1, var2]].to_numpy()
    basis = orth(controls)  # 用奇异值分解得到控制变量列空间的正交基
    residuals = targets - basis @ (basis.T @ targets)

    # 某个指标可由控制变量完全解释时，其残差相关系数未定义
    tolerance = np.finfo(float).eps * max(standardized.shape) * np.linalg.norm(targets, axis=0)
    if np.all(np.linalg.norm(residuals, axis=0) > tolerance):
        correlation = np.corrcoef(residuals, rowvar=False)[0, 1]
        partial_corr_matrix.loc[var1, var2] = correlation
        partial_corr_matrix.loc[var2, var1] = correlation

# 3. 绘制偏相关和 Pearson 相关系数矩阵的热力图
sns.set(font='STSong', font_scale=2.5)  # 设置字体和字体缩放
plt.figure(figsize=(24, 12))
plt.subplot(1, 2, 1)  # 子图1：偏相关系数矩阵热力图
sns.heatmap(partial_corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('偏相关系数矩阵热力图')
plt.subplot(1, 2, 2)  # 子图2：Pearson相关系数矩阵热力图
corr_matrix = indicators.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson相关系数矩阵热力图')
plt.tight_layout()
plt.show()
