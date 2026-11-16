import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import partial_corr
# 1. 读取数据集并处理缺失值
data = pd.read_csv('../data/Annual Statistical Indicators Data.csv').dropna()
data.columns = ['地区', '年份'] + ['A'+str(i) for i in range(1, 12)]  # 重命名列名
sns.set(font='STSong', font_scale=2.5)  # 设置字体和字体缩放
# 2. 计算偏相关系数并绘制热力图
plt.figure(figsize=(24, 12))
# 3.创建一个空的 DataFrame 来存储偏相关系数
partial_corr_matrix = pd.DataFrame(index=data.columns[2:], columns=data.columns[2:])
# 4.计算每对变量的偏相关系数
for var1 in data.columns[2:]:
    for var2 in data.columns[2:]:
        if var1 != var2:
            try:
                result = partial_corr(data=data, x=var1, y=var2, covar=[col for col in data.columns[2:] if col != var1 and col != var2])
                partial_corr_matrix.loc[var1, var2] = result.at['pearson', 'r']
            except Exception as e:
                partial_corr_matrix.loc[var1, var2] = None
        else:
            partial_corr_matrix.loc[var1, var2] = 1.0
partial_corr_matrix = partial_corr_matrix.astype(float)
plt.subplot(1, 2, 1)  # 子图1：偏相关系数矩阵热力图
sns.heatmap(partial_corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('偏相关系数矩阵热力图')
plt.subplot(1, 2, 2)  # 子图2：Pearson相关系数矩阵热力图
corr_matrix = data.iloc[:, 2:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson相关系数矩阵热力图')
plt.tight_layout()
plt.show()