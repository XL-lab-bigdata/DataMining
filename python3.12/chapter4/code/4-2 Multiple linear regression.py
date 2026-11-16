import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
# 1.读取数据集
data = pd.read_csv('../data/auto_mpg.csv')  # 读取数据集
data = data.dropna()  # 缺失值处理
corr_matrix = data.corr()  # 计算相关系数矩阵
# 2.可视化相关系数矩阵
plt.rcParams['font.sans-serif'] = ['STSong']  # 设置显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12, 10))
sns.set(font='STSong', font_scale=2)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('皮尔逊相关系数矩阵热力图', pad=20)
plt.show()
# 3.确定因变量和自变量
X = data[['displacement', 'horsepower', 'weight', 'model_year', 'origin']]
y = data.iloc[:, 7]
# 4.构建模型
X = sm.add_constant(X)  # 添加截距项
model = sm.OLS(y, X).fit()  # 拟合模型
print(model.summary())  # 输出模型摘要信息