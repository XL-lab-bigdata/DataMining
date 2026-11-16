import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
# 1.读取数据集
data = pd.read_csv('../data/livestreaming.csv')
X = data['streamers_count']
y = data['viewers_count']
# 2.构建一元线性回归模型
X = sm.add_constant(X)  # 添加截距项
model = sm.OLS(y, X).fit()  # 拟合模型
print(model.summary())  # 输出模型摘要信息
print("\n一元线性回归拟合模型结果图：")
# 3.绘制一元线性回归拟合图
plt.rcParams['font.sans-serif']=['STSong']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(5, 3))
plt.plot(X['streamers_count'], y,'c.')  # X['streamers_count']：只画出X变量的值，不包括截距项
plt.plot(X['streamers_count'],model.predict(X),'salmon')
plt.xlabel('主播数量')
plt.ylabel('观众数量')
plt.show()