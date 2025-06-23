import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
plt.rcParams['font.sans-serif'] = 'SimSun'  # 设置字体为宋体
plt.rcParams.update({'font.size': 24})

# 1.数据读取
data = pd.read_parquet('../data/creditcard.parquet')
data = data.sample(n=50000, random_state=20)
# 2.特征工程
data['Hour'] = data["Time"].apply(lambda x: divmod(x, 3600)[0])  # 将秒转为小时
X = data.drop(['Time', 'Class'], axis=1) # 将删除“Time”、“Class”列后的数据存储到变量X中
Y = data.Class # 将“Class”（异常标签）存储到变量Y中
# 3.LOF异常检测
LOF = LocalOutlierFactor(n_neighbors=25)  # 计算局部离群值因子
pred = LOF.fit_predict(X)  # 进行训练，得到的结果为-1或者1
Y_scores = LOF.negative_outlier_factor_  # 提取LOF分值
data['scores'] = Y_scores
print("检测出的异常值数量为:", np.sum(pred == -1))
# 4.绘制Top N 准确率曲线
accuracies = []
for n in range(1, 101):
    df_n = data.sort_values(by='scores', ascending=True).head(n)
    accuracy = df_n[df_n['Class'] == 1].shape[0] / n
    accuracies.append(accuracy)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), accuracies, color='#C69287', linestyle='-', linewidth=2, marker='o', markersize=5)
plt.xlabel('Top N',fontsize=30)
plt.ylabel('准确率',fontsize=30)
plt.title('Top N 准确率结果')
plt.xticks(np.arange(0, 101, step=10))
plt.yticks(np.arange(0, 0.5, step=0.1))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
