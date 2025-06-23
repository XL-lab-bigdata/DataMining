import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.导入数据集
dataset = pd.read_csv('../data/Mall_Customers.csv')
X = np.array(dataset.iloc[:, [3, 4]])
# 3.使用肘部法则确定聚类数量
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
# 4.绘制肘部法则图
plt.figure(figsize=(10, 8))
plt.rc('font', size=30)
plt.plot(range(1, 11), wcss, marker="o",linewidth=2,c="dodgerblue")
plt.plot(5,44448.45544793371,c="r",marker="X",markersize=20)
plt.xlabel('簇数量',fontsize=30)
plt.ylabel('距离平方和',fontsize=30)
plt.xticks(fontsize = 30)
plt.ticklabel_format(style="sci",scilimits=(-1,2),axis="y")
plt.yticks(fontsize = 30)
plt.show()
# 5.K-Means聚类
kmeans = KMeans(n_clusters=5, init='k-means++',
                max_iter=300, random_state=0)
Y_Kmeans = kmeans.fit_predict(X)
# 6.创建用于绘制边界的网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                     np.linspace(y_min, y_max, 1000))
# 7.预测聚类标签
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# 8.绘制聚类边界和结果
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.1,cmap='Set2')
colors = ['grey', 'salmon', 'peru', 'palegreen', 'steelblue']
sns.scatterplot(x=X[:, 0], y = X[:, 1],
                hue=Y_Kmeans,palette="Set2")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], s=100,
            c='tab:red', label='质心')
plt.xlabel('年收入',fontsize=30)
plt.ylabel('消费评分',fontsize=30)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize=20)
plt.show()