import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.导入数据集、预处理
df = pd.read_csv('../data/CC GENERAL.csv')
df = df.drop(['CUST_ID'], axis=1)
df = df.dropna()
X = pd.DataFrame(StandardScaler().fit_transform(df))
# 3.PCA降维
X = np.asarray(X)
pca = PCA(n_components=2, random_state=24)
X_pca = pca.fit_transform(X)
# 4.实施DBSCAN聚类
dbscan = DBSCAN(eps=2, min_samples=4)
y_dbscan = dbscan.fit_predict(X_pca)
# 5.实施K-means聚类
kmeans = KMeans(n_clusters=2, random_state=24)
y_kmeans = kmeans.fit_predict(X_pca)
# 6.DBSCAN算法结果可视化
plt.figure(figsize=(14, 9))
sns.scatterplot(x=X_pca[:, 0],y=X_pca[:, 1],hue=y_dbscan,palette="Set2",s=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize = 20)
plt.title("DBSCAN算法聚类结果", fontsize=40)
plt.show()
# 7.K-means算法可视化
plt.figure(figsize=(14, 9))
sns.scatterplot(x=X_pca[:, 0],y=X_pca[:, 1],hue=y_kmeans,palette="Set2",s=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize = 20)
plt.title("K-means算法聚类结果", fontsize=40)
plt.show()
# 8.计算轮廓系数
silhouette_dbscan = silhouette_score(X_pca, y_dbscan)
silhouette_kmeans = silhouette_score(X_pca, y_kmeans)
print(f'DBSCAN算法的轮廓系数: {silhouette_dbscan:.3f}')
print(f'K-means算法的轮廓系数: {silhouette_kmeans:.3f}')