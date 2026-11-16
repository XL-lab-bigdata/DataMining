import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering, KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_kernels
import seaborn as sns
# 1.设置中文显示
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.读取数据
SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',').T
nStock = len(SP500[:, 0])
# 3.标准化数据
X = (SP500 - np.mean(SP500, axis=1).reshape(-1, 1)) / np.std(SP500, axis=1).reshape(-1, 1)
# 4.使用t-SNE降维
Y = manifold.TSNE(n_components=2, random_state=0).fit_transform(X)
# 5.计算谱聚类和K-means的Davies-Bouldin指数
best_db_score_spectral = float('inf')
best_db_score_kmeans = float('inf')
best_n_clusters_spectral = 0
best_n_clusters_kmeans = 0
best_labels_spectral = None
best_labels_kmeans = None
for n_clusters in range(2, 11):
    # 谱聚类
    affinity = pairwise_kernels(Y, metric='rbf')
    labels_spectral = spectral_clustering(affinity=affinity,
                                          n_clusters=n_clusters)
    db_score_spectral = davies_bouldin_score(Y, labels_spectral)
    # K-means聚类
    labels_kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(Y)
    db_score_kmeans = davies_bouldin_score(Y, labels_kmeans)
    if db_score_spectral < best_db_score_spectral:
        best_db_score_spectral = db_score_spectral
        best_n_clusters_spectral = n_clusters
        best_labels_spectral = labels_spectral
    if db_score_kmeans < best_db_score_kmeans:
        best_db_score_kmeans = db_score_kmeans
        best_n_clusters_kmeans = n_clusters
        best_labels_kmeans = labels_kmeans
# 6.输出谱聚类的最佳聚类数和DB指数
print(f"谱聚类最佳聚类数: {best_n_clusters_spectral}")
print(f"谱聚类最佳DB指数: {best_db_score_spectral:.3f}")
# 7.绘制谱聚类结果
plt.figure(figsize=(14, 9))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=best_labels_spectral,
                alpha=1, palette="Set2",s=100)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize=20,bbox_to_anchor=(1, 1))
plt.title("谱聚类结果", fontsize=40)
plt.show()
# 8.输出K-means的最佳聚类数和DB指数
print(f"K-means最佳聚类数: {best_n_clusters_kmeans}")
print(f"K-means最佳DB指数: {best_db_score_kmeans:.3f}")
# 9.绘制K-means聚类结果
plt.figure(figsize=(14, 9))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=best_labels_kmeans,
                alpha=1, palette="Set2",s=100)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("x", fontsize=40)
plt.ylabel("y", fontsize=40)
plt.legend(fontsize=20,bbox_to_anchor=(1, 1))
plt.title("K-means聚类结果", fontsize=40)
plt.show()
