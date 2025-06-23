from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 1.生成随机数据和标签
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# 2.使用K-means聚类
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)
# 3.可视化数据和聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()
# 4.计算外部指标
print("列联矩阵为\n",metrics.cluster.contingency_matrix(y_true, y_kmeans))
print("对称混淆矩阵为\n",metrics.cluster.pair_confusion_matrix(y_true, y_kmeans))
print("Fowlkes-Mallows指数为{:.3f}".
      format(metrics.fowlkes_mallows_score(y_true, y_kmeans)))
print("兰德系数为{:.3f}".
      format(metrics.rand_score(y_true, y_kmeans)))
print("调整兰德系数为{:.3f}".
      format(metrics.adjusted_rand_score(y_true, y_kmeans)))
print("互信息为{:.3f}".
      format(metrics.mutual_info_score(y_true, y_kmeans)))
print("标准化互信息为{:.3f}".
      format(metrics.normalized_mutual_info_score(y_true, y_kmeans)))
print("调整互信息为{:.3f}".
      format(metrics.adjusted_mutual_info_score(y_true, y_kmeans)))
print("同质性为{:.3f}".
      format(metrics.homogeneity_score(y_true, y_kmeans)))
print("完整性为{:.3f}".
      format(metrics.completeness_score(y_true, y_kmeans)))
print("V-measure为{:.3f}".
      format(metrics.v_measure_score(y_true, y_kmeans)))
# 5.计算内部指标
print("轮廓系数为{:.3f}".
      format(metrics.silhouette_score(X,y_kmeans)))
print("Calinski-Harabasz指数为{:.3f}".
      format(metrics.calinski_harabasz_score(X, y_kmeans)))
print("Davies-Bouldin指数为{:.3f}".
      format(metrics.davies_bouldin_score(X, y_kmeans)))
# 6.使用K-means聚类
kmeans = KMeans(n_clusters=6, random_state=0)
y_kmeans = kmeans.fit_predict(X)
# 7.可视化数据和聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()
# 8.计算外部指标
print("列联矩阵为\n",metrics.cluster.contingency_matrix(y_true, y_kmeans))
print("对称混淆矩阵为\n",metrics.cluster.pair_confusion_matrix(y_true, y_kmeans))
print("Fowlkes-Mallows指数为{:.3f}".
      format(metrics.fowlkes_mallows_score(y_true, y_kmeans)))
print("兰德系数为{:.3f}".
      format(metrics.rand_score(y_true, y_kmeans)))
print("调整兰德系数为{:.3f}".
      format(metrics.adjusted_rand_score(y_true, y_kmeans)))
print("互信息为{:.3f}".
      format(metrics.mutual_info_score(y_true, y_kmeans)))
print("标准化互信息为{:.3f}".
      format(metrics.normalized_mutual_info_score(y_true, y_kmeans)))
print("调整互信息为{:.3f}".
      format(metrics.adjusted_mutual_info_score(y_true, y_kmeans)))
print("同质性为{:.3f}".
      format(metrics.homogeneity_score(y_true, y_kmeans)))
print("完整性为{:.3f}".
      format(metrics.completeness_score(y_true, y_kmeans)))
print("V-measure为{:.3f}".
      format(metrics.v_measure_score(y_true, y_kmeans)))
# 9.计算内部指标
print("轮廓系数为{:.3f}".
      format(metrics.silhouette_score(X,y_kmeans)))
print("Calinski-Harabasz指数为{:.3f}".
      format(metrics.calinski_harabasz_score(X, y_kmeans)))
print("Davies-Bouldin指数为{:.3f}".
      format(metrics.davies_bouldin_score(X, y_kmeans)))