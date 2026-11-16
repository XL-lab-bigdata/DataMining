import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.绘制树状层次图的函数
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)
# 3.导入数据并分离特征和标签
data = np.loadtxt('../data/seeds_dataset.txt')
X = data[:, :-1]
# 4.从数据中拟合层次聚类并绘制树状图
model = AgglomerativeClustering(distance_threshold=0,
                                n_clusters=None)
model = model.fit(X)
plt.figure(figsize=(10, 10))
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("簇中的样本数量",fontsize=30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 30)
plt.show()
# 5.用PCA进行降维以便于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# 6.层次聚类
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)
print("簇为3时, Calinski-Harabasz指数为{:.3f}".
      format(calinski_harabasz_score(X, labels)))
print("调整兰德系数值为{:.3f}".
      format( adjusted_rand_score(data[:, -1], labels)))
plt.figure(figsize=(10, 10))
sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1],
                hue=labels, palette="Set2")
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize=20)
plt.show()