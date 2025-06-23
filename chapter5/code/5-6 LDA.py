import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#1.生成三维数据集
X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, n_repeated=0,
                           n_classes=3, n_clusters_per_class=1, class_sep=0.5, random_state=10)

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams.update({'font.size': 35})
fig = plt.figure(figsize=(20, 10))
#2.绘制原始数据集的散点图
ax = fig.add_subplot(121, projection='3d')
shapes = ['o', '^', 's']
colors = ['blue', 'green', 'yellow']
color_mapping = {0: colors[0], 1: colors[1], 2: colors[2]}
for i in range(3):
    markers = shapes[i]
    scatter_colors = color_mapping[i]
    ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2], marker=markers, c=scatter_colors, label=f'Class {i}')
ax.legend(frameon=False)
#3.调用LDA算法进行降维，并拟合数据
lda = LinearDiscriminantAnalysis(n_components=2)
X_new = lda.fit_transform(X, y)
plt.subplot(122)
for i in range(3):
    markers = shapes[i]
    scatter_colors = colors[i]
    plt.scatter(X_new[y == i, 0], X_new[y == i, 1], marker=markers, c=scatter_colors, label=f'Class {i}')
plt.legend(frameon=False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()