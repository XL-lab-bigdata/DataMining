import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap
from matplotlib.ticker import MaxNLocator
# 1.构建数据集
n_points = 5000
X, color = make_s_curve(n_points, random_state=0)
n_neighbors = 30
n_components = 2
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams.update({'font.size': 35})
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)
ax.tick_params(axis='both', which='major')  # Increase axis label size
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.zaxis.set_major_locator(MaxNLocator(integer=True))
# 2.设置近邻点个数为50，将样本点映射至二维空间中
Y = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X)
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax2.tick_params(axis='both', which='major')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()