import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#1.构建数据集
digits = datasets.load_digits(n_class=5)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n = 15
img = np.zeros((10 * n, 10 * n))
#2.可视化原始数据集
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8 , iy:iy +8 ] = X[i * n + j].reshape((8,8))
plt.figure(figsize=(8,8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()
# 3.调用TNE函数，拟合数据
tsne = manifold.TSNE(n_components=2, random_state=501)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
#4.可视化降维结果
plt.figure(figsize=(8,8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i,0], X_norm[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()