from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
# 1. 生成数据集
digits = load_digits(n_class=8)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n = 5000
X, y = X[:n], y[:n]
# 2. 调用UMAP算法将数据集降至二维
reducer_umap = UMAP(n_components=2, random_state=0)
embedding_umap = reducer_umap.fit_transform(X)
scaler_umap = MinMaxScaler()
embedding_umap = scaler_umap.fit_transform(embedding_umap)
# 3. 调用t-SNE算法将数据集降至二维
reducer_tsne = TSNE(n_components=2, random_state=0)
embedding_tsne = reducer_tsne.fit_transform(X)
scaler_tsne = MinMaxScaler()
embedding_tsne = scaler_tsne.fit_transform(embedding_tsne)
# 4. 可视化UMAP算法结果
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams.update({'font.size': 35})
fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('UMAP')
for i in range(embedding_umap.shape[0]):
 plt.text(
 embedding_umap[i, 0],
 embedding_umap[i, 1],
 str(y[i]),
 color=plt.cm.tab10(y[i]),
 fontdict={"weight": "bold"},
 va="center",
 ha="center",
)
plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=y, cmap='viridis', s=30, edgecolors='k')
for digit in range(len(np.unique(y))):
 digit_indices = np.where(y == digit)
 plt.text(
 np.mean(embedding_umap[digit_indices, 0]),
 np.mean(embedding_umap[digit_indices, 1]),
 str(digit),
 color='black',
 va="center",
 ha="center",
)
plt.tick_params(axis='both', which='major')
# 5. 可视化t-SNE算法结果
plt.subplot(1, 2, 2)
plt.title('t-SNE')
for i in range(embedding_tsne.shape[0]):
 plt.text(
 embedding_tsne[i, 0],
 embedding_tsne[i, 1],
 str(y[i]),
 color=plt.cm.tab10(y[i]),
 va="center",
 ha="center",
)
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=y, cmap='viridis', s=30, edgecolors='k')
for digit in range(len(np.unique(y))):
 digit_indices = np.where(y == digit)
 plt.text(
 np.mean(embedding_tsne[digit_indices, 0]),
 np.mean(embedding_tsne[digit_indices, 1]),
 str(digit),
 color='black',
 va="center",
 ha="center",
)
plt.tick_params(axis='both', which='major')
plt.tight_layout()
plt.show()