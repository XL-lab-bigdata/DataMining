import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.datasets import fetch_olivetti_faces as fof

# 1.读取数据
faces = fof()
X_faces = faces.data
y_faces = faces.target
ind = y_faces < 5
X_faces = X_faces[ind, :]
y_faces = y_faces[ind]

plt.figure(figsize=(20, 10))
for i in range(50):  # 显示前50张面孔
    plt.subplot(5, 10, i + 1)
    plt.imshow(X_faces[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# 2.调用MDS算法对距离矩阵降维
def mapData(dist_matrix, X, y, metric, title):
    mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)
    pts = mds.fit_transform(dist_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    color = ['r', 'g', 'b', 'c', 'm']
    sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=y, palette=color, ax=ax)

    for x, ind in zip(X[1:], range(1, pts.shape[0])):
        im = x.reshape(64, 64)
        imagebox = OffsetImage(im, zoom=0.3, cmap=plt.cm.gray)
        i = pts[ind, 0]
        j = pts[ind, 1]
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)
        renderer = fig.canvas.get_renderer()
        bbox = ab.get_window_extent(renderer=renderer)
        w = (max(pts[:, 0]) - min(pts[:, 0])) / bbox.width
        h = (max(pts[:, 1]) - min(pts[:, 1])) / bbox.height
        rect = patches.Rectangle((i - w, j - h), w * 2, h * 2,
                                 edgecolor=color[y[ind]], linewidth=1, fill=False)
        ax.add_patch(rect)

    ax.legend(fontsize='large', title_fontsize='large')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 20})
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('mds1.png', dpi=600)
    plt.show()

# 3.计算欧氏距离矩阵
dist_euclid = euclidean_distances(X_faces)
mapData(dist_euclid, X_faces, y_faces, True, 'MDS')