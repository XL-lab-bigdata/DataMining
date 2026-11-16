import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.datasets import fetch_olivetti_faces

# 1.读取数据
faces = fetch_olivetti_faces()
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

    for point, image, label in zip(pts, X, y):
        imagebox = OffsetImage(image.reshape(64, 64), zoom=0.3, cmap=plt.cm.gray)
        annotation = AnnotationBbox(
            imagebox, tuple(point),
            bboxprops={'boxstyle': 'square,pad=0.08', 'edgecolor': color[label],
                       'linewidth': 1, 'facecolor': 'none'},
        )
        ax.add_artist(annotation)

    ax.legend(fontsize='large', title_fontsize='large')
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 20})
    ax.set_title(title)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('mds1.png', dpi=600)
    plt.show()

# 3.计算欧氏距离矩阵
dist_euclid = euclidean_distances(X_faces)
mapData(dist_euclid, X_faces, y_faces, True, 'MDS')
