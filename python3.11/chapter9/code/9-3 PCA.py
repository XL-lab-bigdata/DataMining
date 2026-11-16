import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 1.读取数据
engine_data = pd.read_csv('../data/Engine_data.csv')
# 2.使用PCA进行降维，将数据压缩到3个主成分中
pca = PCA(n_components=3)
engine_data_pca = pca.fit_transform(engine_data)
# 3.计算样本的对数似然，分数越低表示越异常
scores = pca.score_samples(engine_data)
# 4.绘图
font1 = {'family': 'STSong','weight': 'normal','size': 18}  # 画图字体设置
def plot_fig(ax, Y, scores, elev, azim):
    p = ax.scatter(Y[::10, 0], Y[::10, 1], Y[::10, 2], c=scores[::10], marker='+', alpha=0.4)
    fig.colorbar(p)   # 绘制colorbar
    x, y, z = 3 * pca.components_
    planes = [np.r_[axis[:2], -axis[1::-1]].reshape((2, 2)) for axis in [x, y, z]]
    # 设置坐标轴标题
    ax.set_xlabel('主成分1', fontdict=font1, labelpad=15)
    ax.set_ylabel('主成分2', fontdict=font1, labelpad=15)
    ax.set_zlabel('主成分3', fontdict=font1, labelpad=15)
    ax.view_init(elev=elev, azim=azim)
    # 调整XYZ轴数字的大小
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121, projection='3d')
plot_fig(ax, engine_data_pca, scores, elev=30, azim=-120)  # azim为绕z轴的方位角，elev为相对于xy平面的仰角
ax = fig.add_subplot(122, projection='3d')
plot_fig(ax, engine_data_pca, scores, elev=10, azim=-130)  # 角度二
plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()
