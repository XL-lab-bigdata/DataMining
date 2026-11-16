import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import calinski_harabasz_score
# 1.设置绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)
# 2.读取数据
original = pd.read_parquet("../data/openfoodfacts.parquet")
nutrition_cols = ["energy_100g", "fat_100g", "carbohydrates_100g",
                  "sugars_100g", "proteins_100g", "salt_100g"]
nutrition_table = original[nutrition_cols].dropna()
# 3.数据预处理：标准化和PCA降维
X_train = StandardScaler().fit_transform(nutrition_table)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """用给定的位置和协方差画一个椭圆"""
    ax = ax or plt.gca()
    #将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    #画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width,
                             nsig * height, angle, **kwargs))
def plot_gmm(gmm):
    for pos, covar, w in zip(gmm.means_,
                             gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=0.1 )
    plt.show()
# 4.评估不同簇数量下的聚类性能
scores = []
n_components_range = range(2, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type="full", random_state=1, max_iter=300)
    labels = gmm.fit_predict(X_train_pca)
    score = calinski_harabasz_score(X_train_pca, labels)
    scores.append(score)
# 5.确定最佳聚类数
best_n_components = n_components_range[np.argmax(scores)]
print(f"最佳聚类数：{best_n_components}，"
      f"Calinski-Harabasz指数为 {max(scores):.3f}")
# 6.使用最佳聚类数进行聚类
best_gmm = GaussianMixture(n_components=best_n_components,
                           covariance_type="full", random_state=1, max_iter=300)
best_labels = best_gmm.fit_predict(X_train_pca)
print("前5个样本分属不同簇的概率分布：")
print(np.round(best_gmm.predict_proba(X_train_pca[:5]), decimals=3))
print("前5个样本分属不同簇的标签：")
print(best_gmm.predict(X_train_pca[:5]))
# 7.可视化不同簇数量下的Calinski-Harabasz分数
plt.figure(figsize=(10, 8))
plt.rc('font', size=30)
plt.plot(n_components_range, scores, marker='o',c="dodgerblue")
plt.xlabel('聚类数',fontsize=30)
plt.ylabel('Calinski-Harabasz指数',fontsize=30)
plt.ticklabel_format(style="sci",scilimits=(-1,2),axis="y")
plt.xticks(n_components_range,fontsize=30)
plt.yticks(fontsize=30)
plt.show()
# 8.可视化聚类结果
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1],
                hue=best_labels, palette="Set2")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plot_gmm(best_gmm)