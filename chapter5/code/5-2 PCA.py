import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 1. 数据加载与预处理
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
species_names = iris.target_names

# 2. PCA降维
pca = PCA(n_components=2)
pca_2D = pca.fit_transform(iris_scaled)
pca2D_df = pd.DataFrame(pca_2D, columns=['PC1', 'PC2'])
pca2D_df['cluster'] = species_names[iris.target]  # 直接使用品种名称

# 3. 可视化设置
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['SimSun'], 'font.size': 15})
palette = sns.color_palette("husl", 3)
def confidence_ellipse(data, ax, n_std=3.0, facecolor='none', **kwargs): # 椭圆绘制函数
    x, y = data['PC1'], data['PC2']
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2, angle=np.degrees(np.arctan2(*v[:, 0][::-1])), facecolor=facecolor,**kwargs)
    return ax.add_patch(ellipse)
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150) # 创建画布
scatter = sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette=palette, data=pca2D_df, s=60, alpha=0.85, edgecolor='w', linewidth=0.8, ax=ax) # 绘制带品种名称的散点图
for idx, species in enumerate(species_names):  # 为每个品种添加椭圆
    cluster_data = pca2D_df[pca2D_df['cluster'] == species]
    confidence_ellipse(cluster_data, ax=ax, n_std=2.2,edgecolor=palette[idx], linewidth=1.8,zorder=0, facecolor=palette[idx],alpha=0.12, linestyle='--')
ax.legend(loc='lower right',frameon=False)
ax.set_xlabel('主成分1 (PC1)')
ax.set_ylabel('主成分2 (PC2)')
ax.grid(True, alpha=0.25, linestyle=':')
plt.tight_layout()
plt.show()

# 4. 主成分载荷分析
loadings = pd.DataFrame(
pca.components_.T,
columns=['PC1', 'PC2'],
index=iris_scaled.columns)
print("\nPC1 主要载荷特征：\n", loadings['PC1'].abs().nlargest(5))
49. print("\nPC2 主要载荷特征：\n", loadings['PC2'].abs().nlargest(5))