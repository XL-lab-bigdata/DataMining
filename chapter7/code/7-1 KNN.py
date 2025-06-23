import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1.读入数据集
data = pd.read_csv('../data/advertisement.csv', encoding='ansi')
# 2.划分训练集和测试集
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 3.特征归一化
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
# 4.构建、训练模型
knn_clf = KNeighborsClassifier(n_neighbors=10, p=1, weights='distance') 
knn_clf.fit(X_train_standard, y_train)
y_predict = knn_clf.predict(X_test_standard)
# 5.评价模型
print('accuracy', accuracy_score(y_predict, y_test))

import matplotlib.pyplot as plt
import numpy as np
# 绘制决策边界的函数
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.arange(axis[0], axis[1], 0.1), 
        np.arange(axis[2], axis[3], 0.1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EDD3ED', '#FFFCCC', '#C8EFD4'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
# 6. 绘制不同k值下的模型决策边界
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12, 3.5))
params = [(1, 'K=1'), (9, 'K=9'), (49, 'K=49')]  # 设置子图个数和参数
for i, (k, title) in enumerate(params, start=1):
    plt.subplot(1, 3, i)
    knn_clf_all = KNeighborsClassifier(n_neighbors=k)
    knn_clf_all.fit(X_train[['每天花在网站上的时间', '用户年龄']], y_train)
    plot_decision_boundary(knn_clf_all, axis=[30, 95, 15, 65])
    type1 = X_test[y_test==0]
    type2 = X_test[y_test==1]
    y_predict = knn_clf_all.predict(X_test[['每天花在网站上的时间', '用户年龄']])
    print('K=',k, ',accuracy=', accuracy_score(y_predict, y_test))
    plt.scatter(type1['每天花在网站上的时间'], type1['用户年龄'], color='#FF7F27', edgecolors='k', s=35, label='否')
    plt.scatter(type2['每天花在网站上的时间'], type2['用户年龄'], color='#92D050', edgecolors='k', s=35, label='是')
    plt.title(title)
    plt.xlabel('每日网站浏览时长')
    plt.ylabel('用户年龄')
    if i == 1:
        plt.legend(ncol=2, loc='upper left')
plt.tight_layout()
plt.show()