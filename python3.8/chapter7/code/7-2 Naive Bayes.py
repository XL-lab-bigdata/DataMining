import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 1.读入数据、数据一致性处理
data = pd.read_csv('../data/star.csv', encoding='ansi')  # 读入数据集
print("颜色特征的不同取值：", set(data['颜色'])) # 查看“颜色”特征的取值
data['颜色'] = data['颜色'].str.replace('Blue white|Blue-White|Blue-white', 'Blue White', regex=True) 
data['颜色'] = data['颜色'].replace({'white': 'White', 'yellowish': 'Yellowish', 
                                     'yellow-white': 'White-Yellow'})
# 2.特征编码
dummies_color = pd.get_dummies(data['颜色'],prefix='颜色')  # 对颜色特征进行独热编码
dummies_smass = pd.get_dummies(data['SMASS规格'],prefix='SMASS规格')  # 对SMASS规格特征进行独热编码
data = pd.concat([data, dummies_color, dummies_smass], axis=1)  # 将编码后的特征并入数据集
color = data['颜色']
data = data.drop(['颜色', 'SMASS规格'], axis=1)  # 删除编码前的特征
# 3.划分训练集和测试集
X, y = data.drop(columns=['类型']), data['类型']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=200)  # 随即划分训练集和测试集
# 4.可视化数据样本，构造颜色字典和类型字典，不同的marker代表不同类型的行星
color_dict={'Yellowish': "#FFFF99", 'White-Yellow': "#FFFFE6", 
         'Pale yellow orange': "#FFD700", 'White': "#E6E6E6", 
         'Red': "#FF3333", 'Blue': "#3333FF", 
         'Orange-Red': "#FF6633", 'Whitish': "#FDFDFD", 
         'Blue White': "#D8D8D8", 'Yellowish White': "#FFFFCC",
         'Orange': "#FF9933"}
marker_dict = {0: "p", 1: "h", 2: "s",  3: "p",  4: "*",  5: "o"}
plt.rcParams['font.sans-serif'] = ['STSong']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置负号正常显示
plt.figure(figsize=(6,3))  # 设置画布大小
for idx, (subset, title) in enumerate([(X_train, '训练集'), (X_test, '测试集')], 1):
    plt.subplot(1, 2, idx)
    color_subset = color.iloc[subset.index]
    subset = subset.reset_index(drop=True)
    for i in range(len(subset)):
        plt.scatter(subset['温度'][i], subset['相对光度'][i], 
                    s=np.log2(subset['相对半径'][i] + 1) * 50,
                    alpha=0.8, marker=marker_dict[list(y_train)[i]], 
                    edgecolor='white', color=color_dict[list(color_subset)[i]])
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    plt.xlabel('温度', fontsize=12)
    plt.ylabel('相对光度', fontsize=12)
    plt.title(title)
    plt.grid(False)
plt.tight_layout()
plt.show()
# 5.构建、训练、评价模型
clf = GaussianNB()  # 构建模型
clf.fit(X_train, y_train)  # 训练模型
y_pred = clf.predict(X_test)  # 用训练完成的模型对测试集进行预测
print('预测准确率：', round(accuracy_score(y_pred, y_test),3))  # 评价模型
# 6.计算混淆矩阵并绘制热力图
cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
f, ax = plt.subplots(figsize=(3,2.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
ax.set_xlabel('预测值') #x轴
ax.set_ylabel('真实值') #y轴
plt.tight_layout()
plt.show()