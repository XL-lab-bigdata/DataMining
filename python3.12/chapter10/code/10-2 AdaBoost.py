import pandas as pd
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import random
from itertools import accumulate
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

# 1.数据预处理
df_acc_list, ad_acc_list = [], []
data = pd.read_csv("../data/horseColic.csv", names=[i for i in range(28)])  # 导入数据
case_ids = data[2]  # 第三列为医院病例号
X = data.iloc[:, :-1].drop(columns=2)
y = data.iloc[:, -1]
model_num = 100

# 2.模型构建和训练
for seed in random.sample(range(1, model_num*2), model_num):
    # 按病例号划分，同一病例的记录留在同一侧
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=case_ids))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    #构建决策树分类模型
    dtc = DecisionTreeClassifier(max_depth=7, min_samples_leaf=7)
    dtc.fit(X_train, y_train)
    df_acc_list.append(accuracy_score(y_test, dtc.predict(X_test)))
    #构建Adaboost分类模型
    clf = ensemble.AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=7, min_samples_leaf=7), n_estimators=100, algorithm='SAMME', learning_rate=0.95)
    clf.fit(X_train, y_train)
    ad_acc_list.append(accuracy_score(y_test, clf.predict(X_test)))

# 3.绘制性能曲线
plt.figure(figsize=(15,5))
font = {'family': 'serif', 'serif': ['STSong'], 'size':25}
plt.rc('font', **font)
plt.plot(list(range(1, model_num+1)), np.array(list(accumulate(df_acc_list)))/np.array(range(1, model_num+1)), label='决策树')
plt.plot(list(range(1, model_num+1)), np.array(list(accumulate(ad_acc_list)))/np.array(range(1, model_num+1)), label='AdaBoost')
plt.legend(frameon=False)
plt.xlabel('实验次数')
plt.ylabel('平均准确率')
plt.show()

# 4.打印模型评价
print('决策树平均准确率：', round(sum(df_acc_list)/len(df_acc_list),4))
print('AdaBoost平均准确率：', round(sum(ad_acc_list)/len(ad_acc_list),4))
