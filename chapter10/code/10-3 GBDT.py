import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import warnings
import random
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
warnings.filterwarnings("ignore")  # 忽略所有警告

# 1.数据预处理
df_acc_list, ad_acc_list, gbc_acc_list = [], [], []
data = pd.read_csv("../data/charging_pile.csv") # 导入数据
X = data.drop(columns=['id','label'], axis=1)
y = data['label']
print(y.value_counts())

# 2.模型构建和训练
for seed in tqdm(random.sample(range(1, 101), 50)):
    # 随机划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= seed)
    #构建决策树分类模型
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train.ravel())
    df_acc_list.append(accuracy_score(y_test, dtc.predict(X_test)))
    #构建Adaboost分类模型
    clf = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, algorithm='SAMME', learning_rate=0.95)
    clf.fit(X_train, y_train.ravel())
    ad_acc_list.append(accuracy_score(y_test, clf.predict(X_test)))
    # 构建GBDT分类模型
    gbc = ensemble.GradientBoostingClassifier()
    gbc.fit(X_train, y_train.ravel())
    gbc_acc_list.append(accuracy_score(y_test, gbc.predict(X_test)))

# 3.绘制性能曲线
plt.figure(figsize=(10,5))
font = {'family': 'serif', 'serif': ['Times New Roman'], 'size':25}
plt.rc('font', **font)
plt.plot(list(range(1, 51)), df_acc_list, label='Decision tree')
plt.plot(list(range(1, 51)), ad_acc_list, label='AdaBoost')
plt.plot(list(range(1, 51)), gbc_acc_list, label='GBDT')
plt.legend(fontsize='x-small', frameon=False, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.8))
plt.xlabel('iter')
plt.ylabel('Accuracy')
plt.show()

# 4.打印模型评价
print('决策树的平均准确率：', sum(df_acc_list)/len(df_acc_list))
print('AdaBoost的平均准确率：', sum(ad_acc_list)/len(ad_acc_list))
print('梯度提升树的平均准确率：', sum(gbc_acc_list)/len(gbc_acc_list))