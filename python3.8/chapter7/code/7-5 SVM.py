from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# 1.读入数据
data = pd.read_csv('../data/inferior user.csv', encoding='ansi')
print(data['是否为不良账户'].value_counts(), '\n')  # 查看数据平衡性
# 2.划分训练集和测试集
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 3.特征归一化
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
# 4.构建绘制不同gamma值下的模型评价指标的函数
def evaluate_gamma_values(X_train_standard, X_test_standard, y_train, y_test, gamma_values):
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    for gamma in gamma_values:
        clf = SVC(kernel='rbf', gamma=gamma)  # 构建SVM模型,核函数选择高斯核函数
        clf.fit(X_train_standard, y_train)  # 使用训练集训练模型
        y_pred = clf.predict(X_test_standard)  # 对测试集预测相应的类别
        accuracy_list.append(accuracy_score(y_pred, y_test))
        precision_list.append(precision_score(y_pred, y_test))
        recall_list.append(recall_score(y_pred, y_test))
        f1_list.append(f1_score(y_pred, y_test))
    return accuracy_list, precision_list, recall_list, f1_list
# 5.生成不同范围的gamma值列表
gamma_values1 = np.linspace(0.01, 1, 100)
gamma_values2 = np.linspace(0.001, 0.1, 100)
accuracy_list1, precision_list1, recall_list1, f1_list1 = evaluate_gamma_values(X_train_standard, X_test_standard, y_train, y_test, gamma_values1)
accuracy_list2, precision_list2, recall_list2, f1_list2 = evaluate_gamma_values(X_train_standard, X_test_standard, y_train, y_test, gamma_values2)
# 6.绘制不同gamma值下的模型评价指标
plt.rcParams['font.sans-serif']=['STSong']     
plt.rcParams['axes.unicode_minus'] = False  # 设置负号正常显示
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(gamma_values1, accuracy_list1)
plt.plot(gamma_values1, precision_list1)
plt.plot(gamma_values1, recall_list1)
plt.plot(gamma_values1, f1_list1)
plt.title('gamma 0.1-0.99')
plt.xlabel('gamma')
plt.subplot(1, 2, 2)
plt.plot(gamma_values2, accuracy_list2, label='准确率')
plt.plot(gamma_values2, precision_list2, label='精准率')
plt.plot(gamma_values2, recall_list2, label='召回率')
plt.plot(gamma_values2, f1_list2, label='F1得分')
plt.legend(ncol=2, frameon=False,loc='upper right', fontsize=13)
plt.title('gamma 0.01-0.1')
plt.xlabel('gamma')
plt.tight_layout()
plt.show()