import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score	
from matplotlib import pyplot as plt

# 1.数据预处理
data = pd.read_csv('../data/crisis.csv', encoding='ansi')  # 读入数据
data = data.drop(['国家'], axis=1)  # 删除特征
data['银行危机'] = LabelEncoder().fit_transform(data['银行危机']) # 对“银行危机”进行编码
X_train, X_test, y_train, y_test = train_test_split(data.drop('银行危机', axis=1), data['银行危机'], random_state=666)  # 划分数据集

# 2.模型构建和训练
dt_clf = DecisionTreeClassifier()  # 构建决策树模型
dt_clf.fit(X_train, y_train)  # 训练决策树模型
y_pred_dt = dt_clf.predict(X_test)  # 用决策树模型进行预测
rf_clf = RandomForestClassifier()  # 构建随机森林模型
rf_clf.fit(X_train, y_train)  # 训练随机森林模型
y_pred_rf = rf_clf.predict(X_test)  # 用随机森林模型进行预测

# 3.打印模型评价
def print_model_evaluation(model_name, y_pred):
    print(f'***** {model_name} *****')
    print('accuracy', accuracy_score(y_pred, y_test))
    print('precision', precision_score(y_pred, y_test))
    print('recall', recall_score(y_pred, y_test))
    print('f1 score', f1_score(y_pred, y_test))
print_model_evaluation('决策树', y_pred_dt)
print_model_evaluation('随机森林', y_pred_rf)

# 4.绘制n_estimators参数对预测性能的影响曲线
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
for n in range(1,101):
    rf_clf = RandomForestClassifier(n_estimators=n)  # 构建随机森林模型
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    accuracy_list.append(accuracy_score(y_pred_rf, y_test))
    precision_list.append(precision_score(y_pred_rf, y_test))
    recall_list.append(recall_score(y_pred_rf, y_test))
    f1_list.append(f1_score(y_pred_rf, y_test))

# 5.绘制性能曲线
plt.rcParams['font.sans-serif']=['STSong']     
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(4, 4))
plt.plot(list(range(1, 101)), accuracy_list, label='accuracy')
plt.plot(list(range(1, 101)), precision_list, label='precision')
plt.plot(list(range(1, 101)), recall_list, label='recall')
plt.plot(list(range(1, 101)), f1_list, label='f1 score')
plt.xlabel('n_estimators')
plt.ylabel('评价指标')
plt.legend()
plt.show()