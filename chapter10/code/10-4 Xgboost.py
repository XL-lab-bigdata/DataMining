import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from matplotlib import pyplot as plt

# 1.数据预处理
data = pd.read_csv('../data/cellphone.csv', encoding='ansi')  # 读入数据集
X_train, X_test, y_train, y_test = train_test_split(data.drop('价格范围', axis=1), data['价格范围'], random_state=666)  # 划分数据集

# 2.模型构建和训练
dt_clf = DecisionTreeClassifier()  # 构建决策树模型
dt_clf.fit(X_train, y_train)  # 训练决策树模型
dbdt_clf = GradientBoostingClassifier()  # 构建梯度提升树模型
dbdt_clf.fit(X_train, y_train)  # 构建梯度提升树模型
xgb_classifier = xgb.XGBClassifier(  # 构建XGBoost模型
    n_estimators=1000,  # 决策树的数量
    max_depth=3,       # 决策树的最大深度
    learning_rate=0.1, # 学习率
    random_state=42  # 随机种子
)
xgb_classifier.fit(X_train, y_train)  # 训练XGBoost模型

# 3.用三种模型对测试集进行预测
y_pred_dt = dt_clf.predict(X_test)
y_pred_gbdt = dbdt_clf.predict(X_test)
y_pred_xgb = xgb_classifier.predict(X_test)

# 4.打印模型评价
print('***** 决策树 *****')
print('accuracy', accuracy_score(y_pred_dt, y_test))
print('***** 梯度提升树 *****')
print('accuracy', accuracy_score(y_pred_gbdt, y_test))
print('***** XGBoost *****')
print('accuracy', accuracy_score(y_pred_xgb, y_test))

# 5.绘制特征重要性曲线
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文显示
xgb.plot_importance(xgb_classifier.get_booster(), max_num_features=20, importance_type='weight', xlabel='F分数', ylabel='特征', title='特征重要性', show_values=False, color='red')
plt.show()