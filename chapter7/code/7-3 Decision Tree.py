import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np

# 1.读入数据集
data = pd.read_csv('../data/Customer Churn.csv', encoding='ansi')  
# 2.查看数据缺失情况并处理
print(data['飞机乘坐情况'].value_counts(), '\n') 
data['飞机乘坐情况'].replace(to_replace='No Record', value='No', inplace=True)  # 缺失值众数填充
# 3.特征编码
mapping1 = {'Yes': 1, 'No': 0}
data['飞机乘坐情况'] = data['飞机乘坐情况'].map(mapping1)
data['社交媒体同步情况'] = data['社交媒体同步情况'].map(mapping1)
data['预订住宿情况'] = data['预订住宿情况'].map(mapping1)
mapping2 = {'Low Income': 1, 'Middle Income': 2, 'High Income': 3}
data['年收入等级'] = data['年收入等级'].map(mapping2)
# 4.划分训练集和测试集
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)
# 5.使用网格搜索寻找最佳参数组合
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [0.02, 0.04, 0.06, 0.08, 0.1],
    'min_samples_leaf': [0.02, 0.04, 0.06, 0.08, 0.1]
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("最佳参数组合为: ", best_params)
# 6.使用最佳参数构建模型
model = DecisionTreeClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 7.输出模型评估指标
print("召回率：", round(recall_score(y_test, y_pred), 3))
# 8.绘制决策树
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=list(X_train.columns), class_names=['流失' if x == 1 else '未流失' for x in model.classes_], fontsize=10)
plt.show()