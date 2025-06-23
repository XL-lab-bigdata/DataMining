import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
# 1.读入数据
data = pd.read_csv('../data/liver patient.csv', encoding='ansi')
# 2.对性别特征进行独热编码
dummies = pd.get_dummies(data['患者性别'], prefix='患者性别')  
data = data.join(dummies).drop(['患者性别'], axis=1)
# 3.用均值填充缺失值
print("数据的基本描述：")
print(round(data.describe(),2))
data['白蛋白与球蛋白的比例'] = data['白蛋白与球蛋白的比例'].fillna(data['白蛋白与球蛋白的比例'].mean())
# 4.划分训练集和测试集
X = data.drop(columns=['是否患有肝病'])
y = abs(data['是否患有肝病'] - 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 5.特征归一化
standardScaler = StandardScaler()
X_train_standard = standardScaler.fit_transform(X_train)
X_test_standard = standardScaler.transform(X_test)
# 6.构建及训练模型
log_reg = LogisticRegression(C=0.1)
log_reg.fit(X_train_standard, y_train)
# 7.预测并评估模型
y_predict = log_reg.predict(X_test_standard)
print("权重:", log_reg.coef_)
print("截距:", log_reg.intercept_)
print("精准率:", round(precision_score(y_predict, y_test),3))
# 8.计算事件优势比
def calculate_odds(feature_values, labels):
    df = pd.DataFrame({'Feature': feature_values, 'Label': labels})
    odds = df[df['Label'] == 1]['Feature'].mean() / df[df['Label'] == 2]['Feature'].mean()
    return odds
feature_names = X.columns
odds_list = [calculate_odds(data[x], data['是否患有肝病']) for x in X.columns]
feature_odds = sorted(zip(feature_names, odds_list), key=lambda x: abs(x[1]), reverse=True)
sorted_feature_names, sorted_odds = zip(*feature_odds)