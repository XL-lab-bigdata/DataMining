import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
# 1.读入数据
data = pd.read_csv('../data/liver patient.csv', encoding='gb2312')
# 2.对性别特征进行独热编码
dummies = pd.get_dummies(data['患者性别'], prefix='患者性别')  
data = data.join(dummies).drop(['患者性别'], axis=1)
# 3.查看数据的基本描述
print("数据的基本描述：")
print(round(data.describe(),2))
# 4.划分训练集和测试集，用训练集均值填充缺失值
X = data.drop(columns=['是否患有肝病'])
y = abs(data['是否患有肝病'] - 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
ratio_column = '白蛋白与球蛋白的比例'
fill_values = {ratio_column: X_train[ratio_column].mean()}
X_train = X_train.fillna(fill_values)
X_test = X_test.fillna(fill_values)
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
print("精准率:", round(precision_score(y_test, y_predict),3))
# 8.计算其他特征固定时的患病优势比
coefficients = pd.Series(log_reg.coef_[0], index=X.columns)
odds_ratios = np.exp(coefficients.drop(dummies.columns))
print("患病优势比（数值特征每增加一个训练集标准差）：")
print(odds_ratios.sort_values(ascending=False).round(3))
unscaled_coefficients = coefficients / standardScaler.scale_
sex_log_odds_ratio = unscaled_coefficients['患者性别_Female'] - unscaled_coefficients['患者性别_Male']
print("患病优势比（女性相对男性）：", round(np.exp(sex_log_odds_ratio), 3))
