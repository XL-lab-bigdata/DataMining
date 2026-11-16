import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
# 1.读取数据集、数据预处理
data = pd.read_csv('../data/auto_mpg.csv', sep=',')
data = data.dropna()  # 缺失值处理
X = data[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = data.iloc[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 将数据集分为训练集和测试集
X_train_scaled = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)  # 对训练集进行标准化
X_test_scaled = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)  # 对测试集进行标准化
# 2.计算VIF，检验多重共线性
vif = [variance_inflation_factor(X_train_scaled.values, i) for i in range(X_train_scaled.shape[1])]
vif = [round(v, 3) for v in vif]
print('VIF:', vif)
# 3.建立岭回归模型
ridge = Ridge(alpha=1)
ridge.fit(X_train_scaled, y_train)
print('Intercept:', round(ridge.intercept_, 3))
print('Coefficients:', [round(coef, 3) for coef in ridge.coef_])
# 4.预测、模型评估
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
print('Ridge RMSE:', round(rmse_ridge, 3))
r2_ridge = r2_score(y_test, y_pred_ridge)
print('Ridge R2:', round(r2_ridge, 3))