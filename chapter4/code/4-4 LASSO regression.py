import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
# 1.读取数据集、数据预处理
data = pd.read_csv('../data/auto_mpg.csv', sep=',')
data = data.dropna()  # 缺失值处理
X = data[['displacement', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = data.iloc[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 将数据集分为训练集和测试集
X_train_scaled = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)  # 对训练集进行标准化
X_test_scaled = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)  # 对测试集进行标准化
# 2.训练并评估多元线性模型
model_LR = LinearRegression()
model_LR.fit(X_train_scaled, y_train)
y_pred_LR = model_LR.predict(X_test_scaled)
r2_LR = r2_score(y_test, y_pred_LR)
rmse_LR = mean_squared_error(y_test, y_pred_LR, squared=False)
print('线性回归 R2 分数:', round(r2_LR, 3))
print('线性回归 RMSE:', round(rmse_LR, 3))
# 3.计算VIF，检验多重共线性
vif = [variance_inflation_factor(X_train_scaled.values, i) for i in range(X_train_scaled.shape[1])]
vif = [round(v, 3) for v in vif]
print('VIF:', vif)
# 4.训练并评估岭回归模型
ridge = Ridge(alpha=1)
ridge.fit(X_train_scaled, y_train)
print('Intercept:', round(ridge.intercept_, 3))
print('Coefficients:', [round(coef, 3) for coef in ridge.coef_])
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
print('Ridge RMSE:', round(rmse_ridge, 3))
r2_ridge = r2_score(y_test, y_pred_ridge)
print('Ridge R2:', round(r2_ridge, 3))
# 5.建立LASSO回归模型
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
print("LASSO模型系数：", np.round(lasso.coef_, 3))
print("LASSO模型截距：", round(lasso.intercept_, 3))
y_pred_lasso = lasso.predict(X_test_scaled)  # 预测测试集结果
# 6.模型评估
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print('LASSO 回归 R2:', round(r2_lasso, 3))
print('LASSO 回归 RMSE:', round(rmse_lasso, 3))
# 7.结果对比分析，此处略，详见4-4 LASSO regression.py
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 17})
plt.figure(figsize=(15, 5))

models = {'线性回归': y_pred_LR, '岭回归': y_pred_ridge, 'LASSO 回归': y_pred_lasso}
for i, (model_name, predictions) in enumerate(models.items()):
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_test, predictions)
    plt.plot([0, 60], [0, 60], '--', color='red')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(model_name)
plt.tight_layout()

names = X_train.columns  # 假设names是特征名称的列表
coefficients = [model_LR.coef_, ridge.coef_.flatten(), lasso.coef_]  # 模型的系数
model_names = ['线性回归', '岭回归', 'LASSO 回归']
bar_positions = np.arange(len(names))
bar_width = 0.25
colors = ['#F5CD78', '#CAE5EE', '#F2BDC9']  # 定义每个模型的颜色

plt.figure(figsize=(12, 6))
for i, (coef, name, color) in enumerate(zip(coefficients, model_names, colors)):
    plt.barh(bar_positions + i * bar_width, coef, height=bar_width, color=color, label=name)
plt.xlabel('系数大小')
plt.ylabel('特征')
plt.title('回归系数')
plt.yticks(bar_positions + bar_width, names)
plt.legend()

plt.show()
