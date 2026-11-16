import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
# 1.读取数据集
data = pd.read_excel('../data/IT income.xlsx')
X1 = data[['工龄']]
X2 = data['工龄']
Y = data['薪水']
# 2.二次曲线拟合数据
poly = PolynomialFeatures(degree=2)  # 设置最高次项为二次项
X11 = poly.fit_transform(X1)  # 将原有的X转换为一个新的二维数组X1，该二维数组包含新生成的二次项数据和原有的一次项数据。
model = LinearRegression()
model.fit(X11, Y)
r2 = r2_score(Y, model.predict(X11))
print("-----------------二次曲线-----------------")
print("R2:", round(r2, 3))
print("模型系数：", np.round(model.coef_, 3))
# 3.幂函数拟合数据
def power_func(x, a, b):
    return a * np.power(x, b)
popt, pcov = curve_fit(power_func, X2, Y)
r_squared = r2_score(Y, power_func(X2, *popt))
print("-----------------幂函数-------------------")
print('模型方程为：y = {:.3f} * x^({:.3f})'.format(*popt))
print('R2 = {:.3f}'.format(r_squared))
# 4.绘制模型拟合图
plt.rcParams['font.sans-serif'] = ['STSong']  # 设置显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 创建一个包含两个子图的图形
# 5.在第一个子图上绘制散点图和二次曲线拟合曲线
axs[0].plot(X1, model.predict(X11), color='red', label='二次曲线')  # 绘制二次曲线拟合曲线
axs[0].scatter(X1, Y, color='#F4B183', label='数据')  # 绘制散点图
axs[0].set_xlabel('工龄')  # 设置 x 轴标签
axs[0].set_ylabel('薪水')  # 设置 y 轴标签
axs[0].legend(prop={'size': 15})  # 添加图例，prop参数设置图例中字体大小
# 6.在第二个子图上绘制散点图和幂函数拟合曲线
axs[1].plot(X2, power_func(X2, *popt), 'r-', label='幂函数')  # 绘制幂函数拟合曲线
axs[1].scatter(X2, Y, label='数据')
axs[1].set_xlabel('工龄')
axs[1].set_ylabel('薪水')
axs[1].legend(prop={'size': 15})
plt.tight_layout()  # 调整子图布局
plt.show()  # 展示图像