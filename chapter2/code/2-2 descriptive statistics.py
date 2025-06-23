import pandas as pd
import numpy as np

pd.set_option('display.width', 300) # 设置字符显示宽度
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
df = pd.read_csv('../data/skincare.csv')  # 读取数据文件
# 1.数据分组
grouped_data = df.groupby('类型')['价格']  # 按Label分组，统计Price列数据
# 2.计算集中趋势度量：平均值、中位数、众数
Central_Tendency = grouped_data.agg(['mean', 'median'])  # 对grouped_data进行聚合操作，计算每组的平均值和中位数
Central_Tendency.columns = ['平均值', '中位数']  # 将结果表的列名重命名为中文的“平均值”和“中位数”
mode = grouped_data.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)  # 使用lambda函数计算每组的众数，如果没有众数则返回NaN
Central_Tendency['众数'] = mode  # 将众数的结果添加为新的列“众数”
print('集中趋势度量结果：')
print(round(Central_Tendency,2))
# 2.计算离散程度度量：极差、平均差、方差、标准差、异众比率、四分位差、变异系数
range = grouped_data.apply(np.ptp)  # 极差
mean_diff = grouped_data.apply(lambda x: np.mean(np.abs(x - x.mean())))  # 平均差
var = grouped_data.apply(np.var)  # 方差
std = grouped_data.apply(np.std)  # 标准差
z_ratio = grouped_data.apply(lambda x: (x.value_counts().max() / len(x)))  # 异众比率
quartile_deviation = grouped_data.apply(lambda x: np.percentile(x, 75)-np.percentile(x, 25))  # 四分位差
coef_of_variation = grouped_data.apply(lambda x: np.std(x) / np.mean(x))  # 变异系数
Dispersion = pd.DataFrame({'极差': range,
               '平均差': mean_diff,
               '方差':var,
               '标准差':std,
               '异众比率': z_ratio,
               '四分位差': quartile_deviation,
               '变异系数': coef_of_variation})
print('离散程度度量结果：')
print(round(Dispersion,2))
# 3.计算分布形态度量：偏度、峰度
Distribution = grouped_data.agg([pd.Series.skew, pd.Series.kurtosis])
Distribution.columns = ['偏度', '峰度']
print('分布形态度量结果：')
print(round(Distribution,2))

