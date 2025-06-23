import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from minepy import MINE
from sklearn.feature_selection import mutual_info_regression
# 1.加载数据集
vibration_data = pd.read_csv('../data/bridge.csv')
# 2.确保时间戳被正确解析为datetime对象
vibration_data['timestamp'] = pd.to_datetime(vibration_data['timestamp'])
# 3.可视化振动幅度与时间的关系
plt.figure(figsize=(14, 6))
plt.rcParams['font.sans-serif'] = 'SimSun'  # 设置字体为宋体
plt.scatter(vibration_data['timestamp'], vibration_data['amplitude'], alpha=0.6, s=10)
plt.title('桥梁振动幅度与时间关系图', fontsize=20)
plt.xlabel('时间', fontsize=20)
plt.ylabel('震动幅度', fontsize=20)
plt.grid(False)  # 去掉网格
plt.tick_params(axis='both', which='major', labelsize=16)  # 设置刻度标签字体大小
plt.show()
# 4.计算相关系数
t_start = vibration_data['timestamp'].min()  # 获取数据集中的最早时间
time_in_seconds = (vibration_data['timestamp'] - t_start).dt.total_seconds().values
mutual_info = mutual_info_regression(np.expand_dims(time_in_seconds, axis=1), vibration_data['amplitude'].values)[0]  # 计算互信息
mine = MINE(alpha=0.6, c=15)
mine.compute_score(time_in_seconds, vibration_data['amplitude'].values)
mic_value = mine.mic()  # 计算MIC
pearson_corr, pearson_p_value = pearsonr(time_in_seconds, vibration_data['amplitude'].values)  # Pearson相关
spearman_corr, spearman_p_value = spearmanr(time_in_seconds, vibration_data['amplitude'].values)  # Spearman相关
kendall_corr, kendall_p_value = kendalltau(time_in_seconds, vibration_data['amplitude'].values)  # Kendall相关
print("互信息:", mutual_info)
print("MIC值:", mic_value)
print("Pearson相关系数:", pearson_corr)
print("Spearman相关系数:", spearman_corr)
print("Kendall相关系数:", kendall_corr)
