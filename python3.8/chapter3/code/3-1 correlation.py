import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('../data/livestreaming.csv', sep=',')
df1 = data.iloc[-24:, :]
# 1.创建一个画布和两个子图
plt.rcParams.update({'font.size': 20, 'font.sans-serif': ['STSong'], 'axes.unicode_minus': False})
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
# 2.散点图
axs[0].scatter(data['streamers_count'], data['viewers_count'], color='#F4B183')
axs[0].set(xlabel='主播数量', ylabel='观众数量', title='观众数量和主播数量散点图')
axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# 3.折线图
df1['time'] = pd.to_datetime(df1['time'])
df1['hour'] = df1['time'].dt.hour
axs[1].plot(df1['hour'], df1['streamers_count'], label='主播数量')  # 第一条曲线
axs[1].set(xlabel='时间',ylabel='数量', title='一天内流量变化图')
axs[1].tick_params(axis='both', which='both', labelsize=20)
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2 = axs[1].twinx()  # 第二条曲线
ax2.plot(df1['hour'], df1['viewers_count'], color='#EC5D3B', label='观众数量')
ax2.set_ylabel('观众数量')
ax2.tick_params(axis='both', which='both', labelsize=20)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axs[1].legend(loc='upper left', fontsize=16)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.89), fontsize=16)
plt.xticks([0, 3, 7, 11, 15, 19, 23])  # 设置x轴标签格式
plt.subplots_adjust(wspace=0.2)  # 调整子图之间的间距
plt.show()
# 4.计算皮尔逊相关系数
pearson_corr = df1['streamers_count'].corr(df1['viewers_count'])