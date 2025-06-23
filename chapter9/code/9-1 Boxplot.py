import matplotlib.pyplot as plt
import pandas as pd

# 1.绘图参数设置
plt.rcParams['font.sans-serif'] = ['STSong']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
font1 = {'family': 'STSong', 'weight': 'normal', 'size': 20} # 字体定义
# 2.读取数据集
all_data = pd.read_parquet('../data/Web network traffic.parquet').dropna(axis=0, how='any') # 删除缺省值
all_data['Country'] = all_data['Page'].apply(lambda x: x.split('_')[1].split('.')[0]) #读取网页流量数据
country_list = ['en', 'ja', 'de', 'fr', 'zh', 'ru', 'es']
all_data = all_data[all_data['Country'].isin(country_list)] # 提取country_list中七种语言的数据
plot_data = [[] for _ in range(len(country_list))]
for c, country in enumerate(country_list):
    plot_data[c] = [all_data.iloc[i, -2] for i in range(len(all_data)) if country == all_data.iloc[i, -1]] # 将七种语言的数据分别存储到plot_data列表中的每个子列表中
plot_data = [[data for data in plot_data[i] if data < 500][:80] for i in range(len(plot_data))] # 为了画图可读性，去除数值太高的数据
# 3.构建箱线图
bplot = plt.boxplot(plot_data, notch=True, vert=True, patch_artist=True) # 绘图
# 4.可视化
colors = ['lightpink', 'lightblue', 'darkseagreen', 'wheat', 'thistle', 'rosybrown', 'lightgray']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.grid(axis='y')
plt.xticks(range(1, 8), labels=['英语', '日语', '德语', '法语', '中文', '俄语', '葡萄牙语'],fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('语  言', fontdict=font1)
plt.ylabel('网络流量/字 节', fontdict=font1)
plt.tight_layout()
plt.show()

