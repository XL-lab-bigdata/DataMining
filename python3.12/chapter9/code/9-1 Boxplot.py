from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 1.绘图参数设置
plt.rcParams['font.sans-serif'] = ['STSong']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
font1 = {'family': 'STSong', 'weight': 'normal', 'size': 20} # 字体定义
# 2.读取数据集
traffic_path = Path('../data/Web network traffic.parquet')
if traffic_path.is_file():
    all_data = pd.read_parquet(traffic_path)
else:
    all_data = pd.read_csv(traffic_path.with_suffix('.csv'))

traffic_column = all_data.columns[-1]
complete_rows = all_data.notna().all(axis=1)  # 删除任一天缺少数据的页面
all_data = all_data.loc[complete_rows, ['Page', traffic_column]].copy()
project = all_data['Page'].str.rsplit('_', n=3).str[1]
all_data['Language'] = project.str.split('.').str[0]
language_list = ['en', 'ja', 'de', 'fr', 'zh', 'ru', 'es']
all_data = all_data[all_data['Language'].isin(language_list)]  # 提取七种语言的数据
plot_data = []
for language in language_list:
    language_traffic = all_data.loc[all_data['Language'] == language, traffic_column]
    plot_data.append(language_traffic[language_traffic < 500].head(80).tolist())  # 取前 80 个低于 500 的值
# 3.构建箱线图
bplot = plt.boxplot(plot_data, notch=True, patch_artist=True) # 绘图
# 4.可视化
colors = ['lightpink', 'lightblue', 'darkseagreen', 'wheat', 'thistle', 'rosybrown', 'lightgray']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.grid(axis='y')
plt.xticks(range(1, 8), labels=['英语', '日语', '德语', '法语', '中文', '俄语', '西班牙语'],fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('语  言', fontdict=font1)
plt.ylabel('网页浏览次数', fontdict=font1)
plt.tight_layout()
plt.show()

