import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import collections

# 1.读取数据
df = pd.read_csv('../data/Diamonds.csv', encoding="gb2312")  # 读取文件
plt.rcParams['font.sans-serif'] = 'SimSun'  # 设置字体为宋体
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 8))  # 设置图形的宽和高
# 2.绘制条形图
plt.subplot(2, 3, 1)
color_list = collections.Counter(df['颜色'])  # 使用collections模块的Counter函数统计每个字母的数量
colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J']  # 指定颜色列表的顺序创建字母和对应数量的列表
counts = [color_list[color] for color in colors]  # 按照colors列表中的顺序依次获取各颜色的数量，存储在counts中
plt.bar(colors, counts, facecolor='#F7C4C1', edgecolor='k', alpha=0.8)  # 利用plt.bar()函数绘制条形图
plt.xlabel('颜色')
plt.ylabel('数量')
plt.title('条形图（颜色）')
# 3.绘制饼图
plt.subplot(2, 3, 2)
clarity = df['清晰度'].value_counts()  # 统计各清晰度等级的钻石数量
plt.pie(clarity.values, labels=clarity.index, colors=['#FCDFBE', '#F3DAC0', '#F7C4C1', '#E3C6E0', '#CECCE5', '#C3E2EC', '#BCD1BC', '#DBEDC5'])  # 利用plt.pie()函数绘制饼图
plt.title('饼图（清晰度）')
# 4.绘制箱线图
plt.subplot(2, 3, 3)
plt.grid(True)  # 显示网格
plt.boxplot(df['长'],
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markeredgecolor": "k", "markersize": 5})
plt.title('箱线图（长）')
plt.ylabel('长')
# 5.绘制直方图
plt.subplot(2, 3, 4)
plt.hist(df['价格'], bins=10, facecolor='#BCD1BC', edgecolor='k',alpha=0.8)  # 利用plt.hist()函数绘制直方图
plt.title("直方图（价格）")
plt.xlabel("价格")
plt.ylabel("频数")
# 6.绘制折线图
plt.subplot(2, 3, 5)
counts = df['切割质量'].value_counts()  # 统计各切割质量等级的钻石数量
plt.plot(counts, alpha=0.8, marker='*', color='#DB7093')  # 利用plt.plot()函数绘制折线图
plt.title('折线图（切割质量）')
plt.xlabel('切割质量')
plt.ylabel('数量')
# 7.绘制散点图
df_data = df[df['切割质量'] == '好']  # 提取切割质量为“好“的钻石数据
plt.subplot(2, 3, 6)
plt.scatter(df_data['克拉'], df_data['价格'], color='#E3C6E0', alpha=0.5,  s=20)  # 利用plt.scatter()函数绘制散点图
plt.title('散点图（克拉与价格）')
plt.xlabel('克拉')
plt.ylabel('价格')
plt.tight_layout()  # 调整子图位置
plt.show()

