import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# 读入数据
data = pd.read_parquet('../data/transaction.parquet')

# 查看各列的缺失值
missing_values = data.isnull().sum()
print(missing_values)

# 删除缺失“商品描述”字段的数据记录
data_processed = data.dropna(subset=['商品描述'])

# 查看处理后数据的数值型数据分布情况
print(data_processed.describe())

# 筛选数据
data_processed = data_processed[data_processed['数量'] >= 1]
data_processed = data_processed[data_processed['单价'] > 0]

# 查看筛选后数据的数值型数据分布情况
print(data_processed.describe())

# 重置索引
data_processed = data_processed.reset_index(drop=True)

# 设置分层抽样的列名
stratum_column = '国家'

# 设置要抽取的样本数量
sample_size = 10000

# 创建分层抽样对象
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)

# 使用分层抽样对象进行分层抽样
for train_index, test_index in stratified_splitter.split(data_processed, data_processed[stratum_column]):
    stratified_sample = data_processed.iloc[test_index]

# 打印分层抽样后的数据样本
print(stratified_sample)

# 对分层抽样后的数据进行独热编码
encoded_data = pd.get_dummies(stratified_sample, columns=['国家'])
print(encoded_data)
