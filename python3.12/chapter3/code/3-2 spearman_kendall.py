import pandas as pd
from scipy.stats import kendalltau, spearmanr
df = pd.read_csv('../data/amazon_reviews.csv')  # 读取数据集
df['ordinal_rating'] = df['star_rating'].apply(lambda x: 1 if x <= 2 else 2 if x == 3 else 3)  # 将评分转换为数据
# 1.计算Kendl相关系数和p值
kendall_corr, kendall_pval = kendalltau(df['ordinal_rating'], df['helpful_votes'])
# 2.计算Speaan相关系数和p值
spearman_corr, spearman_pval = spearmanr(df['ordinal_rating'], df['helpful_votes'])
print('Kendall相关系数:', kendall_corr)
print('Spearman相关系数:', spearman_corr)