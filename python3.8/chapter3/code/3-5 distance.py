import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
# 1.导入数据
data = pd.read_csv('../data/ratings.csv')
# 2.根据userId、movieId、rating三列得到评分行
pivot_df = data.pivot(index='userId', columns='movieId', values='rating')
# 3.填充NaN值为0
pivot_df_subset = pivot_df.fillna(0)
print(pivot_df_subset)
# 4.创建新的DataFrame用于存储矩阵
matrix_df_a = pd.DataFrame(index=pivot_df_subset.index, columns=pivot_df_subset.index)
matrix_df_b = pd.DataFrame(index=pivot_df_subset.index, columns=pivot_df_subset.index)
# 5.填充欧式距离矩阵
for i in pivot_df_subset.index:
    for j in pivot_df_subset.index:
        # 计算用户i和用户j之间的欧式距离
        distance = euclidean(pivot_df_subset.loc[i], pivot_df_subset.loc[j])
        matrix_df_a.loc[i, j] = distance
print(matrix_df_a)
# 6.填充余弦相似度矩阵
for i in pivot_df_subset.index:
    for j in pivot_df_subset.index:
        # 计算用户i和用户j之间的余弦相似度
        similarity = 1 - cosine(pivot_df_subset.loc[i], pivot_df_subset.loc[j])
        matrix_df_b.loc[i, j] = similarity
print(matrix_df_b)
# 7.保存结果
matrix_df_a.to_csv('距离相关分析-欧氏距离.csv')
matrix_df_b.to_csv('距离相关分析-余弦相似度.csv')