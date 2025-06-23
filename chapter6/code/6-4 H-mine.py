import pandas as pd
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import hmine
from mlxtend.frequent_patterns import association_rules
# 1.数据导入
data = pd.read_parquet('../data/kosarak.parquet')
retrieved_data = [ast.literal_eval(row) for row in data['data']]
# 2.数据热编码，并使用稀疏表示
te = TransactionEncoder()
oht_ary = te.fit(retrieved_data).transform(retrieved_data, sparse=True)
sparse_df = pd.DataFrame.sparse.from_spmatrix(oht_ary, columns=te.columns_)
# 3.调用hmine函数，生成频繁项集和关联规则
frequent_pattern = hmine(sparse_df, min_support=0.01, use_colnames=True, verbose=0)
rules = association_rules(frequent_pattern ,metric = 'confidence',min_threshold = 0.85)
rules = rules[rules['lift'] >= 1]
# 4.结果展示
print(frequent_pattern)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])