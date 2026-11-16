from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import pandas as pd
# 1.导入数据
data = pd.read_parquet('../data/user_orders_hourofday.parquet')
# 2.调用fpgrowth函数, 生成频繁项集与关联规则
frequent_itemsets = fpgrowth(data, min_support=0.03, use_colnames= True)  # 生成频繁项集
rules = association_rules(frequent_itemsets,metric = 'confidence', min_threshold = 0.35)  # 生成关联规则
rules = rules[rules['lift']>=1.2]  #设置提升度阈值，筛选关联规则
# 3.结果展示
print(frequent_itemsets)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])