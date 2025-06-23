import pandas as pd
from mlxtend.frequent_patterns import apriori  # 生成频繁项集
from mlxtend.frequent_patterns import association_rules  # 生成强关联规则
# 1.导入数据
dataSet = pd.read_csv('../data/groceries.csv')
# 2.调用apriori函数，生成频繁项集与关联规则
if __name__ == '__main__':    
    frequent_itemsets = apriori(dataSet, min_support=0.02, use_colnames=True)  # apriori算法生成频繁项集  
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)  # 使用频繁项集生成强关联规则    
# 3.结果展示
print(frequent_itemsets)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])