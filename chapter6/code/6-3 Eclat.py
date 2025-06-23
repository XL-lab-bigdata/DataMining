import pandas as pd
from pyECLAT import ECLAT 
from mlxtend.frequent_patterns import association_rules
# 1.导入数据，并将其处理为特定格式
Data = pd.read_csv('../data/symptoms_of_diabetes_patients.csv')
Data.columns = range(len(Data.columns))
eclat_instance = ECLAT(data = Data, verbose=True) 
# 2.调用eclat_instance.fit生成频繁项集
get_ECLAT_indexes, get_ECLAT_supports = eclat_instance.fit(min_support=0.25, min_combination=1, max_combination=4, separator=' & ',verbose=True)
eclat_instance.df_bin # 数据展示
# 3.生成关联规则
items_list = [frozenset(element.split(' & ')) for element in list(get_ECLAT_supports.keys())]
frequentsets = pd.DataFrame(list(zip(list(get_ECLAT_supports.values()), items_list)), columns=['support', 'itemsets']) # 将频繁项集处理成特定格式
rules = association_rules(frequentsets, metric="confidence", min_threshold=0.8)
# 4.结果展示
print(frequentsets)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])