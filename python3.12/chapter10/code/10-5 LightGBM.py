import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 1.数据预处理
df = pd.read_csv('../data/stroke.csv')
x = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0) # 训练集测试集切分
ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['gender', 'work_type', 'smoking_status']),
    ('numeric', 'passthrough', ['age', 'hypertension', 'heart_disease']),
    ('binary', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
     ['ever_married', 'Residence_type']),
], remainder='passthrough')

# 2.模型构建和训练
# 将编码与模型放在同一流水线中，每一折只用训练数据拟合编码器
model = Pipeline([('preprocess', ct), ('classifier', LGBMClassifier(verbosity=-1))])
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("LGBM模型准确率：",accuracy_score(y_test,y_pred))

# 3.网格搜索交叉验证参数---learning_rate参数
parameters = {'classifier__learning_rate': [0.01, 0.05, 0.1]}
grid = GridSearchCV(model,
                    scoring="accuracy",
                    param_grid=parameters,
                    cv=10)
grid.fit(X_train,y_train)
print('参数learning_rate的最佳取值:{0}'.format(grid.best_params_))
print('LGBM最佳模型得分:{0}'.format(grid.best_score_))
print(grid.cv_results_['mean_test_score'])
print(grid.cv_results_['params'])

# 4.网格搜索交叉验证参数---feature_fraction参数
parameters = {'classifier__feature_fraction': [0.6, 0.8, 1],}
grid = GridSearchCV(model,
                    param_grid=parameters,
                    scoring="accuracy",
                    cv=10)
grid.fit(X_train,y_train)
print('参数feature_fraction的最佳取值:{0}'.format(grid.best_params_))
print('LGBM最佳模型得分:{0}'.format(grid.best_score_))
print(grid.cv_results_['mean_test_score'])
print(grid.cv_results_['params'])
