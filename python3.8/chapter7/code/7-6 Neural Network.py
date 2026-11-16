from keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import time
# 1.读入数据集
data = pd.read_csv('../data/gender.csv', encoding='ansi')  # 读入数据集
print(data['性别'].value_counts())  # 查看两类样本是否均衡
encoder = LabelEncoder()
data['性别'] = encoder.fit_transform(data['性别'])  # 对特征进行编码
# 2.划分训练集和测试集
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
# 3.编写测试不同隐藏层数量的预测性能的函数
def test_hidden_layer_num(num_layers):
    model = Sequential()
    model.add(Dense(units=32, input_dim=7, activation='relu'))  # 输入层
    for _ in range(num_layers):  # 添加指定数量的隐藏层
        model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # 输出层
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型
    start_time = time.time()  # 记录训练开始时间
    model.fit(X_train, y_train, epochs=100, verbose=0)  # 训练模型，指定epochs（训练轮数），以确保模型有足够的训练次数来学习数据的模式。
    end_time = time.time()  # 记录训练结束时间
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    return [accuracy_score(y_pred, y_test), 
            precision_score(y_pred, y_test), 
            recall_score(y_pred, y_test), 
            f1_score(y_pred, y_test), 
            end_time-start_time]
res_layer = []
for i in np.arange(1,41):
    res_layer.append(test_hidden_layer_num(i))
res_layer = pd.DataFrame(res_layer)
# 4.绘制神经网络模型的结果
plt.rcParams['font.sans-serif']=['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1,41), res_layer[0], label='准确率')
plt.plot(np.arange(1,41), res_layer[1], label='精准率')
plt.plot(np.arange(1,41), res_layer[2], label='召回率')
plt.plot(np.arange(1,41), res_layer[3], label='F1得分')
plt.legend(ncol=1, frameon=False,loc='lower left')
plt.xlabel('隐藏层数量')
plt.ylabel('评价指标')
plt.subplot(1, 2, 2)
plt.plot(np.arange(1,41), res_layer[4], label='accuracy')
plt.xlabel('隐藏层数量')
plt.ylabel('运行时间（秒）')
plt.tight_layout()
plt.show()