import random
# import keras
import pandas as pd
import numpy as np
import xgboost as xgb   # 极端梯度提升，其基本思想为：一棵树一棵树逐渐地往模型里面加，每加一棵CRAT决策树时，要使得整体的效果
# 有所提升。使用多棵决策树（多个单一的弱分类器）构成组合分类器，并且给每个叶子节点赋与一定的权值。
import lightgbm as lgb   # LightGBM是个快速的、分布式的、高性能的基于决策树算法的梯度提升框架。可用于排序、分类、回归以及很多其他的机器学习任务中。
# from keras.callbacks import EarlyStopping
# from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('m-X-train.csv')
X_train = df.values
df = pd.read_csv('m-X-test.csv')
X_test = df.values

df = pd.read_csv('m-y-train.csv')
y_train = df.values[:, 0]
df = pd.read_csv('m-y-test.csv')
y_test = df.values[:, 0]

scaler = StandardScaler()   # 先通过计算训练集中特征的均值、标准差，对每个特征进行独立居中和缩放
scaler.fit(X_train)   # 计算均值和标准差，用于以后的缩放
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)    # 将数据的分布转为正态分布

# ocsvm
# idx = np.argwhere(y_train == 0)[:, 0]
# ocsvm_model = OneClassSVM()
# ocsvm_model.fit(X_train[idx])
# y_pred = ocsvm_model.predict(X_train)
# y_pred = np.where(y_pred == 1, 0, 1)
# print('ocsvm', precision_score(y_train, y_pred), recall_score(y_train, y_pred), f1_score(y_train, y_pred),
#       f1_score(y_train, y_pred, average='micro'), f1_score(y_train, y_pred, average='macro'))

# iforest
# iforest_model = IsolationForest(contamination=0.04)
# iforest_model.fit(X_train)
# y_pred = iforest_model.predict(X_train)
# y_pred = np.where(y_pred == 1, 0, 1)
# print('iforest', precision_score(y_train, y_pred), recall_score(y_train, y_pred), f1_score(y_train, y_pred),
#       f1_score(y_train, y_pred, average='micro'), f1_score(y_train, y_pred, average='macro'))

# lof
# lof_model = LocalOutlierFactor(contamination=0.04)
# y_pred = lof_model.fit_predict(X_train)
# y_pred = np.where(y_pred == 1, 0, 1)
# print('lof', precision_score(y_train, y_pred), recall_score(y_train, y_pred), f1_score(y_train, y_pred),
#       f1_score(y_train, y_pred, average='micro'), f1_score(y_train, y_pred, average='macro'))

# lgb
lgb_model = lgb.LGBMClassifier()  # GBM 梯度提升机
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
print('lgb', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# xgb
xgb_model = xgb.XGBClassifier()     # 分类器
xgb_model.fit(X_train, y_train)     # 模型 训练
y_pred = xgb_model.predict(X_test)   # 预测值
print('xgb', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# precision_score精准率（实际为正样本占被预测为正的样本的比例）
# recall_score召回率（正例样本中被预测为正例的比例）
# f1_score（两倍精确率乘以召回率/精确率加召回率）   Macro Average宏平均是指在计算均值时使每个类别具有相同的权重
# Micro Average微平均是指赋予所有类别的每个样本相同的权重

# tree
tree_model = DecisionTreeClassifier()   # 决策树
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print('tree', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# knn
knn_model = KNeighborsClassifier()   # K近邻算法（k-近邻算法采用测量不同特征值之间的距离方法进行分类。）
# 输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。
# 最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print('knn', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# svm
svm_model = SVC()   # 支持向量分类
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print('svm', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# NB
nb_model = GaussianNB()   # 高斯朴素贝叶斯（找出特征输出Y和特征X的联合分布P(X,Y),然后用P(Y|X)=P(X,Y)/P(X)得出）
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print('NB', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# LR
lr_model = LogisticRegression()   # 逻辑回归（sigmod）
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print('LR', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# RF
rf_model = RandomForestClassifier()  # 随机森林分类器(随机森林集成了所有的分类投票结果，将投票次数最多的类别指定为最终的输出，
# 这就是一种最简单的 Bagging 思想)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print('RF', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))

# DL
# dl_model = keras.Sequential()
# dl_model.add(keras.layers.Reshape((10, -1), input_shape=(100,)))
# dl_model.add(keras.layers.LSTM(128))
# dl_model.add(keras.layers.Dropout(0.5))
# dl_model.add(keras.layers.Dense(1, activation='sigmoid'))
# dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# dl_model.summary()
# dl_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[
#     EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
# ])
# y_pred = dl_model.predict(X_test)
# y_pred = np.where(y_pred > 0.5, 1, 0)
# print('DL', precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
#       f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))
