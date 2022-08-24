import random
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('data1.csv')     # 处理data1.csv中的数据，分为y:类型（0，1） x:[2:]列的值     训练集，测试集
data = list(df.values)  # 取值
mac_set = set()
mac_map = dict()
data_map = dict()

for i in data:
    mac_set.add(i[1])  # 集合里增加值
    if i[0] == -1:
        type_ = 0
    else:
        type_ = 1
    mac_map[i[1]] = type_    # mac_map里放 值：类型 的字典
    try:
        data_map[i[1]]             # data_map里的 值：类型 类型为1
    except:
        data_map[i[1]] = list()       # 将类型为1的键:i[1] 值:列表

    data_map[i[1]].append(i)      # data_map里键：i[1] 值:data

tmp = []   # 列表
for k, v in data_map.items():     # 返回字典的（值，类型）对列表的副本
    if len(v) < 10:     # data长度小于10，将i[1]放入列表
        tmp.append(k)
for j in tmp:
    data_map.pop(j)      # 移除i[1]   data长度小于10,数据预处理
    mac_set.remove(j)

train_mac = []
test_mac = []

for i in mac_set:
    if random.randint(1, 4) == 1:
        test_mac.append(i)
    else:
        train_mac.append(i)           # 随机将mac_set分为训练集和测试集

X_train = []
y_train = []
X_test = []
y_test = []

for i in tqdm(train_mac):
    for _ in range(100):
        track = []
        for _ in range(10):
            track.extend(random.choice(data_map[i])[2:])   # 随机抽取数据
        X_train.append(track)     # data_map[i])[2:]
        y_train.append(mac_map[i])    # mac_map里放 值：类型 的字典  类型

for i in tqdm(test_mac):
    for _ in range(100):
        track = []
        for _ in range(10):
            track.extend(random.choice(data_map[i])[2:])
        X_test.append(track)
        y_test.append(mac_map[i])

df = pd.DataFrame(np.array(X_train))
df.to_csv('m-X-train.csv', index=False)    # 存储时不加索引
df = pd.DataFrame(np.array(y_train))
df.to_csv('m-y-train.csv', index=False)

df = pd.DataFrame(np.array(X_test))
df.to_csv('m-X-test.csv', index=False)
df = pd.DataFrame(np.array(y_test))
df.to_csv('m-y-test.csv', index=False)
