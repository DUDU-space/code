import json
import random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

with open("type_map.json") as f:    # 地点，地点类型
    point_map = json.loads(f.read())    # 反序列化方法,将json格式数据解码为python对象。

word2vec_model1 = Word2Vec.load('word2vec.model')
word2idx1 = dict()    # 新建一个空字典

vocab_list1 = list()
a = word2vec_model1.wv.key_to_index
for i in a:
    b = [i, a[i]]
    vocab_list1.append(b)

embeddings_matrix1 = np.zeros((len(word2vec_model1.wv.index_to_key) + 1, word2vec_model1.vector_size))
for i in range(len(vocab_list1)):
    word = vocab_list1[i][0]
    word2idx1[word] = i + 1
    embeddings_matrix1[i + 1] = vocab_list1[i][1]

word2vec_model2 = Word2Vec.load('word2vec-type.model')
word2idx2 = dict()

vocab_list2 = list()
c = word2vec_model2.wv.key_to_index
for i in c:
    d = [i, c[i]]
    vocab_list2.append(d)
embeddings_matrix2 = np.zeros((len(word2vec_model2.wv.index_to_key) + 1, word2vec_model2.vector_size))
for i in range(len(vocab_list2)):
    word = vocab_list2[i][0]
    word2idx2[word] = i + 1
    embeddings_matrix2[i + 1] = vocab_list2[i][1]

df = pd.read_csv('data3#.csv')
data = list(df.values)        # 创建列表
mac_set = set()
mac_map = dict()
data_map = dict()

for i in data:
    mac_set.add(i[0])     # type mac 地点序列
    if i[1] == -1:
        type_ = 0
    else:
        type_ = 1
    mac_map[i[0]] = type_   # mac_map：mac type
    try:
        data_map[i[0]]
    except:
        data_map[i[0]] = list()

    data_map[i[0]].append(i)     # data_map:mac i

tmp = []
for k, v in data_map.items():
    if len(v) < 10:
        tmp.append(k)
for j in tmp:
    data_map.pop(j)
    mac_set.remove(j)

train_mac = []
test_mac = []

for i in mac_set:
    if random.randint(1, 4) == 1:
        test_mac.append(i)
    else:
        train_mac.append(i)

X_train1 = []
X_train2 = []
y_train = []
X_test1 = []
X_test2 = []
y_test = []

for i in tqdm(train_mac):
    for _ in range(100):
        track = []
        for _ in range(10):
            track.extend(random.choice(data_map[i])[2:])
        X_train1.append(track)
        X_train2.append(track)
        y_train.append(mac_map[i])

for i in tqdm(test_mac):
    for _ in range(100):
        track = []
        for _ in range(10):
            track.extend(random.choice(data_map[i])[2:])   # track：地点序列
        X_test1.append(track)
        X_test2.append(track)
        y_test.append(mac_map[i])

for i in tqdm(range(len(X_train1))):
    X_train1[i] = " ".join(X_train1[i])
    X_train1[i] = X_train1[i].split(' ')
    for _ in range(500):
        X_train1[i].append('')
    X_train1[i] = X_train1[i][:300]
    for j in range(len(X_train1[i])):
        try:
            X_train1[i][j] = word2idx1[X_train1[i][j]]
        except:
            X_train1[i][j] = 0
    X_train1[i] = np.array(X_train1[i])

for i in tqdm(range(len(X_train2))):
    X_train2[i] = " ".join(X_train2[i])
    X_train2[i] = X_train2[i].split(' ')
    for _ in range(500):
        X_train2[i].append('')
    X_train2[i] = X_train2[i][:300]
    for j in range(len(X_train2[i])):
        try:
            X_train2[i][j] = word2idx2[point_map[X_train2[i][j]]]
        except:
            X_train2[i][j] = 0
    X_train2[i] = np.array(X_train2[i])

for i in tqdm(range(len(X_test1))):
    X_test1[i] = " ".join(X_test1[i])
    X_test1[i] = X_test1[i].split(' ')
    for _ in range(500):
        X_test1[i].append('')
    X_test1[i] = X_test1[i][:300]
    for j in range(len(X_test1[i])):
        try:
            X_test1[i][j] = word2idx1[X_test1[i][j]]
        except:
            X_test1[i][j] = 0
    X_test1[i] = np.array(X_test1[i])

for i in tqdm(range(len(X_test2))):
    X_test2[i] = " ".join(X_test2[i])
    X_test2[i] = X_test2[i].split(' ')
    for _ in range(500):
        X_test2[i].append('')
    X_test2[i] = X_test2[i][:300]
    for j in range(len(X_test2[i])):
        try:
            X_test2[i][j] = word2idx2[point_map[X_test2[i][j]]]
        except:
            X_test2[i][j] = 0
    X_test2[i] = np.array(X_test2[i])

df = pd.DataFrame(np.array(X_train1))
df.to_csv('w-X-train1.csv', index=False)
df = pd.DataFrame(np.array(X_train2))
df.to_csv('w-X-train2.csv', index=False)
df = pd.DataFrame(np.array(y_train))
df.to_csv('w-y-train.csv', index=False)

df = pd.DataFrame(np.array(X_test1))
df.to_csv('w-X-test1.csv', index=False)
df = pd.DataFrame(np.array(X_test2))
df.to_csv('w-X-test2.csv', index=False)
df = pd.DataFrame(np.array(y_test))
df.to_csv('w-y-test.csv', index=False)
