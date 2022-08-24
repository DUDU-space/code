import ipykernel
import keras     # Keras是一个高层神经网络API
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score

# Sklearn (全称 Scikit-Learn) 是基于 Python 语言的机器学习工具。有六大任务模块：分别是分类、回归、聚类、降维、模型选择和预处理
word2vec_model1 = Word2Vec.load('word2vec.model')
vocab_list1 = [(k, word2vec_model1.wv[k]) for k, v in word2vec_model1.wv.vocab.items()]
embeddings_matrix1 = np.zeros((len(word2vec_model1.wv.vocab.items()) + 1, word2vec_model1.vector_size))
for i in range(len(vocab_list1)):
    embeddings_matrix1[i + 1] = vocab_list1[i][1]

embeddings_matrix1 = np.load("encode_now.txt.npy")
embeddings_matrix1 = np.insert(embeddings_matrix1, 0, np.zeros(64), axis=0)

word2vec_model2 = Word2Vec.load('word2vec-type.model')
vocab_list2 = [(k, word2vec_model2.wv[k]) for k, v in word2vec_model2.wv.vocab.items()]
embeddings_matrix2 = np.zeros((len(word2vec_model2.wv.vocab.items()) + 1, word2vec_model2.vector_size))
for i in range(len(vocab_list2)):
    embeddings_matrix2[i + 1] = vocab_list2[i][1]

df = pd.read_csv('w-X-train1.csv')
X_train1 = df.values
X_train1 = np.reshape(X_train1, (-1, 10, 30))
df = pd.read_csv('w-X-train2.csv')
X_train2 = df.values
X_train2 = np.reshape(X_train2, (-1, 10, 30))
df = pd.read_csv('w-X-test1.csv')
X_test1 = df.values
X_test1 = np.reshape(X_test1, (-1, 10, 30))
df = pd.read_csv('w-X-test2.csv')
X_test2 = df.values
X_test2 = np.reshape(X_test2, (-1, 10, 30))

df = pd.read_csv('w-y-train.csv')
y_train = df.values
df = pd.read_csv('w-y-test.csv')
y_test = df.values

# embedding_layer1 = keras.layers.Embedding(len(embeddings_matrix1),
#                                           embeddings_matrix1.shape[1],
#                                           embeddings_initializer=keras.initializers.Constant(embeddings_matrix1),
#                                           trainable=False)
# embedding_layer2 = keras.layers.Embedding(len(embeddings_matrix2),
#                                           embeddings_matrix2.shape[1],
#                                           embeddings_initializer=keras.initializers.Constant(embeddings_matrix2),
#                                           trainable=False)
#
# input1 = keras.layers.Input(shape=(10, 30,), dtype='float32')
# input2 = keras.layers.Input(shape=(10, 30,), dtype='float32')
# x1 = keras.layers.Flatten()(input1)
# x2 = keras.layers.Flatten()(input2)
# x1 = embedding_layer1(x1)
# x2 = embedding_layer2(x2)
# x = keras.layers.concatenate([x1, x2], axis=2)
# x = keras.layers.GRU(units=128)(x)
# x = keras.layers.Dropout(0.5)(x)
# x = keras.layers.BatchNormalization()(x)
# x = keras.layers.Dense(128, activation='relu')(x)
# x = keras.layers.Dropout(0.5)(x)
# x = keras.layers.BatchNormalization()(x)
# x = keras.layers.Dense(1, activation='sigmoid')(x)
#
# model = keras.Model([input1, input2], x)
# model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
#
# model.fit((X_train1, X_train2), y_train, epochs=20, batch_size=256, validation_data=((X_test1, X_test2), y_test))
# model.save("w2v-lstm.h5")

model = keras.models.load_model("w2v-lstm.h5")
y_pred = model.predict((X_test1, X_test2))
y_pred = np.where(y_pred > 0.5, 1, 0)
print(precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro'))
