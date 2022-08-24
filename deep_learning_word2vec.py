import pandas as pd
from gensim.models.word2vec import Word2Vec

# df = pd.read_csv('data3.csv')    # type mac ap_mac
# data = list(df.values)
#
# b = []
# for i in range(len(data)):
#     a = data[i][2].split(" ")
#     b.append(a)
#
# model = Word2Vec(min_count=5, vector_size=64)
# model.build_vocab(b)   # 遍历一次语料库建立词典
# model.train(b, total_examples=model.corpus_count, epochs=model.epochs)    # 遍历语料库建立神经网络模型
# model.save("word2vec.model")

df = pd.read_csv('data3.csv')    # type mac ap_mac
data = list(df.values)

b = list()
for i in range(len(data)):
    a = data[i][2].split(" ")
    b.append(a)

print(b)

# model = Word2Vec(vector_size=64)
# model.build_vocab(b)   # 遍历一次语料库建立词典
# model.train(b)    # 遍历语料库建立神经网络模型
# model.save("2.model")
