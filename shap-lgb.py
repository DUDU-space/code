import random
import shap
import keras
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from gensim.models import Word2Vec
from keras.preprocessing import text
import matplotlib.pyplot as plt

df = pd.read_csv('data3.csv')
data = df.values
X_train = []
y_train = []
for i in data:
    X_train.append(i[2])
    y_train.append(0 if i[0] == -1 else 1)

X_train = np.array(X_train)
y_train = np.array(y_train)


class TextPreprocessor(object):
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None

    def create_tokenizer(self, text_list):
        tokenizer = text.Tokenizer(num_words=self._vocab_size)     # 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示
        tokenizer.fit_on_texts(text_list)   # 用以训练的文本列表
        self._tokenizer = tokenizer

    def transform_text(self, text_list):
        text_matrix = self._tokenizer.texts_to_matrix(text_list)   # 待向量化的文本列表
        return text_matrix


VOCAB_SIZE = 17
processor = TextPreprocessor(VOCAB_SIZE)
processor.create_tokenizer(X_train)
X_train = processor.transform_text(X_train)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

attrib_data = np.array(random.choices(X_train, k=2000))
explainer = shap.TreeExplainer(model, attrib_data)   # shap “模型解释”包，反映出每一个样本中的特征的影响力
num_explanations = 200
shap_vals = explainer.shap_values(np.array(random.choices(X_train, k=num_explanations)))  # 传入特征矩阵X，以计算SHAP值

words = processor._tokenizer.word_index   # 保存所有word对应的编号id
word_lookup = list()
for i in words.keys():
    word_lookup.append(i)
word_lookup = [''] + word_lookup
print(word_lookup)

word_lookup = [
    '',
    'Road side',
    'Hotel',
    'Home community',
    'School',
    'Government',
    'Hospital',
    'Shopping mall',
    'Police station',
    'Super market',
    'Labor market',
    'Bank',
    'Internet cafe',
    'Restaurant',
    'Cinema',
    'Gas station',
    'Scenic spot'
]

plt.rcParams.update({'font.size': 100})   # 自定义图形的各种默认属性
shap.summary_plot(shap_vals, feature_names=word_lookup, class_names=["resident", "suspect"])
# 每一个样本绘制其每个特征的SHAP值，可以提供直观的理解整体模式，并允许发现预测异常值