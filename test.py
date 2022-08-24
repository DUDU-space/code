import pandas as pd
import json
import random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

# embeddings = np.load("encode_now.txt.npy")
# print(embeddings)
# pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', 50)
#
# data = pd.read_csv('data7.csv')
# data1 = data.groupby(by='0').apply(lambda x:[','.join(x['1'])])
# print(data1.head())
# data1.to_csv('data8.csv')

# df = pd.read_csv('data.csv')
# # data = list(df.values)
# for ddf in df:
#
#     print(type(ddf[0]))

# d = pd.DataFrame(np.array(data))
# d.to_csv('data7.csv', index=False)
#
df = pd.read_csv('data.csv', dtype={'3': np.float64,'4': np.float64,})
a = np.array(df)
point = []
for i in range(len(a)):
    x = []
    lon = float(a[i][3])
    x.append(lon)
    lat = float(a[i][4])
    x.append(lat)
    point.append(x)
print(point)


# d = pd.DataFrame(np.array(a))
# d.to_csv('data8.csv', index=False)

# df["1"] = df.iloc[:,3:5].values.tolist()
#
# d = pd.DataFrame(np.array(df))
# d.to_csv('data7.csv', index=False)

#     train_data.append(
#         InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     index += 1
# return train_data
