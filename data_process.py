# coding:utf-8
import json
import pandas as pd
import numpy as np
from geohash import encode

# # 打开json文件
# with open("point_map.json") as f:    # 地点，地点类型  wx4eqyv
#     point_map = json.loads(f.read())
#
# print(point_map)
#
#
# # geohash 编码
# a = encode(39984094000116319236)
# b = encode(39984198000116319322)
# c = encode(39984674000116319810)
#
# print(a,b,c)
#
# # 处理坐标数据，将坐标连在一起，形成唯一的地点标识
# df = pd.read_csv('data.csv',encoding='UTF-8')
# data = list(df.values)
#
# for i in range(len(data)):
#
#     a = float('% .6f' % (data[i][3])) * 10 ** 6
#     b = float('% .6f' % (data[i][4])) * 10 ** 6
#     c = int(a)
#     d = int(b)
#     e = str(c) + str(d)
#     data[i][3] = str(e)
#
# d = pd.DataFrame(np.array(data))
# d.to_csv('data2.csv', index=False)


# # 地点—地点类型映射
# point_map = dict()
# df = pd.read_csv('data2.csv')
# data = np.array(df.values)
#
# for i in range(len(data)):
#
#     point_map[data[i][2]] = data[i][1]
#
# with open("type_map.json", "w") as f:
#     f.write(json.dumps(point_map))
#
#
# 将同一对象的地点整合在一起
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',None)

df = pd.read_csv('data2.csv')
df['3']=df['3'].astype(str)
df1=df.groupby(['0']).apply(lambda x:','.join(x['3']))
print(df1.head())
df1.to_csv('data3.csv')