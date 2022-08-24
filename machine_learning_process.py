from collections import Counter
from tqdm import tqdm
from geohash import encode    # geohash常用于将二维的经纬度转换为字符串，有现成的库可以将其编码和解码。
from helper import getDistance, getDegree   # 依据经纬度坐标计算距离及角度
import time
import numpy as np
import pymysql    # PyMySQL 是在 Python3.x 版本中用于连接 MySQL 服务器的一个库
import pandas as pd

# 打开数据库，创建连接
# "localhost"主机地址，root用户，密码，数据库
db = pymysql.connect("localhost", "root", "root", "track")
# 创建游标
cursor = db.cursor()

# 使用  execute()  方法执行 SQL 语句   select查询语句
sql = 'select mac from track.result group by mac'  # by排序，默认升序
cursor.execute(sql)
# fetchall获取剩余结果所有数据
macs = [i[0] for i in cursor.fetchall()]
track = []
point_map = dict()
# distance 总距离
# velocity 速度
# velocity_change_rate 速度的方差
# time 时间
# stop_rate 静止时间
# point_num 点数量
# geohash_num geohash数量
# geohash_rate geohash比例
# heading_change_rate 方向变化大于90度的比例
# curve_rate 总距离/直线距离 折线率
# most_common_time_slice 前5的时间片
for mac in tqdm(macs):   # macs,i[0]
    sql = 'select * from track.result where mac = %s order by time'
    cursor.execute(sql, (mac))
    res = cursor.fetchall()    # mac=%s
    res = list(res)
    res.append((0, 0, 0, 10 ** 30, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    tmp = {
        'distance': 0,
        'time': 0,
        'velocity_list': list(),
        'point_list': list(),
        'keep_time': 0,
        'geohash_set': set(),
        'heading_change': 0,
        'time_slice': list(),
        'last_degree': None,
        'keep_num': 0,
    }
    tmp['point_list'].append((res[0][5], res[0][6]))
    try:
        tmp['time_slice'].append((time.localtime(res[0][3]).tm_min // 15) + 1)  # minutes
    except:
        pass
    point_map[(res[0][5], res[0][6])] = res[0][10]
    for i in range(1, len(res)):
        # 1 mac
        # 2 ap_mac
        # 3 time
        # 4 type
        # 5 longitude
        # 6 latitude
        # 8 age
        # 9 gender
        # 10 place type
        if res[i][3] - res[i - 1][3] > 3600 or len(tmp['point_list']) > 30:
            tmpp = {
                'type': res[i - 1][4],
                'mac': res[i - 1][1],
                # 'distance': tmp['distance'],
                'velocity': (tmp['distance'] / tmp['time']) * 3.6 if tmp['time'] else 0,    # c = a if a>b else b
                # 'time': tmp['time'] // 60,
                'stop_rate': tmp['keep_time'] / tmp['time'] if tmp['time'] else 0,
                # 'point_num': len(tmp['point_list']) - tmp['keep_num'],
                # 'geohash_num': len(tmp['geohash_set']),
                'geohash_rate': len(tmp['geohash_set']) / len(tmp['point_list']),
                'heading_change_rate':
                    tmp['heading_change'] / (len(tmp['point_list']) - 2) if len(tmp['point_list']) >= 3 else 0,
            }
            try:
                zx_distance = getDistance(
                    float(tmp['point_list'][0][0]),
                    float(tmp['point_list'][0][1]),
                    float(tmp['point_list'][-1][0]),
                    float(tmp['point_list'][-1][1])
                )
                tmpp['curve_rate'] = tmp['distance'] / zx_distance
            except:
                tmpp['curve_rate'] = 0

            tmpp['velocity_change_rate'] = np.nan_to_num(np.std(tmp['velocity_list']))
            # 使用0代替数组x中的nan(not a number)元素，使用有限的数字代替inf元素   np.std求标准差
            time_slice = Counter(tmp['time_slice']).most_common()
            # most_common()函数用来实现Top n 功能
            # for j in range(5):
            #     try:
            #         tmpp['time_slice_{}'.format(j + 1)] = time_slice[j][0]
            #     except:
            #         tmpp['time_slice_{}'.format(j + 1)] = 0

            tmp = {
                'distance': 0,
                'time': 0,
                'velocity_list': list(),
                'point_list': list(),
                'keep_time': 0,
                'geohash_set': set(),
                'heading_change': 0,
                'time_slice': list(),
                'last_degree': None,
                'keep_num': 0,
            }
            tmp['point_list'].append((res[i][5], res[i][6]))
            try:
                tmp['time_slice'].append(time.localtime(res[i][3]).tm_hour
                                         + (time.localtime(res[i][3]).tm_min // 15) + 1)
            except:
                pass

            # 排除一些特殊情况
            # if tmpp['point_num'] < 5:
            #     continue
            # if tmpp['distance'] < 100:
            #     continue
            if tmpp['velocity'] > 120:
                continue

            track.append([j for j in tmpp.values()])

        geohash = encode(float(res[i][5]), float(res[i][6]))
        try:
            distance = getDistance(float(res[i][5]), float(res[i][6]), float(res[i - 1][5]), float(res[i - 1][6]))
        except:
            distance = 0
        try:
            degree = getDegree(float(res[i][5]), float(res[i][6]), float(res[i - 1][5]), float(res[i - 1][6]))
        except:
            degree = tmp['last_degree']

        if res[i][3] - res[i - 1][3] == 0:
            continue

        if distance > 300:
            tmp['distance'] += distance
        tmp['time'] += res[i][3] - res[i - 1][3]
        tmp['velocity_list'].append(distance / (res[i][3] - res[i - 1][3]))
        tmp['point_list'].append((res[i][5], res[i][6]))
        if res[i][5] == res[i - 1][5] and res[i][6] == res[i - 1][6]:  # 停留
            tmp['keep_time'] += res[i][3] - res[i - 1][3]
            tmp['keep_num'] += 1
        tmp['geohash_set'].add(geohash)
        if tmp['last_degree'] is not None:
            mx = max(degree, tmp['last_degree'])
            mi = min(degree, tmp['last_degree'])
            if 90 < mx - mi < 270:
                tmp['heading_change'] += 1
        tmp['last_degree'] = degree
        try:
            tmp['time_slice'].append((time.localtime(res[i][3]).tm_min // 15) + 1)
        except:
            pass
        point_map[(res[i][5], res[i][6])] = res[i][10]

columns = [
    'type', 'mac', 'velocity', 'stop_rate', 'geohash_rate',
    'heading_change_rate', 'curve_rate', 'velocity_change_rate'
]
# for i in range(5):
#     columns.append('time_slice_{}'.format(i))

df = pd.DataFrame(np.array(track), columns=columns)
df.to_csv('data1.csv', index=False)
