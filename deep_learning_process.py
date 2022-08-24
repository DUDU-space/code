import json  # 轻量级的数据交换格式，对 JSON 数据进行编解码

from tqdm import tqdm
import numpy as np
import pymysql
import pandas as pd

db = pymysql.connect("localhost", "root", "root", "track")
cursor = db.cursor()

sql = 'select mac from track.result group by mac'
cursor.execute(sql)

macs = [i[0] for i in cursor.fetchall()]
track = []
point_map = dict()

for mac in tqdm(macs):
    sql = 'select * from track.result where mac = %s order by time'   # 所有同一mac地址的记录
    cursor.execute(sql, (mac))
    res = cursor.fetchall()
    res = list(res)
    res.append((0, 0, 0, 10 ** 30, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    point_list = list()
    point_list.append(res[0][2])
    point_map[res[0][2]] = res[0][10]
    # point_list.append(res[0][10] if res[0][10] else '')
    for i in range(1, len(res)):
        # 1 mac
        # 2 ap_mac  点的标记 point_list
        # 3 time
        # 4 type
        # 5 longitude
        # 6 latitude
        # 8 age
        # 9 gender
        # 10 place type
        if res[i][3] - res[i - 1][3] > 3600 or len(point_list) > 30:
            # 排除一些特殊情况
            if len(point_list) < 5:
                point_list = list()
                point_list.append(res[i][2])
                point_map[res[i][2]] = res[i][10]
                # point_list.append(res[i][10] if res[i][10] else '')
                continue

            f = 0
            for point in point_list:
                for type_ in ['路边', '小区', '执法部门', '教育场所', '综合体']:
                    if type_ == point_map[point]:
                        f = 1
            if f == 0:
                point_list = list()
                point_list.append(res[i][2])
                point_map[res[i][2]] = res[i][10]
                # point_list.append(res[i][10] if res[i][10] else '')
                continue

            point_list.insert(0, res[i - 1][4])
            point_list.insert(1, res[i - 1][1])
            track.append([point_list[0], point_list[1], " ".join(point_list[2:])])

            point_list = list()

        point_list.append(res[i][2])
        # place_type = res[i][10] if res[i][10] else ''
        # point_list.append(place_type)
        point_map[res[i][2]] = res[i][10]

with open("point_map.json", "w") as f:
    f.write(json.dumps(point_map))    # 对python对象进行序列化。将一个Python对象进行JSON格式的编码。

df = pd.DataFrame(np.array(track))
df.to_csv('data2.csv', index=False)    # type mac ap_mac
# df.to_csv('data3.csv', index=False)
