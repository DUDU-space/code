import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv('data1.csv')

# distance
# velocity
# time
# stop_rate
# point_num
# geohash_num
# geohash_rate
# heading_change_rate
# curve_rate
# velocity_change_rate
# time_slice_0
# time_slice_1
# time_slice_2
# time_slice_3
# time_slice_4

res = defaultdict(int)

r = df['velocity']
# r = r[(r >= np.percentile(r, 5)) & (r <= np.percentile(r, 95))]

cnt = 0
for i in range(10):
    down = i * 10
    up = (i + 1) * 10
    p = len(np.where((r >= down) & (r < up))[0])
    res[i] = p
    cnt += p

res[10] = r.shape[0] - cnt

# size = 100
#
# min_value = np.min(r)
# max_value = np.max(r)
#
# stride = (max_value - min_value) / size
#
# for i in range(size):
#     down = i * stride + min_value
#     up = (i + 1) * stride + min_value
#     p = len(np.where((r >= down) & (r < up))[0])
#     res[(down + up) / 2] = p

# for i in r:
#     res[i] += 1

data = []
for k, v in res.items():
    print(k, '\t', v)
