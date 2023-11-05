# file_name:      npu.py
# create_time:    2023/10/31-22:07

# numpy usage
import numpy as np
import datetime
from spikePy.single_unit_analysis import inter_spike_interval_gram


def time_spend(foo):
    def warp(*args, **kwargs):
        s = datetime.datetime.now()
        foo(*args, **kwargs)
        e = datetime.datetime.now()
        print(f"Time spend: {(e - s).microseconds / 1000} ms.")
    return warp


def genSeq(size=1000):
    np.random.seed(26)
    a = np.random.normal(loc=120, scale=40, size=size)
    return np.sort(a)


# numpy 能否实现按条件分段?
@time_spend
def testSegment():
    # c 为事件序列
    c = np.random.randint(low=30, high=150, size=2000)
    # center front back 确定了段所在区间
    f, b = 2, 3
    tmp_res_list = []
    for ci in c:
        tmp_res_list.append(a[ci-f: ci+b])
    res = np.stack(tmp_res_list, axis=1)
    res = res.T
    print(res.shape)
    print(c[-1])
    print(res[-1])


if __name__ == '__main__':
    a = genSeq()
    testSegment()

