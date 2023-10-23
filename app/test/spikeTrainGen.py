# file_name:      testSpikeTrain.py
# create_time:    2023/10/18-10:53
## 如何生成/得到一个 SpikeTrain 数据结构

### 按照一定的随机分布生成一个 SpikeTrain, 使用 elephant 的函数例如
# StationaryPoissonProcess.generate_spiketrain()
from elephant.spike_train_generation import StationaryPoissonProcess
from neo import SpikeTrain
from quantities import Hz, ms
import numpy as np

np.random.seed(0)
train1 = StationaryPoissonProcess(100 * Hz, t_start=0 * ms, t_stop=500 * ms).generate_spiketrain()
train2 = SpikeTrain([1.2, 6.4, 14.6, 26.3, 30.2, 31.2, 32.9, 34.1, 53.1, 73.2, 94.6] * ms,
                    t_start=0 * ms, t_stop=100 * ms)


def info(t: SpikeTrain):
    """
    neo.SpikeTrain:
    It is an ensemble of action potentials (spikes) emitted by
    the same unit in a period of time.
    """
    print(t)
    print(type(t))
    print(len(t))
    print(t.t_start)
    print(t.t_stop)
    print(t.sampling_rate)
    print("------------------------")


info(train1)
info(train2)
