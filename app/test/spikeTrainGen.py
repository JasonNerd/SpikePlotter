# file_name:      testSpikeTrain.py
# create_time:    2023/10/18-10:53
## 如何生成/得到一个 SpikeTrain 数据结构

### 按照一定的随机分布生成一个 SpikeTrain, 使用 elephant 的函数例如
# StationaryPoissonProcess.generate_spiketrain()
from elephant.spike_train_generation import StationaryPoissonProcess
from quantities import Hz, ms
import numpy as np

np.random.seed(0)
train1 = StationaryPoissonProcess(100*Hz, t_start=0*ms, t_stop=500*ms).generate_spiketrain()
print(train1)
print(type(train1))
print(len(train1))

"""
neo.SpikeTrain:
It is an ensemble of action potentials (spikes) emitted by 
the same unit in a period of time.
"""

