# file_name:      analogSignalWhat.py
# create_time:    2023/10/18-21:23

# neo.AnalogSignal 是什么, 字面意义上它是指连续信号
# 数据表示为二维列表()
from neo import AnalogSignal
from quantities import Hz, ms
from elephant.spike_train_generation import StationaryPoissonProcess, StationaryGammaProcess
import numpy as np
from elephant.statistics import time_histogram


np.random.seed(1)
spp = StationaryPoissonProcess(rate=100 * Hz, t_start=0 * ms, t_stop=100 * ms)
sgp = StationaryGammaProcess(rate=100 * Hz, shape_factor=3, t_start=0 * ms, t_stop=100 * ms)
train1 = spp.generate_spiketrain()
train2 = sgp.generate_spiketrain()
print(len(train1))
print(len(train2))
print(train1.times)
print(train2.times)

# 计算 time histogram
th_t1 = time_histogram([train1], bin_size=10*ms)
print(th_t1.flatten())
th_t2 = time_histogram([train2], bin_size=10*ms)
print(th_t2.flatten())
th_t1_t2 = time_histogram([train1, train2], bin_size=10*ms)
print(th_t1_t2.flatten())
th_t2_t1 = time_histogram([train2, train1], bin_size=10*ms)
print(th_t2_t1.flatten())
"""
A representation of several continuous, analog signals that
have the same duration, sampling rate and start time.
Basically, it is a 2D array: dim 0 is time, dim 1 is channel index
"""
