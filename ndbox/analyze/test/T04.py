# file_name:      T04.py
# create_time:    2023/11/8-14:42

import numpy as np
import quantities as pq
import elephant
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.spike_train_synchrony import spike_contrast

np.random.seed(24)
spiketrains = [homogeneous_poisson_process(rate=20 * pq.Hz,
               t_stop=10 * pq.s) for _ in range(10)]
synchrony, trace = spike_contrast(spiketrains, return_trace=True)
print(len(trace.contrast))
print(np.array(trace.contrast))
print(trace.bin_size)
print(trace.bin_size)

