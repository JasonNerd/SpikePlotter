# file_name:      testPSTHPlot.py
# create_time:    2023/11/3-10:08

import matplotlib.pyplot as plt
import numpy as np

from spikePy.event_relate_analysis import peri_stimulus_time_histogram
from spikePy.plot.era import plot_psth
from test.usage.split_stack import genSeq


fig, axes = plt.subplots()

t_start, t_stop, bin_size = 40, 200, 0.02
spike_train = genSeq()
event_train = np.array([32, 49, 73, 122, 163])
bias_start, bias_stop = 2, 5
hist, bin_edges = peri_stimulus_time_histogram(spike_train, event_train, bias_start, bias_stop,
                                               bin_size, t_start=t_start, t_stop=t_stop)
plot_psth(hist, bin_edges, axes)
plt.show()

