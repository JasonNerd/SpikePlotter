# file_name:      testPSRGPlot.py
# create_time:    2023/11/2-20:21

import matplotlib.pyplot as plt
import numpy as np

from spikePy.event_relate_analysis import peri_stimulus_raster_gram
from spikePy.plot.era import plot_psrg
from test.usage.split_stack import genSeq

fig, axes = plt.subplots()

# t_start, t_stop, bin_size = 0, 10, 0.5
# spike_train = np.array([0.3, 0.6, 1.4, 2.2, 2.7, 3.6,
#                         4.3, 5.7, 6.3, 6.4, 6.8, 6.92,
#                         7.32, 7.9, 8.5, 8.62, 9.23, 9.46])

t_start, t_stop, bin_size = 40, 200, 0.02
spike_train = genSeq()
event_train = np.array([32, 49, 73, 122, 163])
bias_start, bias_stop = 2, 5
epochs, bin_edges = peri_stimulus_raster_gram(spike_train, event_train,
                                              bias_start, bias_stop, bin_size, t_stop=t_stop)
# print("------Epochs------")
# print(epochs)
# print("------bin_edges------")
# print(bin_edges)

plot_psrg(epochs, bin_edges, axes)
plt.show()
