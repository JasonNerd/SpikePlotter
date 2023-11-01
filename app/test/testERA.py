# file_name:      testERA.py
# create_time:    2023/11/1-16:02
import numpy as np

from spikePy.event_relate_analysis import split_epochs

if __name__ == '__main__':
    t_start = 0
    t_stop = 10
    bin_size = 0.5
    spike_train = np.array([0.3, 0.6, 1.4, 2.2, 2.7, 3.6,
                            4.3, 5.7, 6.3, 6.4, 6.8, 6.92,
                            7.32, 7.9, 8.5, 8.62, 9.23, 9.46])
    event_train = np.array([1.5, 4.2, 6.8, 8.4])
    bias_start = 0.5
    bias_stop = 1.5
    print(split_epochs(spike_train, event_train, bias_start,
                       bias_stop, bin_size, t_stop=t_stop))
"""
print("spike_train = ", spike_train)
print()
print("binned_spike_train = ", binned_spike_train)
print()
print("event_train = ", event_train)
print()
print("event_train_bins = ", event_train_bins)
print()

spike_train =  [0.3  0.6  1.4  2.2  2.7  3.6  4.3  5.7  6.3  6.4  6.8  6.92 7.32 7.9
 8.5  8.62 9.23 9.46]

binned_spike_train =  [1 1 1 0 1 1 0 1 1 0 0 1 2 2 1 1 0 2 2 0]

event_train =  [1.5 4.2 6.8 8.4]

event_train_bins =  [ 3  8 14 17]

(array([[1, 0, 1, 1],
       [1, 1, 0, 0],
       [2, 1, 1, 0],
       [0, 2, 2, 0]], dtype=int64), array([ 3,  8, 14, 17]))
"""
