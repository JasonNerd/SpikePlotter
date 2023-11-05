# file_name:      testISIPlot.py
# create_time:    2023/11/2-18:55

from spikePy.plot.sua import plot_isi
from spikePy.single_unit_analysis import *
from test.usage.split_stack import genSeq
import matplotlib.pyplot as plt


def testISI():
    """
    测试 plot_isi 的正确性
    """
    fig, axes = plt.subplots()
    sp = genSeq()
    print(sp)
    intervals = inter_spike_interval_gram(sp, min_width=0.002, max_width=2)
    plot_isi(intervals, bin_size=0.01, axes=axes)
    plt.show()


if __name__ == '__main__':
    testISI()




