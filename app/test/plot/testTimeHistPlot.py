# file_name:      testTimeHistPlot.py
# create_time:    2023/11/2-18:54
from spikePy.plot.sua import plot_th
from spikePy.single_unit_analysis import *
from test.usage.split_stack import genSeq
import matplotlib.pyplot as plt

def testTH():
    """
    测试 plot_th 的正确性
    """
    fig, axes = plt.subplots()
    sp = genSeq()
    t_start, t_stop, bin_size = 40, 200, 2
    bin_hist, bin_arr = time_histogram(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size)
    plot_th(bin_hist, bin_arr, axes, form='step')
    plt.show()


if __name__ == '__main__':
    testTH()



