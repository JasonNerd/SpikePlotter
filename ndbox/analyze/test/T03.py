# file_name:      T03.py
# create_time:    2023/11/7-19:41
import numpy as np
import quantities as pq
from ndbox.analyze.multi_unit_analysis import cch_plot
from elephant.spike_train_correlation import cross_correlation_histogram
from datetime import datetime
from elephant.conversion import BinnedSpikeTrain
import matplotlib.pyplot as plt

def gen(r, t):
    from elephant.spike_train_generation import StationaryPoissonProcess
    import quantities as pq
    np.random.seed(26)
    s = StationaryPoissonProcess(rate=r * pq.Hz, t_stop=t * pq.s).generate_spiketrain()
    return s

def test_mine():
    a = gen(200, 10)
    b = gen(150, 10)
    bias_start = 0.2
    bias_stop = 0.2
    bin_size = 0.005
    t1 = datetime.now()
    cch_plot(a.magnitude.flatten(), b.magnitude.flatten(),
             bias_start, bias_stop, bin_size, '../res/cch_plot.png')
    t2 = datetime.now()
    ba = BinnedSpikeTrain(a, bin_size=5*pq.ms)
    bb = BinnedSpikeTrain(b, bin_size=5*pq.ms)
    cc_hist, lags = cross_correlation_histogram(ba, bb, window=[-40, 40])
    t3 = datetime.now()
    print(t2-t1)
    print(t3-t2)
    cc_hist = cc_hist.magnitude.flatten()
    plt.step(lags, cc_hist)
    plt.show()


if __name__ == '__main__':
    test_mine()
