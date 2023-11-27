# file_name:      T02.py
# create_time:    2023/11/6-10:36
import matplotlib.pyplot as plt
from ndbox.analyze.event_related_analysis import psr_plot, psth_plot
import numpy as np


# test the functions in event_related_analysis

# get a spike train with normal distribution around 120
def gen(size=1000):
    np.random.seed(26)
    a = np.random.normal(loc=120, scale=40, size=size)
    return np.sort(a)


event = np.array([26., 69., 130., 176., 203.])

if __name__ == '__main__':
    bias_start, bias_stop = 2.6, 6.5
    bin_size = 0.26
    psr_plot(gen(), event, bias_start, bias_stop, bin_size, "../res/peri_sti_raster.png")
    psth_plot(gen(), event, bias_start, bias_stop, bin_size, "../res/psth_raw.png")
    psth_plot(gen(), event, bias_start, bias_stop, bin_size,
              "../res/psth_with_psr.png", raster_aligned=True, form="step")
