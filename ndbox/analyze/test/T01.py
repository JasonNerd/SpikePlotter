# file_name:      T01.py
# create_time:    2023/11/5-16:05
import numpy as np
from ndbox.analyze.single_unit_analysis import isi_plot, time_hist_plot, raster_plot, \
    autocorrelogram_plot, cum_activity_plot, iff_plot, isi_th_plot, poincare_map_plot

# test if isi work correctly
# ndbox\analyze\single_unit_analysis.py


# get a spike train with normal distribution around 120
def gen(size=1000):
    np.random.seed(26)
    a = np.random.normal(loc=120, scale=40, size=size)
    return np.sort(a)


if __name__ == '__main__':
    # res = isi_plot(gen(), '../res/isi_plot.png', bin_size=0.05, min_width=0.02, max_width=2.02)
    # print(res)
    # res = time_hist_plot(gen(), '../res/time_hist_plot.png', bin_size=2, form='step')
    # print(res)
    # raster_plot(gen(), '../res/raster_plot.png', 130, 150)
    # autocorrelogram_plot(gen(), 2.6, 7.8, 0.26, '../res/autocorrelogram_plot.png', form='step')
    # print(cum_activity_plot(gen(), '../res/cum_activity_plot.png', 180, 240))
    # print(iff_plot(gen(), '../res/iff_plot.png', 100, 160, 10, 1000))
    # print(isi_th_plot(gen(), '../res/isi_th_plot.png', 100, 160, 0.001, 0.1, form='v-line'))
    print(poincare_map_plot(gen(), '../res/poincare_map_plot.png', 100, 160, 0.001, 0.1, form='points'))
