# file_name:      sua.py
# create_time:    2023/11/2-10:52
import spikePy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = [
    'plot_th',
    'plot_isi'
]

from spikePy.single_unit_analysis import time_histogram, inter_spike_interval_gram


def plot_th(hist, bin_edges, axes=None, form='bar', color='#646464',
            xlabel='Time bins(sec)', ylabel='Spike counts', title='Spike time histogram',
            *args, **kwargs):
    """
    Plots a time histogram graph, the data is returned by
    :func: `spikePy.single_unit_analysis.time_histogram`

    Parameters
    ------
    hist: np.ndarray
        the height of each bar
    bin_edges: np.ndarray
        the timestamp of each bar
    form: str, {'bar', 'line', 'step'}
        default 'bar', choose from {'bar', 'curve', 'step'}
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    color: str
        Color of line or area in the plot
    xlabel: str
        The label of x-aixs
    ylabel: str
        The label of y-aixs
    title: str
        The title of the plot

    Returns
    -------
    axes : Axes

    Examples
    ------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots()
    >>> t_start, t_stop, bin_size = 0, 10, 2
    >>> # sp is your spike train data, a list of float or ndarray
    >>> hist, bin_edges=time_histogram(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size)
    >>> plot_th(hist, bin_edges, axes)
    >>> plt.show()
    """
    if axes is None:
        fig, axes = plt.subplots()
    if bin_edges.size <= 1:
        raise ValueError("Empty input.")
    bar_width = bin_edges[1] - bin_edges[0]
    if form == 'bar':
        axes.bar(bin_edges[:-1], hist, width=bar_width, color=color)
    elif form == 'line':
        axes.plot(bin_edges[:-1], hist, color=color)
    elif form == 'step':
        axes.step(bin_edges[:-1], hist, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    return axes

def plot_isi(intervals, bin_size=0.003, axes=None, form='bar', color='#646464',
             xlabel='Spike interval(sec)', ylabel='Spike counts', title='Inter-Spike Interval Distribution',
             *args, **kwargs):
    """
    plot the distribution of given intervals, which is the result of:
    :func: `spikePy.single_unit_analysis.inter_spike_interval_gram`

    Parameters
    ------
    intervals: np.ndarray
        intervals calculated by `inter_spike_interval_gram`
    bin_size: float
        its unit is seconds, the bin_size of isi-distribution
    form: str, {'bar', 'line', 'step'}
        default 'bar', choose from {'bar', 'curve', 'step'}
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    color: str
        Color of line or area in the plot
    xlabel: str
        The label of x-aixs
    ylabel: str
        The label of y-aixs
    title: str
        The title of the plot

    Returns
    -------
    axes : Axes

    Examples
    ------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots()
    >>> # sp is your spike train data, a list of float or ndarray
    >>> intervals = inter_spike_interval_gram(sp, min_width=0.002, max_width=2)
    >>> plot_isi(intervals, bin_size=0.01, axes=axes)
    >>> plt.show()
    """
    t_start = np.floor(intervals.min())
    t_stop = np.ceil(intervals.max()) 
    hist, bin_edges = time_histogram(intervals, bin_size, t_start, t_stop)
    return plot_th(hist, bin_edges, axes, form, color, xlabel, ylabel, title)


