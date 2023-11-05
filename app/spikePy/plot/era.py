# file_name:      era.py
# create_time:    2023/11/2-10:51
from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = [
    'plot_psrg',
    'plot_psth'
]

from spikePy.event_relate_analysis import peri_stimulus_raster_gram
from spikePy.plot.sua import plot_th


def plot_psrg(epochs, bin_edges, axes=None, color='#646464', line_len=0.8, event_names=None,
              xlabel='Time bins(sec)', ylabel='Events', title='Peri-stimulus raster gram',
              *args, **kwargs):
    """
    Plots a raster graph around each event, the data is returned by
    :func: `spikePy.event_relate_analysis.peri_stimulus_raster_gram`

    Parameters
    ------
    epochs: np.ndarray
        All spike bins around the given stimulus in event train
    bin_edges: np.ndarray
        The time bin stamps
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    line_len: float
        The raster line length scale, from 0 to 1
    event_names: List[str] or None
        Epochs is calculated on each event, this arg means the name of each event, can be used
        as y-ticks.
    color: str or List[str]
        Color of raster line, can be an array
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the plot

    Returns
    -------
    axes : Axes

    Examples
    ------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots()
    >>> t_start, t_stop, bin_size = 0, 10, 0.5
    >>> spike_train = np.array([0.3, 0.6, 1.4, 2.2, 2.7, 3.6,
    ...                         4.3, 5.7, 6.3, 6.4, 6.8, 6.92,
    ...                         7.32, 7.9, 8.5, 8.62, 9.23, 9.46])
    >>> event_train = np.array([1.5, 4.2, 6.8, 8.4])
    >>> bias_start, bias_stop = 0.5, 1.5
    >>> epochs, bin_edges = peri_stimulus_raster_gram(spike_train, event_train, bias_start,
    ...                 bias_stop, bin_size, t_stop=t_stop)
    >>> plot_psrg(epochs, bin_edges, axes)
    >>> plt.show()
    """
    if axes is None:
        fig, axes = plt.subplots()
    if bin_edges.size <= 1:
        raise ValueError("Empty input.")
    bin_size = bin_edges[1] - bin_edges[0]
    raster_list = [np.where(epoch > 0)[0]*bin_size+bin_edges[0] for epoch in epochs]
    if event_names is None:
        line_offsets = 1
    else:
        line_offsets = event_names
    axes.eventplot(raster_list, linelengths=line_len, colors=color,
                   lineoffsets=line_offsets)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    return axes


def plot_psth(hist, bin_edges, axes=None, form='bar', color='#646464',
              xlabel='Time bins(sec)', ylabel='Spike counts', title='Peri-stimulus time histogram',
              *args, **kwargs):
    """
    Plots a peri-stimulus time histogram graph, the data is returned by
    :func: `spikePy.event_relate_analysis.peri_stimulus_time_histogram`

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
    >>> # sp is your spike train data
    >>> hist, bin_edges=(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size)
    >>> plot_psth(hist, bin_edges, axes)
    >>> plt.show()
    """
    return plot_th(hist, bin_edges, axes, form, color, xlabel, ylabel, title, args, kwargs)

