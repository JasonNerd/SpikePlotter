import numpy as np

from ndbox.analyze import psth_plot
from ndbox.utils import time_histogram, get_root_logger, plot_th, plot_psrg
from ndbox.utils.registry import ANALYZE_REGISTRY
from typing import List, Tuple
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt


@ANALYZE_REGISTRY.register()
def isi_plot(spike_train, save_path, min_width=0., max_width=np.inf, bin_size=0.05,
             axes=None, form='bar', color='#646464',
             xlabel='Spike interval(sec)', ylabel='Spike counts', title='Inter-Spike Interval Distribution',
             **kwargs):
    """
    Plot the inter-spike-interval distribution histogram of given spike train, return the intervals and its binned version.
    Also, it save the plot to target directory.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    min_width: float
        minimum width of the interval, default value is 0. It means when an interval between
        two neighboring spike is lower than this value, it won't be calculated.
    max_width: float
        maximum width of the interval, default value is np.inf. It means when an interval between
        two neighboring spike is higher than this value, it won't be calculated.
    bin_size: float
        the bin size or resolution, unit is second.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    form: str, {'bar', 'line', 'step'}
        Default 'bar', choose from {'bar', 'curve', 'step'}
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    List[np.ndarray]
        isi_hist: np.ndarray
            the binned intervals of the spike train.
        bin_edges: np.ndarray
            The bin edges.

    Raises
    ------
    ValueError:
        If form is not in {'bar', 'line', 'step'} or
        If input is empty.
    """
    logger = get_root_logger()
    intervals = np.diff(spike_train)
    if (intervals < 0).any():
        logger.warning("Negative intervals detected, please sort the input array.")
    intervals = intervals[(min_width < intervals) & (intervals < max_width)]
    t_start = np.floor(intervals.min())
    t_stop = np.ceil(intervals.max())
    isi_hist, bin_edges = time_histogram(intervals, bin_size, t_start, t_stop)
    plot_th(isi_hist, bin_edges, axes, form, color, xlabel, ylabel, title)
    plt.savefig(save_path)
    return [isi_hist, bin_edges]


@ANALYZE_REGISTRY.register()
def time_hist_plot(spike_train, save_path, bin_size=0.05, output='counts', t_start=None, t_stop=None,
                   axes=None, form='bar', color='#646464', xlabel='Time bins(sec)',
                   ylabel='Spike counts', title='Spike time histogram', **kwargs):
    """
    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    bin_size: float
        the bin size or resolution, unit is second.
    output: {'counts', 'binary', 'rate'} optional
        Normalization of the histogram. Can be one of:
        *  'counts': spike counts at each bin.
        *  'binary': binary spike counts(when >0, view as 1) at each bin.
        *  'rate': mean spike rate per spike-train.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    form: str, {'bar', 'line', 'step'}
        Default 'bar', choose from {'bar', 'curve', 'step'}
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    List[np.ndarray]
        time_hist: np.ndarray
            the binned spike of the spike train.
        bin_edges: np.ndarray
            The bin edges.

    Raises
    ------
    ValueError:
        If output is not in {'bar', 'line', 'step'} or
        If input is empty.
    """
    time_hist, bin_edges = time_histogram(spike_train, bin_size, t_start, t_stop, output)
    plot_th(time_hist, bin_edges, axes, form, color, xlabel, ylabel, title)
    plt.savefig(save_path)
    return [time_hist, bin_edges]


@ANALYZE_REGISTRY.register()
def raster_plot(spike_train, save_path, t_start=None, t_stop=None, resolution=0.002,
                axes=None, color='#646464', xlabel='Time stamps(sec)', ylabel=None, title='Raster plot', **kwargs):
    """
    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    resolution: float
        the bin size or resolution, unit is second.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    None
    """
    hist, bin_edges = time_histogram(spike_train, resolution, t_start, t_stop)
    plot_psrg(hist, bin_edges, axes, color, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(save_path)


@ANALYZE_REGISTRY.register()
def autocorrelogram_plot(spike_train, bias_start, bias_stop, bin_size, save_path,
                         t_start=None, t_stop=None, output='counts',
                         axes=None, color='#646464', line_len=0.8, form='bar',
                         xlabel='Time bins(sec)', ylabel='spike frequency', title='Autocorrelogram',
                         **kwargs):
    """
    Plot the autocorrelogram to describe autocorrelation of spiking activity.

    Parameters
    ------
    spike_train: np.ndarray
        The spike timestamps of a neuron
    bias_start: float
        The timestamp beginning the record.
    bias_stop: float
        The timestamp finishing the record.
    bin_size: float
        the bin width
    save_path: str
        The directory to store the figure.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
    output: {'counts', 'rate'} optional
        Normalization of the histogram. Can be one of:
        *  'counts': spike counts at each bin.
        *  'rate': mean spike rate per spike-train.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    line_len: float
        The raster line length scale, from 0 to 1
    form: str, {'bar', 'line', 'step'}
        default 'bar', choose from {'bar', 'curve', 'step'}
    color: str or List[str]
        Color of raster line, can be an array
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the plot

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)
        hist: np.ndarray
            The values of the histogram.
        bin_edges: np.ndarray
            Time bins of ``(length(hist)+1)``.
    """
    event_train = spike_train[(spike_train > spike_train[0] + bias_start) & (spike_train < spike_train[-1] - bias_stop)]
    return psth_plot(spike_train, event_train, bias_start, bias_stop, bin_size, save_path,
                     t_start, t_stop, output, False, axes, color, line_len, None,
                     form, xlabel, ylabel, title)


@ANALYZE_REGISTRY.register()
def cum_activity_plot(spike_train, save_path, t_start=None, t_stop=None, resolution=0.5,
                      axes=None, color='#646464', xlabel='Time stamps(sec)', ylabel=None, title='Cumulative activity',
                      **kwargs):
    """
    Plot the cumulative activity of a neuron. When detecting a spike, it step up one point.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    resolution: float
        the bin size or resolution, unit is second.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        cum_hist: np.ndarray
            The values of the histogram.
        bin_edges: np.ndarray
            Time bins of ``(length(hist)+1)``.
    """

    hist, bin_edges = time_histogram(spike_train, resolution, t_start, t_stop, output='binary')
    cum_hist = np.cumsum(hist)
    plot_th(cum_hist, bin_edges, axes, 'step', color, xlabel, ylabel, title)
    plt.savefig(save_path)
    return cum_hist, bin_edges


@ANALYZE_REGISTRY.register()
def iff_plot(spike_train, save_path, t_start=None, t_stop=None, min_freq=0, max_freq=500,
             axes=None, color='#646464', xlabel='Time stamps(sec)', ylabel='Frequency(Hz)',
             title='Instantaneous Firing Frequencies', **kwargs):
    """
    Plot the instantaneous firing frequencies of a neuron.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    min_freq: float
        If freq lower than this value, it won't be considered.
    max_freq: float
        If freq higher than this value, it won't be considered.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        cum_hist: np.ndarray
            The values of the histogram.
        bin_edges: np.ndarray
            Time bins of ``(length(hist)+1)``.
    """
    logger = get_root_logger()
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = np.ceil(np.max(spike_train))
    spike_train = spike_train[(spike_train >= t_start) & (spike_train < t_stop)]
    sp = spike_train[:-1]
    intervals = np.diff(spike_train)
    if (intervals < 0).any():
        logger.warning("Negative intervals detected, please sort the input array.")
    intervals = intervals[0 < intervals]
    sp = sp[0 < intervals]
    freq = 1. / intervals
    band_filter = (freq > min_freq) & (freq < max_freq)
    freq = freq[band_filter]
    sp = sp[band_filter]
    spike_train = np.append(sp, spike_train[-1])
    plot_th(freq, spike_train, axes, form='v-line', color=color, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(save_path)
    return freq, spike_train


@ANALYZE_REGISTRY.register()
def isi_th_plot(spike_train, save_path, t_start=None, t_stop=None, min_width=0., max_width=np.inf,
                axes=None, form='v-line', color='#646464', xlabel='Time stamps(sec)',
                ylabel='Spike interval(sec)', title='Inter-Spike Interval vs. Time',
                **kwargs):
    """
    Plot the inter-spike-interval versus time of given spike train, return the intervals and its timestamps.
    Also, it save the plot to target directory.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    min_width: float
        minimum width of the interval, default value is 0. It means when an interval between
        two neighboring spike is lower than this value, it won't be calculated.
    max_width: float
        maximum width of the interval, default value is np.inf. It means when an interval between
        two neighboring spike is higher than this value, it won't be calculated.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    form: str
        choose from {'points', 'v-line'}. Default is 'v-line'.
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    List[np.ndarray]
        isi_time_hist: np.ndarray
            the binned intervals of the spike train.
        bin_edges: np.ndarray
            The bin edges.
    """
    logger = get_root_logger()
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = np.ceil(np.max(spike_train))
    spike_train = spike_train[(spike_train >= t_start) & (spike_train < t_stop)]
    sp = spike_train[:-1]
    intervals = np.diff(spike_train)
    if (intervals < 0).any():
        logger.warning("Negative intervals detected, please sort the input array.")
    band_filter = (intervals > min_width) & (intervals < max_width)
    sp = sp[band_filter]
    intervals = intervals[band_filter]
    spike_train = np.append(sp, spike_train[-1])
    plot_th(intervals, spike_train, axes, form=form, color=color, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(save_path)
    return intervals, spike_train


@ANALYZE_REGISTRY.register()
def poincare_map_plot(spike_train, save_path, t_start=None, t_stop=None, min_width=0., max_width=np.inf,
                      axes=None, color='#646464', xlabel='The i-th interval(sec)',
                      ylabel='The (i-1)th interval(sec)', title='Poincare Map',
                      **kwargs):
    """
    Plot the Poincare map of given spike train, return the intervals and its timestamps.
    Also, it save the plot to target directory.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike timestamps of a neuron, not binned.
    save_path: str
        The directory to store the figure.
    t_start: float
        The spike-train beginning timestamp.
    t_stop: float
        The spike-train finishing timestamp.
    min_width: float
        minimum width of the interval, default value is 0. It means when an interval between
        two neighboring spike is lower than this value, it won't be calculated.
    max_width: float
        maximum width of the interval, default value is np.inf. It means when an interval between
        two neighboring spike is higher than this value, it won't be calculated.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created.
    form: str
        choose from {'points', 'v-line'}. Default is 'v-line'.
    color: str or List[str]
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the figure

    Returns
    -------
    List[np.ndarray]
        poincare_x: np.ndarray
            spike_train[i]-spike_train[i]
        poincare_y: np.ndarray
            spike_train[i-1]-spike_train[i-2]
    """
    logger = get_root_logger()
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = np.ceil(np.max(spike_train))
    spike_train = spike_train[(spike_train >= t_start) & (spike_train < t_stop)]
    intervals = np.diff(spike_train)
    if (intervals < 0).any():
        logger.warning("Negative intervals detected, please sort the input array.")
    band_filter = (intervals > min_width) & (intervals < max_width)
    intervals = intervals[band_filter]
    poincare_x = intervals[1:]
    poincare_y = intervals[:-1]
    poincare_x = np.append(poincare_x, 0)
    plot_th(poincare_y, poincare_x, axes, form='points', color=color, xlabel=xlabel, ylabel=ylabel, title=title)
    plt.savefig(save_path)
    return poincare_y, poincare_x[:-1]


def tuning_curve_plot():
    pass



