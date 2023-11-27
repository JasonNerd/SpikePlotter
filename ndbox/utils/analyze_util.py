import numpy as np
from .logger import get_root_logger
import matplotlib.pyplot as plt

def time_histogram(spike_train, bin_size, t_start=None, t_stop=None,
                   output='counts', *args, **kwargs):
    """
    Calculate the binned version of given spike-train.

    Note
    ------
    1. all params should have same unit, like 's'.
    2. you may make sure that (t_stop-t_start)/bin_size is an integer

    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
    bin_size: float
        the width of histogram bins.
    output: {'counts', 'binary', 'rate'} optional
        Normalization of the histogram. Can be one of:
        *  'counts': spike counts at each bin.
        *  'binary': binary spike counts(when >0, view as 1) at each bin.
        *  'rate': mean spike rate per spike-train.

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)
        hist:
            The values of the histogram.
        bin_edges: np.ndarray
            The bin edges ``(length(hist)+1)``.

    See Also
    ------
    spikePy.plot.sua.plot_th()

    Examples
    ------
    >>> sp = [0.1, 3.2, 3.37, 3.92, 8.2, 9.6]
    >>> t_start, t_stop, bin_size = 0, 10, 2
    >>> time_histogram(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size)
        (array([1, 3, 0, 0, 2], dtype=int64), array([ 0,  2,  4,  6,  8, 10]))
    >>> time_histogram(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size, output='binary')
        (array([1, 1, 0, 0, 1]), array([ 0,  2,  4,  6,  8, 10]))
    >>> time_histogram(sp, t_start=t_start, t_stop=t_stop, bin_size=bin_size, output='rate')
        (array([0.1, 0.3, 0. , 0. , 0.2]), array([ 0,  2,  4,  6,  8, 10]))
    """
    logger = get_root_logger()
    if not t_start:
        t_start = 0
    if not t_stop:
        t_stop = np.ceil(spike_train.max())
    # calculate the hist
    duration = t_stop - t_start
    if duration / bin_size != int(duration / bin_size):
        logger.warning(f"The last bin's size is smaller, due to "
                       f"bin-count=({t_stop}-{t_start})/{bin_size}, "
                       f"and this is not an integer.")
    bin_arr = np.arange(t_start, t_stop, bin_size)
    bin_arr = np.append(bin_arr, t_stop)
    hist, bin_edges = np.histogram(spike_train, bin_arr)
    if output == 'counts':
        pass
    elif output == 'rate':
        hist = hist / duration
    elif output == 'binary':
        hist = np.where(hist > 0, 1, 0)
    else:
        raise ValueError(f'Parameter output ({output}) is not valid.')
    return hist, bin_edges


def peri_stimulus_raster_gram(spike_train, event_train, bias_start, bias_stop, bin_size,
                              t_start=None, t_stop=None, *args, **kwargs):
    """
    Split the spike_train using given event(stimulus/mark/flag) time series, and return
    the split epochs and corresponding event bins. Epochs[i] is the spike_train timestamps
    around event_train[i], that is [event_train[i]-bias_start, event_train[i]+bias_stop).
    The epochs can be viewed as rasters(peri-event)

    See Also
    ------
    spikePy.event_relate_analysis.peri_stimulus_time_histogram

    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron
    event_train: np.ndarray or List[float]
        The stimulus time sequence
    bias_start: float
        The time bias before the event.
    bias_stop: float
        The time bias after the event.
    bin_size: float
        the width bins
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)
        epochs: np.ndarray
            All spike bins around the given stimulus in event train
        bin_edges: np.ndarray
            Time bins of ``(length(epochs[0])+1)``.
    """
    logger = get_root_logger()
    if not t_start:
        t_start = 0
    if not t_start:
        t_stop = np.ceil(spike_train[-1])
    binned_spike_train, _ = time_histogram(spike_train, bin_size, t_start, t_stop)
    if (event_train + bias_stop > t_stop).any():
        logger.warning("Some event timestamps overflow, they will be discard.")
        event_train = event_train[(event_train > t_start + bias_start) & (event_train <= t_stop - bias_stop)]
    event_train_bins = (event_train - t_start) / bin_size
    event_train_bins = np.round(event_train_bins).astype(np.int32)
    bias_start_bin = int(bias_start / bin_size)
    bias_stop_bin = int(bias_stop / bin_size)
    # 计算分出的 epoch 结果
    tmp_list = []
    for e in event_train_bins:
        tmp_list.append(binned_spike_train[e - bias_start_bin: e + bias_stop_bin])
    epochs = np.stack(tmp_list, axis=1)
    epochs = epochs.T
    return epochs, (np.array(list(range(epochs.shape[1] + 1))) - bias_start_bin)*bin_size


def plot_th(hist, bin_edges, axes=None, form='bar', color='#646464',
            xlabel='Time bins(sec)', ylabel='Spike counts', title='Spike time histogram',
            **kwargs):
    """
    Plots a time histogram graph, and save the figure to given save path

    Parameters
    ------
    hist: np.ndarray
        The height of each bar
    bin_edges: np.ndarray
        The timestamp of bar edges
    form: str, {'bar', 'line', 'step', 'v-line', 'points}
        Default 'bar', choose from {'bar', 'curve', 'step', 'v-line', 'points'}
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    color: str
        Color of line or area in the plot
    xlabel: str
        The label of x-axis
    ylabel: str
        The label of y-axis
    title: str
        The title of the plot
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
    elif form == 'v-line':
        axes.vlines(bin_edges[:-1], np.zeros(hist.size), hist, colors=color)
    elif form == 'points':
        axes.scatter(bin_edges[:-1], hist, color=color, s=3)
    else:
        raise ValueError("Output form invalid.")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    return axes


def plot_psrg(epochs, bin_edges, axes=None, color='#646464', line_len=0.8, event_names=None,
              xlabel='Time bins(sec)', ylabel='Events', title='Peri-stimulus raster gram',
              **kwargs):
    if epochs.ndim < 2:
        epochs = np.array([epochs])
    if axes is None:
        fig, axes = plt.subplots()
    if bin_edges.size <= 1:
        raise ValueError("Empty input.")
    bin_size = bin_edges[1] - bin_edges[0]
    raster_list = [np.where(epoch > 0)[0] * bin_size + bin_edges[0] for epoch in epochs]
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



