# file_name:      event_related_analysis.py
# create_time:    2023/11/6-10:20
import numpy as np
from ndbox.utils import time_histogram, get_root_logger, plot_th, plot_psrg, peri_stimulus_raster_gram
from ndbox.utils.registry import ANALYZE_REGISTRY
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes


@ANALYZE_REGISTRY.register()
def psr_plot(spike_train, event_train, bias_start,
             bias_stop, bin_size, save_path, t_start=None, t_stop=None,
             axes=None, color='#646464', line_len=0.8, event_names=None,
             xlabel='Time bins(sec)', ylabel='Events', title='Peri-stimulus raster gram',
             **kwargs):
    """
    Plot the peri-stimulus raster gram
    Split the spike_train using given event(stimulus/mark/flag) time series, and return
    the split epochs and corresponding bin edges. Epochs[i] is the spike_train timestamps
    around event_train[i], that is [event_train[i]-bias_start, event_train[i]+bias_stop).
    Sometimes, [event_train[i]-bias_start, event_train[i]+bias_stop) indicates
    the i-th trail's interval.

    Parameters
    ------
    spike_train: np.ndarray
        The spike timestamps of a neuron
    event_train: np.ndarray
        The stimulus time sequence
    bias_start: float
        The time bias before the event.
    bias_stop: float
        The time bias after the event.
    bin_size: float
        the bin width
    save_path: str
        The directory to store the figure.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
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
    ------
    Tuple(np.ndarray, np.ndarray)
        epochs: np.ndarray
            All spike bins around the given stimulus in event train
        bin_edges: np.ndarray
            Time bins of ``(length(epochs[0])+1)``.
    """
    epochs, bin_edges = peri_stimulus_raster_gram(spike_train, event_train, bias_start, bias_stop,
                                                  bin_size, t_start, t_stop)
    plot_psrg(epochs, bin_edges, axes, color, line_len, event_names, xlabel, ylabel, title)
    plt.savefig(save_path)
    return epochs, bin_edges


@ANALYZE_REGISTRY.register()
def psth_plot(spike_train, event_train, bias_start, bias_stop, bin_size, save_path,
              t_start=None, t_stop=None, output='counts', raster_aligned=False,
              axes=None, color='#646464', line_len=0.8, event_names=None, form='bar',
              xlabel='Time bins(sec)', ylabel='Events', title='Peri-stimulus time histogram',
              **kwargs):
    """
    Plot the peri-stimulus time histogram

    Parameters
    ------
    spike_train: np.ndarray
        The spike timestamps of a neuron
    event_train: np.ndarray
        The stimulus time sequence
    bias_start: float
        The time bias before the event.
    bias_stop: float
        The time bias after the event.
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
    raster_aligned: bool
        if align with the peri-stimulus-raster-plot
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    line_len: float
        The raster line length scale, from 0 to 1
    event_names: List[str] or None
        Epochs is calculated on each event, this arg means the name of each event, can be used
        as y-ticks.
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
    duration = bias_start + bias_stop
    epochs, bin_edges = peri_stimulus_raster_gram(spike_train, event_train, bias_start, bias_stop,
                                                  bin_size, t_start, t_stop)
    hist = np.sum(epochs, axis=0)
    if output == 'counts':
        pass
    elif output == 'rate':
        hist = hist / duration
    else:
        raise ValueError(f'Parameter output ({output}) is not valid.')
    if raster_aligned:
        fig, axes = plt.subplots(2, 1)
        plot_psrg(epochs, bin_edges, axes=axes[0], event_names=event_names, xlabel=None, title=title)
        plot_th(hist, bin_edges, axes[1], form, color, xlabel, ylabel=None, title=None)
    else:
        plot_th(hist, bin_edges, axes, form, color, xlabel, ylabel, title)
    if save_path is not None:
        plt.savefig(save_path)
    return hist, bin_edges


def joint_psth_plot():
    pass

def syn_event_plot():
    pass

