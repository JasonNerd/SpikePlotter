# file_name:      single_unit_analysis.py
# create_time:    2023/10/26-10:03
"""
single unit analysis

Histogram analysis
***************
    time_histogram
    inter_spike_interval_gram
    raster_gram
    cumulative_activity_gram
    instantaneous_firing_rate_gram
    auto_correlation_gram

Metrics analysis
***************
    cv
    cv2
    lv

"""
from typing import List, Tuple
import numpy as np
import warnings
import spikePy.conversion.spike_train_conv as sc

__all__ = [
    "time_histogram",
    "inter_spike_interval_gram",

]


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
    # transfer to 1D array
    spike_train = sc.to1dArray(spike_train)
    if not t_start:
        t_start = 0
    if not t_stop:
        t_stop = spike_train[-1]
    # calculate the hist
    duration = t_stop - t_start
    if duration / bin_size != int(duration / bin_size):
        warnings.warn(f"The last bin's size is smaller, due to "
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


def inter_spike_interval_gram(spike_train, min_width=0., max_width=np.inf, *args, **kwargs):
    """
    Return an array containing the inter-spike intervals of given spike train.

    Note
    ------
    1. the timestamp in spike_train should be sorted ascend, otherwise negative value may occur

    See Also
    ------
    spikePy.plot.sua.plot_isi()

    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron
    min_width: float
        minumum width of the interval, default value is 0
    max_width: float
        maximum width of the interval, default value is np.inf

    Returns
    ------
    np.ndarray
        the inter-spike intervals of given spike train.

    Examples
    ------
    >>> sp = [0.1, 3.2, 3.37, 3.92, 8.2, 9.6]
    >>> inter_spike_interval_gram(sp)
        [3.1 0.17 0.55 4.28 1.4]
    """
    spike_train = sc.to1dArray(spike_train)
    intervals = sc.detectIfSorted(spike_train)
    return intervals[(min_width < intervals) & (intervals < max_width)]


def raster_gram(spike_train, *args, **kwargs):
    """
    Mark that raster can be a direct way to preview the spike-train

    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron.

    Returns
    ------
    np.ndarray
        just the original spike-train.

    See Also
    ------
    spikePy.plot.plot_raster()
    """
    spike_train = sc.to1dArray(spike_train)
    return spike_train


if __name__ == '__main__':
    sp = [0.1, 3.2, 3.37, 3.92, 8.2, 9.6]
    print(inter_spike_interval_gram(sp, min_width=1, max_width=8))



