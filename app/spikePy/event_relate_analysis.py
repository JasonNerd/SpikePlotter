# file_name:      event_relate_analysis.py
# create_time:    2023/10/26-10:04

"""
event related analysis

Histogram analysis
***************
    split_to_epochs
    peri_stimulus_time_histogram

Metrics analysis
***************
    cv
    cv2
    lv

"""

from typing import List, Tuple
import numpy as np
import warnings
import spikePy.single_unit_analysis as sua

__all__ = [
    "peri_stimulus_raster_gram",
    "peri_stimulus_time_histogram",
]

from spikePy.conversion.spike_train_conv import to1dArray


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
        The timestamp beginning the record.
    bias_stop: float
        The timestamp finishing the record.
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

    Examples
    ------
    >>> t_start, t_stop, bin_size = 0, 10, 0.5
    >>> spike_train = np.array([0.3, 0.6, 1.4, 2.2, 2.7, 3.6,
    ...                         4.3, 5.7, 6.3, 6.4, 6.8, 6.92,
    ...                         7.32, 7.9, 8.5, 8.62, 9.23, 9.46])
    >>> event_train = np.array([1.5, 4.2, 6.8, 8.4])
    >>> bias_start, bias_stop = 0.5, 1.5
    >>> peri_stimulus_raster_gram(spike_train, event_train, bias_start,
    ...                 bias_stop, bin_size, t_stop=t_stop)
        (array([[1, 0, 1, 1],
               [1, 1, 0, 0],
               [2, 1, 1, 0],
               [0, 2, 2, 0]], dtype=int64), array([-1,  0,  1,  2,  3]))
    """
    # Bin the spike-train
    spike_train = to1dArray(spike_train)
    if not t_start:
        t_start = 0
    if not t_stop:
        t_stop = spike_train[-1]
    binned_spike_train, _ = sua.time_histogram(spike_train, bin_size, t_start, t_stop)
    # get event bin id
    event_train = to1dArray(event_train)
    if (event_train + bias_stop > t_stop).any():
        warnings.warn("Some event timestamps overflow, they will be discard.")
        event_train = event_train[event_train <= t_stop - bias_stop]
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


def peri_stimulus_time_histogram(spike_train, event_train, bias_start, bias_stop, bin_size,
                                 output='counts', t_start=None, t_stop=None, *args, **kwargs):
    """
    Split the spike_train using given event(stimulus/mark/flag) time series, and return
    the split epochs and corresponding event bins. Epochs[i] is the spike_train timestamps
    around event_train[i], that is [event_train[i]-bias_start, event_train[i]+bias_stop).
    The epochs can be viewed as rasters(peri-event)

    See Also
    ------
    spikePy.event_relate_analysis.peri_stimulus_raster_gram
    spikePy.plot.plot_psth

    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron
    event_train: np.ndarray or List[float]
        The stimulus time sequence
    output : {'counts', 'rate'} optional
        Normalization of the histogram. Can be one of:
        *  'counts': spike counts at each bin.
        *  'rate': mean spike rate per spike-train.
    bias_start: float
        The timestamp beginning the record.
    bias_stop: float
        The timestamp finishing the record.
    bin_size: float
        the width bins
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)
        hist:
            The values of the histogram.
        bin_edges: np.ndarray
            The bin edges ``(length(hist)+1)``.

    Examples
    ------
    >>> t_start, t_stop, bin_size = 0, 10, 0.5
    >>> spike_train = np.array([0.3, 0.6, 1.4, 2.2, 2.7, 3.6,
    ...                         4.3, 5.7, 6.3, 6.4, 6.8, 6.92,
    ...                         7.32, 7.9, 8.5, 8.62, 9.23, 9.46])
    >>> event_train = np.array([1.5, 4.2, 6.8, 8.4])
    >>> bias_start, bias_stop = 0.5, 1.5
    >>> peri_stimulus_raster_gram(spike_train, event_train, bias_start,
    ...                 bias_stop, bin_size, t_stop=t_stop)
        [4 4 4 1]
    """
    duration = bias_start + bias_stop
    psrg, bin_edges = peri_stimulus_raster_gram(spike_train, event_train, bias_start, bias_stop,
                                                bin_size, t_start, t_stop)
    bin_hist = np.sum(psrg, axis=0)
    if output == 'counts':
        pass
    elif output == 'rate':
        bin_hist = bin_hist / duration
    else:
        raise ValueError(f'Parameter output ({output}) is not valid.')
    return bin_hist, bin_edges
