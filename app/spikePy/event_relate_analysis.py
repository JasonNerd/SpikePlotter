# file_name:      event_relate_anasis.py
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

__all__ = [
    "split_to_epochs",
    "peri_stimulus_time_histogram",
]

def split_to_epochs(spike_train, event_train, bias_start,
                    bias_stop, bin_size, *args, **kwargs):
    """
    Split the spike_train using given event(stimulus/mark/flag) time series, and return
    the split epochs and corresponding time intervals. Epochs[i] is the spike_train timestamps
    around event_train[i], that is [event_train[i]-bias_start, event_train[i]+bias_stop)

    See Also
    ------
    spikePy.peri_stimulus_time_histogram

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

    Returns
    ------
    Tuple(np.ndarray, np.ndarray)


    Examples
    ------
    >>> print("abc")

    """
    pass


def peri_stimulus_time_histogram():
    pass


