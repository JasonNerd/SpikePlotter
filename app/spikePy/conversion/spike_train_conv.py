# file_name:      spike_train_conv.py
# create_time:    2023/10/31-21:46
import warnings
from typing import List
import numpy as np

def to1dArray(spike_train):
    """
    Parameters
    ------
    spike_train: np.ndarray or List[float]
        The spike timestamps of a neuron

    Returns
    ------
    np.ndarray
        the 1-D spike time sequence form of given spike-train.
    """
    if isinstance(spike_train, list):
        spike_train = np.array(spike_train)
    if spike_train.size == 0:
        raise ValueError("Empty input.")
    spike_train = spike_train.flatten()
    return spike_train

def detectIfSorted(spike_train):
    """
    Parameters
    ------
    spike_train: np.ndarray
        The spike timestamps of a neuron

    Returns
    ------
    np.ndarray
        the 1-D spike time sequence form of given spike-train.
    """
    intervals = np.diff(spike_train)
    if (intervals < 0).any():
        warnings.warn("Negative intervals detected, please sort the input array.")
    return intervals
