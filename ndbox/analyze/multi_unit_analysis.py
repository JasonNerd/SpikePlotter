# file_name:      multi_unit_analysis.py
# create_time:    2023/11/7-14:08

import numpy as np

from ndbox.analyze import psth_plot
from ndbox.utils import time_histogram, get_root_logger, plot_th, plot_psrg
from ndbox.utils.registry import ANALYZE_REGISTRY
from typing import List, Tuple
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt


@ANALYZE_REGISTRY.register()
# https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_correlation/elephant.spike_train_correlation.cross_correlation_histogram.html
def cch_plot(target_spike, refer_spike, bias_start, bias_stop, bin_size,
             save_path, t_start=None, t_stop=None, event_train=None, shift_predictor=None,
             axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts', title='Cross-correlograms',
             **kwargs
             ):
    """
    Plot the cross-correlation histogram of two spike-train.

    Notes
    ------
    Shift-Predictor:
    If you do this simultaneously in both cells -- which is usually the whole idea of the experiment -- what
    you are doing is simultaneously increasing the firing rate of both cells at the same time; thus, you've
    introduced a relationship between the firing probabilities of the cells, just by co-stimulating them.

    Parameters
    ----------
    target_spike: np.ndarray
        The target neuron's spike timestamps
    refer_spike: np.ndarray
        The refer neuron's spike timestamps
    bin_size: float
        the bin width
    save_path: str
        The directory to store the figure.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
    event_train: np.ndarray
        The stimulus time sequence
    bias_start: float
        The time bias before the event.
    bias_stop: float
        The time bias after the event.
    shift_predictor: int or str
        A number between (0, len(event_train)) or str choose from {'random', 'average'}.
        Shift-predictor is defined for a series of trials - you take the spikes of one neuron
        in trial 1 and correlate them with the spikes of another neuron in trial 2, etc.
        if 'random', a random shift will be selected.
        if 'average', all possible shift will be considered, and calculate the average.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
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
    Tuple(np.ndarray, np.ndarray)
        cross_hist: np.ndarray
            Cross-correlation histogram
        bin_edges: np.ndarray
            Time bins of ``(length(cross_hist)+1)``.


    References
    -------
    [1] https://plexon.com/wp-content/uploads/2017/06/NeuroExplorer-v5-Manual.pdf

    [2] https://www.med.upenn.edu/mulab/crosscorrelation.html
    """
    logger = get_root_logger()
    if shift_predictor is not None and event_train is None:
        raise ValueError("Event time series(or we say trial time intervals) is not given!")
    elif shift_predictor is None and event_train is not None:
        logger.warning("Shift-predictor not given, a random shift-predictor will be applied.")
    elif shift_predictor is not None and event_train is not None:
        return psth_plot(target_spike, refer_spike, bias_start, bias_stop, bin_size, save_path, t_start, t_stop,
                         'counts', False, axes, color, 0.8, None, 'step', xlabel, ylabel, title)
    else:
        raise NotImplementedError("Shift predictor to be implemented.")


@ANALYZE_REGISTRY.register()
# https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_synchrony/elephant.spike_train_synchrony.spike_contrast.html
def scs_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
             axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts', title='Cross-correlograms',
             **kwargs
             ):
    """
    Spike-Contrast Synchrony plot. It calculate the synchrony of spike trains,
    the spike trains can have different length. We use spike-contrast as a
    measurement of spike-train synchrony.

    Parameters
    ----------
    spike_list: List[np.ndarray] or np.ndarray
        if data type is `List[np.ndarray]`, then spike_list[i] indicates the i-th neuron's
        spike stamps, spike_list[i] is a 1D array.
        if data type is np.ndarray, then it must be 2D array, spike_list[i] is the binned
        firing counts of the i-th neuron. Note that you must pass the `bin_size` argument
        in this way.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
    bin_size: float
        the bin width
    shrink: float
        A multiplier to shrink the bin size on each iteration.
        The value must be in range (0, 1). Default: 0.9
    save_path: str
        The directory to store the figure.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
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
    A history of spike-contrast synchrony, computed for a range of different bin sizes,
    alongside with the maximum value of the synchrony.

    Tuple()
        synchrony: float
            Returns the synchrony of the input spike trains
        trace_contrast: np.ndarray
             the average sum of differences of the number of spikes in subsequent bins
        trace_active: np.ndarray
            the average number of spikes per bin, weighted by the number of spike trains
            containing at least one spike inside the bin;
        trace_synchrony: np.ndarray
            the product of `trace_contrast` and `trace_active`;
        bin_size: np.ndarray
            the X axis, a list of bin sizes that correspond to these traces.


    Raises
    ------
    ValueError
        If bin_shrink_factor is not in (0, 1) range.
        If the input spike trains contain no more than one spike-train.
    TypeError
        If the input spike trains is not a list or ndarray.
    """
    check_input(spike_list, bin_size)
    # TODO
    pass


@ANALYZE_REGISTRY.register()
# https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_correlation/elephant.spike_train_correlation.correlation_coefficient.html
def coefficient_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
                     axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts',
                     title='Cross-correlograms',
                     **kwargs
                     ):
    """
    Correlation Coefficient plot. For each pair of spike trains, the correlation coefficient
    is obtained by binning and at the desired bin size. For an input of N spike trains, an
    N x N matrix is returned. Each entry in the matrix is a real number ranging between
    -1 (perfectly anti-correlated spike trains) and +1 (perfectly correlated spike trains).

    Parameters
    ----------
    spike_list: List[np.ndarray] or np.ndarray
        if data type is `List[np.ndarray]`, then spike_list[i] indicates the i-th neuron's
        spike stamps, spike_list[i] is a 1D array.
        if data type is np.ndarray, then it must be 2D array, spike_list[i] is the binned
        firing counts of the i-th neuron. Note that you must pass the `bin_size` argument
        in this way.
    t_start: float
        The record beginning timestamp.
    t_stop: float
        The record finishing timestamp.
    bin_size: float
        the bin width
    shrink: float
        A multiplier to shrink the bin size on each iteration.
        The value must be in range (0, 1). Default: 0.9
    save_path: str
        The directory to store the figure.
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
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
    A history of spike-contrast synchrony, computed for a range of different bin sizes,
    alongside with the maximum value of the synchrony.

    dict
        'synchrony': float
            Returns the synchrony of the input spike trains
        'trace_contrast': np.ndarray
             the average sum of differences of the number of spikes in subsequent bins
        'trace_active': np.ndarray
            the average number of spikes per bin, weighted by the number of spike trains
            containing at least one spike inside the bin;
        'trace_synchrony': np.ndarray
            the product of `trace_contrast` and `trace_active`;
        'bin_size': np.ndarray
            the X axis, a list of bin sizes that correspond to these traces.


    Raises
    ------
    ValueError
        If bin_shrink_factor is not in (0, 1) range.
        If the input spike trains contain no more than one spike-train.
    TypeError
        If the input spike trains is not a list or ndarray.
    """
    check_input(spike_list, bin_size)


@ANALYZE_REGISTRY.register()
def spade_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
               axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts', title='Cross-correlograms',
               **kwargs
               ):
    pass


# https://elephant.readthedocs.io/en/latest/reference/cell_assembly_detection.html
def cad_plot():
    pass


def check_input(spike_list, bin_size):
    if isinstance(spike_list, np.ndarray):
        if spike_list.ndim < 2:
            raise TypeError("Get a numpy array. However, it's not 2-D.")
        elif bin_size is None:
            raise ValueError("Binned 2D array as input, however, bin_size is None.")
        else:
            raise NotImplementedError("to be continued ... ...")
    if isinstance(spike_list, list):
        if len(spike_list) < 2:
            raise ValueError("Data list should have more than 2 spike trains.")
        if not isinstance(spike_list[0], np.ndarray):
            raise TypeError("A numpy array is expected.")
