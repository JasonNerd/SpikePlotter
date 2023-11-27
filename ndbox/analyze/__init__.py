from copy import deepcopy

from ndbox.analyze.event_related_analysis import psr_plot, psth_plot
from ndbox.analyze.single_unit_analysis import isi_plot, time_hist_plot, raster_plot
from ndbox.utils.registry import ANALYZE_REGISTRY

__all__ = [
    'build_analyze',
    'isi_plot',
    'time_hist_plot',
    'raster_plot',
    'psr_plot',
    'psth_plot'
]


def build_analyze(data, opt):
    """
    Build analyze from options.

    :param data: dict. Data.
    :param opt: dict. Configuration. It must contain:
        type - str. Metric type.
    """

    opt = deepcopy(opt)
    analyze_type = opt.pop('type')
    result = ANALYZE_REGISTRY.get(analyze_type)(**data, **opt)
    return result
