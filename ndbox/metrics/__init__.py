from copy import deepcopy

from ndbox.utils.registry import METRIC_REGISTRY
from .metrics import calculate_R2, calculate_cc, calculate_mse

__all__ = [
    'calculate_metric',
    'calculate_R2',
    'calculate_cc',
    'calculate_mse'
]


def calculate_metric(data, opt):
    """
    Calculate metric from data and options.

    :param data: dict. Data.
    :param opt: dict. Configuration. It must contain:
        type - str. Metric type.
    :return: The result of metric.
    """

    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
