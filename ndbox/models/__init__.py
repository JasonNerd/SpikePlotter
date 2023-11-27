from copy import deepcopy

from .wiener_filter import WienerFilterRegression
from .wiener_cascade import WienerCascadeRegression
from .kalman_filter import KalmanFilterRegression

from ndbox.utils import get_root_logger
from ndbox.utils.registry import MODEL_REGISTRY

__all__ = [
    'build_model',
    'WienerFilterRegression',
    'WienerCascadeRegression',
    'KalmanFilterRegression'
]


def build_model(opt):
    """
    Build model from options.

    :param opt: dict. Configuration. It must contain:
        model_type - str. Model type.
    :return: The specified model.
    """

    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
