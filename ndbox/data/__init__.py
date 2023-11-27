from copy import deepcopy

from .nwb_dataset import NWBDataset

from ndbox.utils import get_root_logger
from ndbox.utils.registry import DATA_REGISTRY

__all__ = [
    'build_dataset',
    'NWBDataset'
]


def build_dataset(opt):
    """
    Build dataset from options.

    :param opt: dict. Configuration. It must contain:
        name - str. Dataset name.
        type - str. Dataset type.
    """

    opt = deepcopy(opt)
    dataset = DATA_REGISTRY.get(opt['type'])(opt)
    logger = get_root_logger()
    logger.info(f"Dataset [{dataset.__class__.__name__}] - {opt['name']} is built.")
    return dataset
