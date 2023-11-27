import os
import joblib

from ndbox.utils import get_root_logger


class MLBaseModel:
    """
    Machine learning base model.
    """

    def __init__(self, opt, params):
        logger = get_root_logger()
        for key, val in params.items():
            if opt.get(key) is None:
                opt[key] = val
                logger.info(f"Parameter '{key}' is not specified, "
                            f"default values '{val}' are used.")
        self.opt = opt

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def load(self, path):
        """
        Load model.

        :param path: str. The path of models to be loaded.
        """

        logger = get_root_logger()
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' File not found!")
        self.model = joblib.load(path)
        logger.info(f"Loading {self.__class__.__name__} model from '{path}'")

    def save(self, path):
        """
        Save model.

        :param path: str. The path of models to be saved.
        """

        joblib.dump(self.model, path)
