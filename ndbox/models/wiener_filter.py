from sklearn import linear_model

from ndbox.utils.registry import MODEL_REGISTRY
from .ml_base_model import MLBaseModel


@MODEL_REGISTRY.register()
class WienerFilterRegression(MLBaseModel):
    """
    Class for the Wiener Filter Decoder. There are no parameters
    to set. This simply leverages the scikit-learn linear regression.
    """

    _default_params = {}

    def __init__(self, opt):
        super().__init__(opt, self._default_params)

    def fit(self, X, y):
        """
        Train Wiener Filter Decoder.

        :param X: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
        :param y: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.
        """

        self.model = linear_model.LinearRegression()
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict outcomes using trained Wiener Cascade Decoder.

        :param X: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.
        :return: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """

        y_pred = self.model.predict(X)
        return y_pred
