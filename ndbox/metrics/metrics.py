import numpy as np

from ndbox.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_R2(y_true, y_pred, **kwargs):
    """
    Function to calculate R2.

    :param y_true: The true outputs (a matrix of size
        number of examples x number of outputs).
    :param y_pred: The predicted outputs (a matrix of
        size number of examples x number of outputs).
    :return: An array of R2s for each output.
    """

    R2_list = []
    for i in range(y_true.shape[1]):
        y_mean = np.mean(y_true[:, i])
        R2 = 1 - (np.sum((y_pred[:, i] - y_true[:, i]) ** 2) /
                  np.sum((y_true[:, i] - y_mean) ** 2))
        R2_list.append(R2)
    R2_array = np.array(R2_list)
    return R2_array


@METRIC_REGISTRY.register()
def calculate_cc(y_true, y_pred, **kwargs):
    cc_list = []
    for i in range(y_true.shape[1]):
        cc = np.corrcoef(y_true[:, i].T, y_pred[:, i].T)[0, 1]
        cc_list.append(cc)
    cc_array = np.array(cc_list)
    return cc_array


@METRIC_REGISTRY.register()
def calculate_mse(y_true, y_pred, **kwargs):
    mse_list = []
    for i in range(y_true.shape[1]):
        mse = np.sum((y_true[:, i] - y_pred[:, i]) ** 2) / y_true.shape[0]
        mse_list.append(mse)
    mse_array = np.array(mse_list)
    return mse_array
