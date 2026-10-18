import numpy as np

def mean_absolute_error(Y, Y_pred):
    """Calculate the mean absolute error"""

    m = Y.shape[0]
    error = Y_pred - Y
    absolute_error = np.abs(error)
    mae = np.sum(absolute_error) / m

    return float(mae)

def root_mean_squared_error(Y, Y_pred):
    """Calculate the root mean square error"""

    m = Y.shape[0]
    error = Y_pred - Y
    sqe = error**2
    rmse = np.sqrt(np.sum(sqe)/m)

    return float(rmse)

def r_squared(Y, Y_pred):
    """Calculate the Coefficient of Determination"""

    m = Y.shape[0]
    y_mean = np.mean(Y)
    error = Y_pred - Y
    mean_error = Y - y_mean
    residual_sum = np.sum(error**2)
    total_sum = np.sum(mean_error**2)
    coeff = 1 - (residual_sum / total_sum)

    return float(coeff)