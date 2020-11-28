import numpy as np

def binary_crossentropy(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -(1 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return np.squeeze(cost).astype(np.float32)


def mean_squared_error(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    cost = (1 / m) * np.sum( (Y - Y_hat)**2 )
    return np.squeeze(cost).astype(np.float32)