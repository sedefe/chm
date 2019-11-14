import numpy as np


def func(x):
    """
    this method should implement VECTORIZED target function
    """
    return np.exp(np.cos(x))


def interpol(X, Y):
    """
    this method should find polynomial interpolation
    :param X: X-values (1xN)
    :param Y: Y-values (1xN)
    :return: coefficients of N-1-degree polynome P (1xN)
    """
    return np.polyfit(X, Y, len(X)-1)
