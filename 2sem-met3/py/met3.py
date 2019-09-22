import numpy as np
from enum import Enum


class ApproxType(Enum):
    algebraic = 0
    legendre = 1
    harmonic = 2


def func(x):
    """
    this method should implement VECTORIZED target function
    """
    return np.exp(np.cos(x))


def approx(X0, Y0, X1, approx_type: ApproxType, dim):
    """
    this method should perform approximation on [-1; 1] interval
    :param X0: X-values (1 x N0)
    :param Y0: Y-values (1 x N0)
    :param X1: approximation points (1 x N1)
    :param approx_type:
        0 - algebraic polynomes (1, x, x^2, ...)
        1 - legendre polynomes
        2 - harmonic
    :param dim: dimension
    :return Y1: approximated Y-values (1 x N1)
    :return a: vector (1 x dim) of approximation coefficients
    :return P: (for approx_type 0 and 1) coefficients of approximation polynome P (1 x dim)
    """
    if approx_type is ApproxType.algebraic:
        return func(X1) + np.random.randn(len(X1)) * 1e-10, \
               np.eye(1, dim)[0], \
               np.eye(1, dim)[0]
    if approx_type is ApproxType.legendre:
        return func(X1) + np.random.randn(len(X1)) * 1e-10, \
               np.eye(1, dim)[0], \
               np.eye(1, dim)[0]
    raise Exception(f'approximation of type {approx_type} not supported yet')
