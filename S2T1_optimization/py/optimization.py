import numpy as np
from utils.utils import call_counter


@call_counter
def func(A, b, x):
    """
    this method implements target function y = 1/2*x.T*A*x + b.T*x, leave it unchanged
    """
    return (1 / 2 * x.T @ A @ x + b.T @ x).item()


def mngs(A, b, x0, eps):
    """
    this method should numerically find min(y),
    where y = 1/2*x.T*A*x + b.T*x
    :param A: matrix NxN
    :param b: matrix Nx1
    :param x0: matrix Nx1
    :param eps: accuracy (see test_met1())
    :return: list of x, list of y
    """
    # this is dummy code, you should implement your own
    x1 = np.linalg.solve(A, -b)
    X = [(x1*p + x0*(1-p)) for p in np.linspace(0, 1-eps, 10)]
    Y = [func(A, b, x) for x in X]
    return X, Y


def mps(A, b, x0, eps):
    """
    this method should numerically find min(y),
    where y = 1/2*x.T*A*x + b.T*x
    :param A: matrix NxN
    :param b: matrix Nx1
    :param x0: matrix Nx1
    :param eps: accuracy (see test_met1())
    :return: list of x, list of y
    """
    # this is dummy code, you should implement your own
    return mngs(A, b, x0, eps/2)
