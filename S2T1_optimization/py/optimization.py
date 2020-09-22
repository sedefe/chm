import numpy as np
from utils.utils import call_counter


@call_counter
def func(A, b, x):
    """
    целевая функция y = 1/2*x.T*A*x + b.T*x
    """
    return (1 / 2 * x.T @ A @ x + b.T @ x).item()


def mngs(A, b, x0, eps):
    """
    метод наискорейшего градиентного спуска для задачи y = 1/2*x.T*A*x + b.T*x
    :return: list of x, list of y
    """
    x1 = np.linalg.solve(A, -b)
    X = [(x1*p + x0*(1-p)) for p in np.linspace(0, 1-eps, 10)]
    Y = [func(A, b, x) for x in X]
    return X, Y


def mps(A, b, x0, eps):
    """
    метод покоординатного спуска для задачи y = 1/2*x.T*A*x + b.T*x
    :return: list of x, list of y
    """
    return mngs(A, b, x0, eps/2)


def newton(A, b, x0, eps):
    """
    метод Ньютона для задачи y = 1/2*x.T*A*x + b.T*x (гессиан равен A)
    :return: list of x, list of y
    """
    return mngs(A, b, x0, eps/4)
