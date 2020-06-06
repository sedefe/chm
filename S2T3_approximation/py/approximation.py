import numpy as np
from enum import Enum


class ApproxType(Enum):
    algebraic = 0
    legendre = 1
    harmonic = 2


def func(x):
    """
    Целевая функция (векторизованная): должна принимать линейный массив и возвращать линейный массив
    """
    return np.exp(np.cos(x))


def approx(x0, y0, x1, approx_type: ApproxType, dim):
    """
    Аппроксимация на интервале [-1; 1]
    Нужно по значениям y0 в точках x0 выдать значения в точках x1
    :param approx_type:
        0 - алгебраические многочлены (1, x, x^2, ...)
        1 - многочлены Лежандра
        2 - гармоники
    :param dim: размерность семейства функций, которыми мы аппроксимируем
    :return y1: значения в точках x1
    :return a: вектор коэффициентов длины dim
    :return P: (для approx_type 0 и 1) коэффициенты аппроксимационного многочлена P
    """
    if approx_type is ApproxType.algebraic:
        return func(x1) + np.random.randn(len(x1)) * 1e-10, \
               np.eye(1, dim)[0], \
               np.eye(1, dim)[0]
    if approx_type is ApproxType.legendre:
        return func(x1) + np.random.randn(len(x1)) * 1e-10, \
               np.eye(1, dim)[0], \
               np.eye(1, dim)[0]
    raise Exception(f'approximation of type {approx_type} not supported yet')
