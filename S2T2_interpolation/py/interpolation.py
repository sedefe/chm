import numpy as np


def func(x):
    """
    Целевая функция (векторизовання)
    """
    return np.exp(np.cos(x))


def interpol(X, Y):
    """
    Полиномиальная интерполяция по X и Y. Нужно вернуть коэффициенты интерполяционного многочлена
    """
    return np.polyfit(X, Y, len(X)-1)
