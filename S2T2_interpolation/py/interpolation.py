import numpy as np


def interpol(xs, ys):
    """
    Полиномиальная интерполяция по X и Y. Нужно вернуть коэффициенты интерполяционного многочлена
    """
    return np.polyfit(xs, ys, len(xs) - 1)
