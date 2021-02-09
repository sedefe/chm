import numpy as np


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    Вычисляем моменты весовой функции с 0-го по max_s-ый на интервале [xl, xr]
    Весовая функция: p(x) = 1 / (x-a)^alpha / (b-x)^beta
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    Оценка погрешности последовательных приближений s0 и s1 по правилу Рунге

    :param m: порядок погрешности
    :param L: кратность шага
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оценка погрешности последовательных приближений s0, s1 и s2 по правилу Эйткена

    :param L: кратность шага
    """
    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    Интерполяционная квадратурная формула

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param xs:      узлы
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    """
    raise NotImplementedError


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    Интерполяционная квадратурная формула типа Гаусса

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param n:       количество узлов
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    """
    raise NotImplementedError


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    Составная квадратурная формула

    :param func:        интегрируемая функция
    :param x0, x1:      интервал
    :param n_intervals: количество интервалов
    :param n_nodes:     количество узлов на каждом интервале
    """
    raise NotImplementedError


def integrate(func, x0, x1, tol):
    """
    Интегрирование с заданной точностью (error <= tol) на интервале [x0, x1]

    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге

    Нужно вернуть:
        - значение интеграла
        - оценку погрешности
    """
    raise NotImplementedError
