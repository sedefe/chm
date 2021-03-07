import numpy as np


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    Вычисляем моменты весовой функции с 0-го по max_s-ый на интервале [xl, xr]
    Весовая функция: p(x) = 1 / (x-a)^alpha / (b-x)^beta, причём гарантируется, что:
        1) 0 <= alpha < 1
        2) 0 <= beta < 1
        3) alpha * beta = 0

    :param max_s:   номер последнего момента
    :return:        список значений моментов
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

    :param m:   порядок погрешности
    :param L:   кратность шага
    :return:    оценки погрешностей s0 и s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оценка порядка главного члена погрешности по последовательным приближениям s0, s1 и s2 по правилу Эйткена
    Считаем, что погрешность равна R(h) = C*h^m + o(h^m)

    :param L:   кратность шага
    :return:    оценка порядка главного члена погрешности (m)
    """
    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    Интерполяционная квадратурная формула

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param xs:      узлы
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    m = moments(len(xs) - 1, x0, x1, **kwargs)
    raise NotImplementedError


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    Интерполяционная квадратурная формула типа Гаусса

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param n:       количество узлов
    :param kwargs:  параметры весовой функции (должны передаваться в moments)
    :return:        значение ИКФ
    """
    raise NotImplementedError


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    Составная квадратурная формула

    :param func:        интегрируемая функция
    :param x0, x1:      интервал
    :param n_intervals: количество интервалов
    :param n_nodes:     количество узлов на каждом интервале
    :param kwargs:      параметры весовой функции (должны передаваться в moments)
    :return:            значение СКФ
    """
    raise NotImplementedError


def integrate(func, x0, x1, tol):
    """
    Интегрирование с заданной точностью (error <= tol)

    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге и выбираем оптимальный размер шага
    Делаем так, пока оценка погрешности не уложится в tol

    :param func:    интегрируемая функция
    :param x0, x1:  интервал
    :param tol:     допуск
    :return:        значение интеграла, оценка погрешности
    """
    raise NotImplementedError
