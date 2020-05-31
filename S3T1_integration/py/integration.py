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
    Оцениваем погрешность приближений s0 и s1 по правилу Рунге
    m - порядок погрешности
    L - кратность s0 и s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    Оцениваем сходимость по правилу Эйткена
    s0, s1, s2: последовательный приближения интеграла
    """
    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    ИКФ
    func: интегрируемая функция
    x0, x1: интервал
    xs: узлы
    **kwargs должны передаваться в moments()
    """
    raise NotImplementedError


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    ИКФ типа Гаусса
    func: интегрируемая функция
    x0, x1: интервал
    n: количество узлов
    **kwargs должны передаваться в moments()
    """
    raise NotImplementedError


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    СКФ
    func: интегрируемая функция
    x0, x1: интервал
    n_intervals: количество интервалов
    n_nodes: количество узлов на каждом интервале
    """
    raise NotImplementedError


def integrate(func, x0, x1, tol):
    """
    Интегрируем с заданной точностью (error <= tol)
    Оцениваем сходимость по Эйткену, потом оцениваем погрешность по Рунге
    Нужно вернуть:
        - значение интеграла
        - оценку погрешности
    """
    raise NotImplementedError
