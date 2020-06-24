import numpy as np
import sympy as sp
from sympy.abc import x


def is_polynome(function):
    """
    Проверяем, не полином ли наша функция
    """
    if function.as_poly(x):
        return True
    return False


def is_elementary(function):
    """
    Проверяем, что наша функция элементарная: exp(x), cos(x), ... или sqrt(x)
    """
    return len(function.args) == 1 or (function.func == sp.Pow and function.args[1] == sp.Number('1/2'))


def is_division(function):
    """
    Проверяем, что наша функция - это дробь g(x)/h(x)
    """
    if function.func == sp.Mul:
        if function.args[0].func == sp.Pow:
            if function.args[0].args[1] == sp.Number(-1):
                return True
        if function.args[1].func == sp.Pow:
            if function.args[1].args[1] == sp.Number(-1):
                return True
    return False


def get_division_args(function):
    """
    Получаем делимое g() и делитель h() для функции f(x) = g(x)/h(x)
    """
    if function.args[0].func == sp.Pow:
        return function.args[1], function.args[0].args[0]
    if function.args[1].func == sp.Pow:
        return function.args[0], function.args[1].args[0]


def estimate_abs(function, x0):
    """
    Оцениваем границы |f(x)| в районе точки x0
    Прикидываемся, что не знаем её точного значения в этой точке, и берём по 10% вниз и вверх
    """
    f0 = function.subs({x: x0})
    return 0.9*abs(f0), 1.1*abs(f0) + 0.1


def calc_elem_func(func, x0, eps):
    """
    Численно находим элементарные функции
    """

    # sqrt(x)
    if func == sp.Pow:
        w0 = max(1, x0)
        while True:
            w1 = (w0 + x0 / w0) / 2
            if abs(w1 - w0) < eps:
                return w1
            w0 = w1

    # exp(x)
    if func == sp.exp:
        s = 0
        k = 0
        u = 1
        while True:
            s += u
            k += 1
            if abs(u) < eps:
                return s
            u *= x0 / k

    # cos(x)
    if func == sp.cos:
        x1 = sp.Mod(x0 + sp.pi, 2 * sp.pi) - sp.pi
        if sp.Abs(x1) <= sp.pi / 4:
            s = 0
            k = 0
            u = 1
            while True:
                s += u
                k += 2
                if abs(u) < eps:
                    return s
                u *= - x1 ** 2 / (k * (k - 1))
        elif sp.Abs(x1) <= 3 * sp.pi / 4:
            return calc_elem_func(sp.sin, sp.pi / 2 - x0, eps)
        else:
            return -calc_elem_func(sp.cos, x1 - sp.pi, eps)

    raise NotImplementedError(f'estimation of "{func}" yet not implemented')


def calc(function, x0, eps):
    # Сначала проверяем, не полином ли это
    if is_polynome(function):
        return function.subs({x: x0})

    # Потом проверяем, не элементарная ли она
    if is_elementary(function):
        g = function.args[0]
        f_du = function.diff()
        _, b = estimate_abs(f_du, x0)
        u = calc(g, float(x0), eps / b)
        return calc_elem_func(function.func, u, eps)

    # Если дошли досюда, то имеем дело со сложной функцией
    if len(function.args) == 2:

        # f(x) = g(x) + h(x)
        if function.func == sp.Add:
            g, h = function.args

            u = calc(g, x0, eps / 2)
            v = calc(h, x0, eps / 2)

            return u + v

        # f(x) = g(x) ** k
        if function.func == sp.Pow:
            k = function.args[1]
            if isinstance(k, sp.Integer):
                g = function.args[0]
                derivative = k * sp.Pow(g, k - 1)
                b_u0, b_u1 = estimate_abs(derivative, x0)

                u = calc(g, x0, eps / b_u1)
                return u ** k
            raise NotImplementedError(f'calculating f(x) ** {k} it too hard for me yet')

        # f(x) = g(x) * h(x)
        if function.func == sp.Mul:
            raise NotImplementedError(function.func)

        raise NotImplementedError(function.func)

    raise NotImplementedError(f'{len(function.args)}-term operations not yet supported')
