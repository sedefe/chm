import numpy as np
from enum import Enum, auto


class Approximation:
    """
    Аппроксимация по МНК на интервале [-1; 1]
    Нужно по значениям ys0 в точках xs0 выдать значения в точках xs1
    :param dim: размерность семейства функций, которыми мы аппроксимируем
    :return ys1: значения в точках xs1
    :return a: (для ApproxType.algebraic и ApproxType.legendre) вектор коэффициентов длины dim
    :return poly: (для ApproxType.algebraic и ApproxType.legendre) коэффициенты аппроксимационного многочлена p
    """

    def __init__(self, xs, ys, dim):
        """
        Метод инициализации
        """
        self.name = 'default approximation'
        self.dim = dim
        self.xs = xs
        self.ys = ys

    def __call__(self, xs):
        """
        Метод вызова
        """
        raise NotImplementedError


class Algebraic(Approximation):
    """
    Аппроксимация семейством алгебраических полиномов (1, x, x^2, ...)
    Метод инициализации не должен использовать numpy.polyfit()
    """
    def __init__(self, xs, ys, dim):
        super().__init__(xs, ys, dim)
        self.name = 'algebraic'
        self.poly = np.polyfit(xs, ys, deg=dim-1)

    def __call__(self, xs):
        ys = np.polyval(self.poly, xs)
        return ys

    def get_poly(self):
        return self.poly


class Legendre(Approximation):
    """
    Аппроксимация семейством полиномов Лежандра
    Метод инициализации не должен использовать numpy.polyfit()
    """
    def __init__(self, xs, ys, dim):
        super().__init__(xs, ys, dim)
        self.name = 'legendre'
        self.poly = np.polyfit(xs, ys, deg=dim-1)

    def __call__(self, xs):
        ys = np.polyval(self.poly, xs)
        return ys

    def get_poly(self):
        return self.poly


class Harmonic(Approximation):
    """
    Аппроксимация семейством гармоник
    Вид семейства описан здесь: https://ru.wikipedia.org/wiki/Тригонометрический_ряд_Фурье
    Метод инициализации не должен использовать numpy.fft.fft()
    Метод вызова не должен использовать numpy.exp()
    """
    def __init__(self, xs, ys, dim):
        assert abs(ys[0] - ys[-1]) < 1e-6, 'not periodic function'
        assert dim % 2 == 1, 'dimension must be odd'

        dim = (dim-1) // 2
        super().__init__(xs[:-1], ys[:-1], dim)
        self.name = 'harmonic'

        self.fft = np.fft.fft(self.ys)

    def __call__(self, xs):
        n = len(self.xs)
        ys1 = np.zeros_like(xs) + self.fft[0] / n
        for k in range(1, self.dim+1):
            ys1 += -np.exp(-1j * np.pi * k * xs) * self.fft[k] / (n/2)
        ys1 = ys1.real

        return ys1
