import numpy as np
from scipy.special import legendre


class Approximation:
    """
    Аппроксимация по МНК на интервале [-1; 1]
    Нужно построить аппроксимацию в заданном семействе по значениям ys в точках xs
    В атрибуте coeffs должен лежать вектор коэффициентов
    :param dim: размерность семейства функций, которыми мы аппроксимируем
    """
    def __init__(self, name, xs, ys, dim):
        """
        Метод инициализации
        """
        self.name = name
        self.dim = dim
        self.xs = xs
        self.ys = ys
        self.coeffs = None

    def __call__(self, xs):
        """
        Метод вызова: выдать значения в точках xs
        """
        raise NotImplementedError


class Algebraic(Approximation):
    """
    Аппроксимация семейством алгебраических многочленов (1, x, x^2, ...)
    Метод инициализации не должен использовать numpy.polyfit()
    """
    def __init__(self, xs, ys, dim):
        super().__init__('algebraic', xs, ys, dim)
        self.coeffs = np.flip(np.polyfit(xs, ys, deg=dim-1))

    def __call__(self, xs):
        return np.polyval(np.flip(self.coeffs), xs)


class Legendre(Approximation):
    """
    Аппроксимация семейством многочлена Лежандра
    Методы инициализации и вызова не должны использовать класс numpy.polynomial.Legendre
    Коэффициенты многочленов Лежандра нужно получать без использования scipy.special.legendre
    """
    def __init__(self, xs, ys, dim):
        super().__init__('legendre', xs, ys, dim)
        self.leg_poly = np.polynomial.Legendre.fit(xs, ys, deg=dim-1)
        self.coeffs = self.leg_poly.coef

    def __call__(self, xs):
        return self.leg_poly(xs)


class Harmonic(Approximation):
    """
    Аппроксимация семейством гармоник
    Вид семейства описан здесь: https://ru.wikipedia.org/wiki/Тригонометрический_ряд_Фурье
    Метод инициализации не должен использовать вычисление ДПФ
    dim всегда нечётное (общее количество синусов и косинусов + константное слагаемое)
    """
    def __init__(self, xs, ys, dim):
        assert abs(ys[0] - ys[-1]) < 1e-6, 'not periodic function'
        assert dim % 2 == 1, 'dimension must be odd'

        dim = (dim-1) // 2
        super().__init__('harmonic', xs[:-1], ys[:-1], dim)

        self.fft = np.fft.fft(self.ys)

    def __call__(self, xs):
        n = len(self.xs)
        ys = np.zeros_like(xs) + self.fft[0] / n

        for k in range(1, self.dim+1):
            a = 2/n * self.fft[k] * (-1)**k         # Fourier magic
            ys += a * np.exp(1j*np.pi * k * xs)     # complex numbers magic
        ys = ys.real

        return ys
