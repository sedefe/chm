import numpy as np
from scipy.interpolate import interp1d


class Interpolation:
    """
    Абстрактный класс для интерполяции

    :param xs: узлы интерполяции
    :param ys: значения в узлах
    """
    def __init__(self, xs, ys):
        """
        Метод инициализации
        """
        self.name = 'default interpolation'
        self.xs = xs
        self.ys = ys

    def __call__(self, xs):
        """
        Метод вызова
        """
        raise NotImplementedError


class LaGrange(Interpolation):
    """
    Интерполяция полиномом https://www.youtube.com/watch?v=Vppbdf-qtGU

    REQ: Метод инициализации не должен использовать numpy.polyfit
    REQ: Метод вызова должен возвращать значения в точках xs, не используя numpy.polyval
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'LaGrange'
        self.poly = np.polyfit(xs, ys, len(xs) - 1)

    def __call__(self, xs):
        return np.polyval(self.poly, xs)


class Spline1(Interpolation):
    """
    Интерполяция ломаной

    REQ: Метод инициализации не должен использовать scipy.interpolate.interp1d
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'Linear'
        self.interp = interp1d(xs, ys, kind='linear', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)


class Spline3(Interpolation):
    """
    Интерполяция кубическим сплайном

    REQ: Метод инициализации не должен использовать scipy.interpolate.interp1d
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'Cubic'
        self.interp = interp1d(xs, ys, kind='cubic', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)
