import numpy as np
from scipy.interpolate import interp1d


class Interpolation:
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
    Интерполяционный полином https://www.youtube.com/watch?v=Vppbdf-qtGU
    Метод инициализации не должен использовать numpy.polyfit()
    Метод вызова должен возвращать значения в точках xs, не используя numpy.polyval()
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'LaGrange'
        self.poly = np.polyfit(xs, ys, len(xs) - 1)

    def __call__(self, xs):
        return np.polyval(self.poly, xs)


class Spline1(Interpolation):
    """
    Ломаная
    Метод инициализации не должен использовать scipy.interpolate.interp1d()
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'Linear'
        self.interp = interp1d(xs, ys, kind='linear', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)


class Spline3(Interpolation):
    """
    Кубический сплайн
    Метод инициализации не должен использовать scipy.interpolate.interp1d()
    """
    def __init__(self, xs, ys):
        super().__init__(xs, ys)
        self.name = 'Cubic'
        self.interp = interp1d(xs, ys, kind='cubic', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)
