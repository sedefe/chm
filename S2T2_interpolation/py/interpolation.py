import numpy as np
from scipy.interpolate import interp1d


class Interpolation:
    def __init__(self, xs, ys):
        """
        Метод инициализации
        """
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
    """
    def __init__(self, xs, ys):
        """
        Метод инициализации не должен использовать numpy.polyfit()
        """
        super().__init__(xs, ys)
        self.poly = np.polyfit(xs, ys, len(xs) - 1)

    def __call__(self, xs):
        """
        Метод вызова должен возвращать значения в точках xs, не используя numpy.polyval()
        """
        return np.polyval(self.poly, xs)


class Spline1(Interpolation):
    """
    Ломаная
    """
    def __init__(self, xs, ys):
        """
        Метод инициализации не должен использовать scipy.interpolate.interp1d()
        """
        super().__init__(xs, ys)
        self.interp = interp1d(xs, ys, kind='linear', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)


class Spline3(Interpolation):
    """
    Кубический сплайн
    """
    def __init__(self, xs, ys):
        """
        Метод инициализации не должен использовать scipy.interpolate.interp1d()
        """
        super().__init__(xs, ys)
        self.interp = interp1d(xs, ys, kind='cubic', fill_value='extrapolate')

    def __call__(self, xs):
        return self.interp(xs)
