import numpy as np
from utils.utils import call_counter


class ODE:
    def __init__(self, y0):
        self.y0 = y0

    @call_counter
    def __call__(self, t, y):
        raise NotImplementedError

    def __getitem__(self, t):
        raise NotImplementedError

    def get_call_counter(self):
        return self.__call__.calls

    def clear_call_counter(self):
        self.__call__.__dict__['calls'] = 0


class Harmonic(ODE):
    """
    y_1' =  a*y_2
    y_2' = -b*y_1
    """
    def __init__(self, y0, a, b):
        self.a = a
        self.b = b
        self.w = np.sqrt(a*b)
        super(Harmonic, self).__init__(y0)

    @call_counter
    def __call__(self, t, y):
        return np.array([self.a*y[1],
                         -self.b*y[0]])

    def __getitem__(self, t):
        return np.array([
            [self.y0[0],  self.y0[1] * self.a / self.w],
            [self.y0[1], -self.y0[0] * self.b / self.w],
        ]) @ np.array([np.cos(self.w * t),
                       np.sin(self.w * t)])


class HarmExp(ODE):
    """
    y_1 = exp(cos(t))
    y_2 = exp(sin(t))
    y_1' = -y_1 * ln(y_2)
    y_2' =  y_2 * ln(y_1)
    """
    def __init__(self):
        super(HarmExp, self).__init__([np.exp(1), 1])

    @call_counter
    def __call__(self, t, y):
        return np.array([-y[0] * np.log(y[1]),
                         y[1] * np.log(y[0])])

    def __getitem__(self, t):
        return np.exp(np.array([np.cos(t), np.sin(t)]))
