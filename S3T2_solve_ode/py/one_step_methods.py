import numpy as np
from copy import copy
from scipy.integrate import RK45, solve_ivp

import S3T2_solve_ode.py.coeffs_collection as collection


class OneStepMethod:
    def __init__(self, **kwargs):
        self.name = 'default_method'
        self.p = None  # order
        self.__dict__.update(**kwargs)

    def step(self, func, t, y, dt):
        """
        make a step: t => t+dt
        """
        raise NotImplementedError


class EulerMethod(OneStepMethod):
    """
    Explicit Euler method (no need to modify)
    order is 1
    """
    def __init__(self):
        super().__init__(name='Euler', p=1)

    def step(self, func, t, y, dt):
        return y + dt * func(t, y)


class RungeKuttaMethod(OneStepMethod):
    """
    Explicit Runge-Kutta method with (A, b) coefficients
    Rewrite step() method without usage of built-in RK45()
    """
    def __init__(self, coeffs: collection.RKScheme):
        super().__init__(**coeffs.__dict__)

    def step(self, func, t, y, dt):
        A, b = self.A, self.b
        rk = RK45(func, t, y, t + dt)
        rk.h_abs = dt
        rk.step()
        return rk.y


class EmbeddedRungeKuttaMethod(RungeKuttaMethod):
    """
    Embedded Runge-Kutta method with (A, b, e) coefficients:
    y1 = RK(func, A, b)
    y2 = RK(func, A, d), where d = b+e
    embedded_step() method should return approximation (y1) AND approximations difference (dy = y2-y1)
    """
    def __init__(self, coeffs: collection.EmbeddedRKScheme):
        super().__init__(coeffs=coeffs)

    def embedded_step(self, func, t, y, dt):
        A, b, e = self.A, self.b, self.e
        c = np.sum(A, axis=0)
        raise NotImplementedError
        return y1, dy
