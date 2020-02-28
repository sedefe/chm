import numpy as np
from scipy.integrate import RK45, solve_ivp


class RungeKuttaCoeffs:
    def __init__(self, A, b, d=None):
        self.A = np.array(A)
        self.b = np.array(b)
        self.d = np.array(d) if d else None


#  classic Runge-Kutta ("The Runge-Kutta") method
rk4_coeffs = RungeKuttaCoeffs(
    A=[
        [0.0, 0.0, 0.0, 0.],
        [0.5, 0.0, 0.0, 0.],
        [0.0, 0.5, 0.0, 0.],
        [0.0, 0.0, 1.0, 0.],
    ],
    b=np.array([1, 2, 2, 1]) / 6
)

#  Dormand-Prince method
dopri_coeffs = RungeKuttaCoeffs(
    A=np.array([
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0]
    ]),
    b=np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
)


def euler(func, t0, y0, dt):
    """
    Explicit Euler method (no need to modify)
    """
    return y0 + dt * func(t0, y0)


def runge_kutta(func, t0, y0, dt, rk_coeffs: RungeKuttaCoeffs):
    """
    Explicit Runge-Kutta method with (A, b) coefficients
    Rewrite it without usage of built-in RK45()
    """
    A, b = rk_coeffs.A, rk_coeffs.b
    rk = RK45(func, t0, y0, t0 + dt)
    rk.h_abs = dt
    rk.step()
    return rk.y
