import numpy as np
import scipy


class RKScheme:
    """
    Runge-Kutta scheme (A,b) of order p
    """
    def __init__(self, name, A, b, p):
        self.name = name
        self.A = np.array(A)
        self.b = np.array(b)
        self.p = p


class EmbeddedRKScheme(RKScheme):
    """
    Embedded Runge-Kutta scheme (A,b,e) of orders p(q)
    p is the order of RK scheme (A,b)
    q is the order of RK scheme (A,b+e)
    """
    def __init__(self, name, A, b, e, p, q):
        super().__init__(name, A, b, p)
        self.e = np.array(e)
        self.q = q


#  classic Runge-Kutta ("The Runge-Kutta") method
rk4_coeffs = RKScheme(
    name='RK4',
    A=[
        [0.0, 0.0, 0.0, 0.],
        [0.5, 0.0, 0.0, 0.],
        [0.0, 0.5, 0.0, 0.],
        [0.0, 0.0, 1.0, 0.],
    ],
    b=np.array([1, 2, 2, 1]) / 6,
    p=4,
)

#  Dormand-Prince method
dopri_coeffs = EmbeddedRKScheme(
    name='DoPri5(4)',
    A=np.array([
        [0,          0,           0,          0,        0,           0,     0],
        [1/5,        0,           0,          0,        0,           0,     0],
        [3/40,       9/40,        0,          0,        0,           0,     0],
        [44/45,      -56/15,      32/9,       0,        0,           0,     0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0,           0,     0],
        [9017/3168,  -355/33,     46732/5247, 49/176,   -5103/18656, 0,     0],
        [35/384,     0,           500/1113,   125/192,  -2187/6784,  11/84, 0],
    ]),
    b=np.array([35/384,    0, 500/1113, 125/192,  -2187/6784,   11/84,   0]),
    e=np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]),
    p=5,
    q=4,
)

