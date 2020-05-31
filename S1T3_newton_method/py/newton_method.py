import numpy as np
import sympy as sp
from sympy.abc import x, y


def solve_scalar(f: sp.Expr, x0, tol):
    """
    Решаем скалярное уравнение y=f(x), начиная с точки x0
    Производную можно получать так: f.diff()
    Подставить значения вместо переменных: float(f.subs(x, x0))
    returns: list of x, list of y
    """
    raise NotImplementedError
    return xs, ys


def solve_plane(f: sp.Matrix, x0, y0, tol):
    """
    Решаем систему двух алгебраических уравнений {f1(x,y) = 0, f2(x,y) = 0}, начиная с точки (x0,y0)
    Якобиан можно получать так: f.jacobian([x, y])
    returns: list of x, list of y, list of np.linalg.norm(z)
    """
    raise NotImplementedError
    return xs, ys, zs
