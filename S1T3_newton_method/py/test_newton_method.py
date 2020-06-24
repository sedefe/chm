import pytest
import numpy as np
import sympy as sp
from sympy.abc import x, y
from sympy.plotting.plot import List2DSeries
import matplotlib.pyplot as plt

from S1T3_newton_method.py.newton_method import solve_scalar, solve_plane


@pytest.mark.parametrize('f,x0',
                         [
                             (x - sp.sin(x) - 0.25, 3.0),
                             (2 ** x * (x - 1) ** 2 - 2, -2.5),
                             (sp.tan(.5 * x + .2) - x ** 2, 2.0),
                         ]
                         )
def test_solve_scalar(f, x0):
    """
    Проверяем решение скалярных уравнений
    """
    interval = -10, 10

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f)

    # plot reference line
    xs_ref = np.linspace(*interval, 1001)
    ys_ref = [f.subs(x, xx) for xx in xs_ref]
    ax1.plot(xs_ref, ys_ref, 'c-')
    ax1.plot(interval, [0, 0], 'k-')

    # get iterations
    tol = 1e-9
    xs, ys = solve_scalar(f, x0, tol)
    assert abs(f.subs(x, xs[-1])) < tol

    # plot iterations
    for i in range(len(xs) - 1):
        ax1.plot([xs[i], xs[i + 1]], [ys[i], 0], 'r:')
        ax1.plot([xs[i + 1], xs[i + 1]], [0, ys[i + 1]], 'g:')
    ax1.plot(xs, ys, 'b*')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # plot accuracy
    ax2.plot(-np.log10(np.abs(ys)), 'b.:')
    ax2.set_xlabel('N step')
    ax2.set_ylabel('Accuracy')
    plt.show()


@pytest.mark.parametrize('f,x0,y0',
                         [
                             (sp.Matrix([sp.sin(x + 1) - y - 1.2, 2 * x + sp.cos(y) - 2]), 0, 0),
                             (sp.Matrix([sp.cos(y + 0.5) - x - 2, sp.sin(x) - 2 * y - 1]), 0, 0),
                             (sp.Matrix([sp.cos(x) + y - 1.2, 2 * x - sp.sin(y - 0.5) - 2]), 0, 0),
                         ])
def test_solve_plane(f, x0, y0):
    """
    Проверяем решение векторных уравнений
    """
    # get iterations
    tol = 1e-9
    xs, ys, zs = solve_plane(f, x0, y0, tol)
    f_eval = f.subs({x: xs[-1], y: ys[-1]})
    assert np.linalg.norm([float(f_eval[0]), float(f_eval[1])]) < tol

    # plot f() lines
    # p = sp.plot(show=False, backend=sp.plotting.plot_backends['matplotlib'])
    p = sp.plot(show=False)
    p.extend(sp.plot_implicit(f[0], depth=1, line_color='k', show=False))
    p.extend(sp.plot_implicit(f[1], depth=1, line_color='k', show=False))

    # plot iterations
    traj = List2DSeries(xs, ys)
    traj.line_color = 'm'
    p.append(traj)
    p.title = f'{f[0]}\n{f[1]}'
    p.xlabel = 'x'
    p.ylabel = 'y'
    p.show()

    # plot accuracy
    plt.figure()
    plt.plot(-np.log10(np.abs(zs)), 'b.:')
    plt.suptitle(p.title)
    plt.xlabel('N step')
    plt.ylabel('Accuracy')
    plt.show()
