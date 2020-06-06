import pytest
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from S2T2_interpolation.py import interpolation
from utils.utils import get_accuracy


@pytest.mark.parametrize('fname,func', [
    ['exp(sin(x))', lambda x: np.exp(np.sin(x))],
    ['cos(exp(x))', lambda x: np.cos(np.exp(x))],
    ['x^4', lambda x: x**4],
])
def test_interpolation(fname, func: callable):
    n = 15
    k_dense = 10
    m = k_dense * n
    a, b = -1, 3

    xs_eq = np.linspace(a, b, n)
    xs_cheb = 1/2 * ((b - a) * np.cos(np.pi * (np.arange(n) + 1/2) / n) + (b + a))
    xs_dense = np.array(sorted([*np.linspace(a, b, m), *xs_eq, *xs_cheb]))
    ys_dense = func(xs_dense)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs_dense, ys_dense, 'k-', label='actual')
    colors = 'bm'
    labels = ['interp-eq', 'interp-cheb']

    for i, xs in enumerate([xs_eq, xs_cheb]):
        ys = func(xs)

        poly = interpolation.interpol(xs, ys)
        assert len(poly) == n, f'polynome length should be {n}'
        ys_dense_num = np.polyval(poly, xs_dense)

        ax1.plot(xs_dense, ys_dense_num, f'{colors[i]}:', label=labels[i])
        ax1.plot(xs, ys, f'{colors[i]}.')
        ax2.plot(xs_dense, get_accuracy(ys_dense, ys_dense_num), f'{colors[i]}-', label=labels[i])

    ys_spline = CubicSpline(xs_eq, func(xs_eq))(xs_dense)

    ax1.set_title(f'{fname}')
    ax1.plot(xs_dense, ys_spline, 'c:', label='spline')
    ax1.legend()

    ax2.set_title('accuracy')
    ax2.plot(xs_dense, get_accuracy(ys_dense, ys_spline), 'c-', label='spline')
    ax2.legend()

    plt.show()
