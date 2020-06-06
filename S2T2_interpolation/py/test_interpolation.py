import pytest
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from S2T2_interpolation.py import interpolation
from utils.utils import get_accuracy

default_student = 0


@pytest.mark.parametrize("student", [default_student])
def test_interpolation(student):
    print(f'running interpolation test for student #{student}')
    n = 5
    m = 10*n
    a, b = -1, 3   # change to what you prefer

    xs_eq = np.linspace(a, b, n+1)
    xs_cheb = 1/2 * ((b - a) * np.cos(np.pi * (2*np.arange(0, n + 1) + 1) / (2*(n + 1))) + (b + a))
    xs_dense = sorted([*np.linspace(a, b, m), *xs_eq, *xs_cheb])
    ys_dense = interpolation.func(xs_dense)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs_dense, ys_dense, 'k-', label='actual')
    styles = ['b.:', 'm.:']
    labels = ['interp-eq', 'interp-cheb']

    for (i, xs) in enumerate([xs_eq, xs_cheb]):
        Y0 = interpolation.func(xs)  # here's your function (change met2.func())

        P = interpolation.interpol(xs, Y0)    # here's your interpolation (change met2.interpol())
        assert len(P) == n+1, f'polynome length should be {n+1}'
        Y2 = np.polyval(P, xs_dense)

        ax1.plot(xs_dense, Y2, styles[i], label=labels[i])
        ax2.plot(xs_dense, get_accuracy(ys_dense, Y2), styles[i], label=labels[i])

    ys_spline = CubicSpline(xs_eq, interpolation.func(xs_eq))(xs_dense)

    ax1.set_title('y(x)')
    ax1.plot(xs_dense, ys_spline, 'c', label='spline')
    ax1.legend()

    ax2.set_title('accuracy')
    ax2.plot(xs_dense, get_accuracy(ys_dense, ys_spline), 'c', label='spline')
    ax2.legend()

    plt.show()
