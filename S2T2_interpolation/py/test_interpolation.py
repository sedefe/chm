import pytest
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from S2T2_interpolation.py import interpolation
from utils.utils import get_log_error

default_student = 0


@pytest.mark.parametrize("student", [default_student])
def test_interpolation(student):
    print(f'running interpolation test for student #{student}')
    N = 5
    M = 10*N
    A = -1  # change to what you prefer
    B = 3   # change to what you prefer

    xs_eq = np.linspace(A, B, N+1)
    xs_cheb = 1/2 * ((B-A) * np.cos(np.pi * (2*np.arange(0, N+1) + 1) / (2*(N + 1))) + (B+A))
    xs_dense = sorted([*np.linspace(A, B, M), *xs_eq, *xs_cheb])
    ys_dense = interpolation.func(xs_dense)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(xs_dense, ys_dense, 'k-', label='actual')
    styles = ['b.:', 'm.:']
    labels = ['interp-eq', 'interp-cheb']

    xss = [xs_eq, xs_cheb]
    for (i, xs) in enumerate(xss):
        Y0 = interpolation.func(xs)  # here's your function (change met2.func())

        P = interpolation.interpol(xs, Y0)    # here's your interpolation (change met2.interpol())
        assert len(P) == N+1, f'polynome length should be {N+1}'
        Y2 = np.polyval(P, xs_dense)

        ax1.plot(xs_dense, Y2, styles[i], label=labels[i])
        ax2.plot(xs_dense, get_log_error(ys_dense, Y2), styles[i], label=labels[i])

    ys_spline = CubicSpline(xs_eq, interpolation.func(xs_eq))(xs_dense)

    ax1.set_title('Y(X)')
    ax1.plot(xs_dense, ys_spline, 'c', label='spline')
    ax1.legend()

    ax2.set_title('accuracy')
    ax2.plot(xs_dense, get_log_error(ys_dense, ys_spline), 'c', label='spline')
    ax2.legend()

    plt.show()
