import pytest
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from s2_interpolation.py import interpolation

default_student = 0


@pytest.mark.parametrize("student", [default_student])
def test_interpolation(student):
    print(f'running interpolation test for student #{student}')
    N = 5
    M = 10*N
    A = -1  # change to what you prefer
    B = 3   # change to what you prefer

    X_eq = np.linspace(A, B, N+1)
    X_cheb = 1/2 * ((B-A) * np.cos(np.pi * (2*np.arange(0, N+1) + 1) / (2*(N + 1))) + (B+A))
    X_dense = sorted([*np.linspace(A, B, M), *X_eq, *X_cheb])
    Y_dense = interpolation.func(X_dense)

    plt.figure(1)
    plt.plot(X_dense, Y_dense, 'k-', label='actual')
    styles = ['b.:', 'm.:']
    labels = ['interp-eq', 'interp-cheb']

    Xs = [X_eq, X_cheb]
    for (i, X) in enumerate(Xs):
        Y0 = interpolation.func(X)  # here's your function (change met2.func())

        P = interpolation.interpol(X, Y0)    # here's your interpolation (change met2.interpol())
        assert len(P) == N+1, f'polynome length should be {N+1}'
        Y2 = np.polyval(P, X_dense)

        plt.figure(1)
        plt.plot(X_dense, Y2, styles[i], label=labels[i])
        plt.figure(2)
        plt.plot(X_dense, np.log10(np.abs(Y_dense - Y2)), styles[i], label=labels[i])

    Y_spline = CubicSpline(X_eq, interpolation.func(X_eq))(X_dense)

    plt.figure(1)
    plt.title('Y(X)')
    plt.plot(X_dense, Y_spline, 'c', label='spline')
    plt.legend()

    plt.figure(2)
    plt.title('log error')
    plt.plot(X_dense, np.log10(np.abs(Y_dense - Y_spline)), 'c', label='spline')
    plt.legend()

    plt.show()
