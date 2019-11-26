import pytest
import numpy as np
import matplotlib.pyplot as plt

from S2T3_approximation.py import approximation


def test_approximation():
    print(f'running approximation test')
    N = 5
    dim = 3
    M = 101

    x0 = np.linspace(-1, 1, N)
    x1 = np.linspace(-1, 1, M)

    # here's your function (change met3.func())
    y0 = approximation.func(x0)
    y1 = approximation.func(x1)

    # here's your approximation (change met3.approx())
    y_algpoly, _, p_algpoly = approximation.approx(x0, y0, x1, approximation.ApproxType.algebraic, dim)
    assert(len(p_algpoly) == dim), f'polynome length should be {dim}'
    assert(all(abs(y1 - y_algpoly) < 1)), 'algebraic polynome approximation is too bad'

    y_legpoly, _, p_legpoly = approximation.approx(x0, y0, x1, approximation.ApproxType.legendre, dim)
    assert(len(p_legpoly) == dim), f'polynome length should be {dim}'
    assert(all(abs(y1 - y_legpoly) < 1)), 'legendre polynome approximation is too bad'
    # assert(all(abs(p_algpoly - p_legpoly) < 1e-3))

    plt.figure(1)
    plt.title('Y(X)')
    plt.plot(x1, y1, 'ko', label='exact')
    plt.plot(x1, y_algpoly, 'b-p', label='algebraic')
    plt.plot(x1, y_legpoly, 'g:*', label='legendre')
    plt.legend()

    plt.figure(2)
    plt.title('log error')
    plt.plot(x1, -np.log10(np.abs(y1 - y_algpoly)), 'b-p', label='algebraic')
    plt.plot(x1, -np.log10(np.abs(y1 - y_legpoly)), 'g:*', label='legendre')
    plt.legend()
    plt.show()
