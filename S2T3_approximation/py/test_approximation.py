import pytest
import numpy as np
import matplotlib.pyplot as plt

from S2T3_approximation.py import approximation
from utils.utils import get_accuracy


def test_approximation():
    n = 5
    dim = 3
    m = 101

    x0 = np.linspace(-1, 1, n)
    x1 = np.linspace(-1, 1, m)

    y0 = approximation.func(x0)
    y1 = approximation.func(x1)

    y_algpoly, _, p_algpoly = approximation.approx(x0, y0, x1, approximation.ApproxType.algebraic, dim)
    assert(len(p_algpoly) == dim), f'polynome length should be {dim}'
    assert(all(abs(y1 - y_algpoly) < 1)), 'algebraic polynome approximation is too bad'

    y_legpoly, _, p_legpoly = approximation.approx(x0, y0, x1, approximation.ApproxType.legendre, dim)
    assert(len(p_legpoly) == dim), f'polynome length should be {dim}'
    assert(all(abs(y1 - y_legpoly) < 1)), 'legendre polynome approximation is too bad'
    # assert(all(abs(p_algpoly - p_legpoly) < 1e-3))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('y(x)')
    ax1.plot(x1, y1, 'ko', label='exact')
    ax1.plot(x1, y_algpoly, 'b.:', label='algebraic')
    ax1.plot(x1, y_legpoly, 'g.:', label='legendre')
    ax1.legend()

    ax2.set_title('accuracy')
    ax2.plot(x1, get_accuracy(y1, - y_algpoly), 'b.:', label='algebraic')
    ax2.plot(x1, get_accuracy(y1, - y_legpoly), 'g.:', label='legendre')
    ax2.legend()
    plt.show()
