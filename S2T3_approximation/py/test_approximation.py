import pytest
import numpy as np
import numpy.polynomial as P
import matplotlib.pyplot as plt

from S2T3_approximation.py.approximation import Algebraic, Legendre, Harmonic
from utils.utils import get_accuracy


@pytest.mark.parametrize('fname, func', [
    ['tg(x) + sin(x)', lambda x: np.tan(x) + np.sin(x)],
    ['|x| - cos(pi*x)', lambda x: np.abs(x) - np.cos(np.pi*x)],
    ['x**2 - 2', lambda x: x**2 - 2],
    ['strobe', lambda x: (np.abs(x) < 1/2).astype(float)],
])
def test_polynomial(fname, func: callable):
    """
    Сравниваем аппроксимацию алгебраическими многочленами и многочленами Лежандра
    """
    n = 15
    dim = 5
    m = 101

    xs0 = np.linspace(-1, 1, n)
    xs1 = np.linspace(-1, 1, m)

    ys0 = func(xs0)
    ys1 = func(xs1)

    colors = 'bg'
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(f'{fname}')
    ax2.set_title('accuracy')
    ax1.plot(xs1, ys1, 'k-', label='exact')
    ax1.plot(xs0, ys0, 'k.')

    coeffs = []
    for color, approx_type in zip(colors, [Algebraic, Legendre]):
        approx = approx_type(xs0, ys0, dim)
        ys1_num = approx(xs1)
        coeffs.append(approx.coeffs)

        ax1.plot(xs1, ys1_num, f'{color}-', label=approx.name)
        ax2.plot(xs1, get_accuracy(ys1, ys1_num), f'{color}-', label=approx.name)

        assert(len(approx.coeffs) == dim), f'{approx_type} polynome length should be {dim}'
        assert(all(abs(ys1 - ys1_num) < 1)), f'{approx_type} polynome approximation is too bad'

    alg_poly = coeffs[0]
    leg_poly = P.legendre.leg2poly(coeffs[1])
    assert(all(abs(alg_poly - leg_poly) < 1e-3)), 'algebraic and legendre are not consistent'

    ax1.legend()
    ax2.legend()
    plt.show()


@pytest.mark.parametrize('fname, func', [
    ['sin(2pi*x) - cos(pi*x)', lambda x: np.sin(2*np.pi*x) - np.cos(np.pi*x)],
    ['strobe', lambda x: (np.abs(x) < 1/2).astype(float)],
])
def test_harmonic(fname, func: callable):
    """
    Сравниваем аппроксимацию алгебраическими многочленами и гармониками
    """
    n = 50
    dim = 21
    m = 99

    xs0 = np.linspace(-1, 1, n)
    xs1 = np.linspace(-1, 1, m)

    ys0 = func(xs0)
    ys1 = func(xs1)

    colors = 'br'
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(f'{fname}')
    ax2.set_title('accuracy')
    ax1.plot(xs1, ys1, 'k-', label='exact')
    ax1.plot(xs0, ys0, 'k.')

    for color, approx_type in zip(colors, [Algebraic, Harmonic]):
        approx = approx_type(xs0, ys0, dim)
        ys1_num = approx(xs1)

        ax1.plot(xs1, ys1_num, f'{color}-', label=approx.name)
        ax2.plot(xs1, get_accuracy(ys1, ys1_num), f'{color}-', label=approx.name)

    ax1.legend()
    ax2.legend()
    plt.show()


@pytest.mark.parametrize('fname, func', [
    ['strobe', lambda x: (np.abs(x) < 1/2).astype(float)],
])
def test_convergence(fname, func: callable):
    """
    Сравниваем сходимость
    """
    n = 50
    m = 99

    xs0 = np.linspace(-1, 1, n)
    xs1 = np.linspace(-1, 1, m)

    ys0 = func(xs0)
    ys1 = func(xs1)

    colors = 'rgbmc'
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('Algebraic')
    axs[1].set_title('Harmonic')
    for ax in axs:
        ax.plot(xs1, ys1, 'k-', label='exact')
        ax.plot(xs0, ys0, 'k.')

    for ax, approx_type in zip(axs, [Algebraic, Harmonic]):
        for color, dim in zip(colors, [1, 11, 21, 31, 41]):
            approx = approx_type(xs0, ys0, dim)
            ys1_num = approx(xs1)

            ax.plot(xs1, ys1_num, f'{color}--', label=f'{approx.name} {dim}-d')
            # ax.plot(xs1, get_accuracy(ys1, ys1_num), f'{color}-', label=f'{approx.name} {dim}-d')

        ax.axis([-1, 1, -0.5, 1.5])
        ax.legend()
    plt.show()
