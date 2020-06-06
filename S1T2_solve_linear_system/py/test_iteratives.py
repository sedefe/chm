import pytest
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from S1T2_solve_linear_system.py.iteratives import richardson, jacobi, seidel
from utils.utils import get_accuracy


@pytest.mark.parametrize('tol, check', [
    [1e-10, True],
    [1e-20, False],
])
def test_iteratives(tol, check):
    """
    Сравниваем итерационные методы
    Q: который метод лучше? Почему?
    Q: почему во втором случае мы не всегда можем достичь заданной точности?
    """
    n = 5
    A = np.array([
        [n + 2, 1, 1],
        [1, n + 4, 1],
        [1, 1, n + 6],
    ], dtype='float64')

    b = np.array([n + 4, n + 6, n + 8], dtype='float64')

    methods = [richardson, jacobi, seidel]
    colors = 'mgb'
    names = ['Richardson', 'Jacobi', 'Gauss-Seidel']

    for method, color, name in zip(methods, colors, names):
        xs, ys = method(A, b, tol)
        plt.plot(range(len(ys)), get_accuracy(ys, np.zeros_like(ys), eps=tol/10), f'{color}.-', label=name)
        if check:
            assert np.linalg.norm(A@xs[-1] - b) <= tol, f'{name} method failed'

    axes = plt.axis()
    plt.plot(axes[:2], -np.log10([tol, tol]), 'k:', label='tolerance')

    plt.suptitle(f'Test iterative methods for tol {tol}')
    plt.ylabel('accuracy')
    plt.xlabel('N iter')
    plt.legend()
    plt.show()
