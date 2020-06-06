import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from S1T2_solve_linear_system.py.iteratives import richardson, jacobi, seidel
from utils.utils import get_accuracy


def test_iteratives():
    """
    Сравниваем итерационные методы
    Q: который лучше? Почему?
    """
    n = 5
    A = np.array([
        [n + 2, 1, 1],
        [1, n + 4, 1],
        [1, 1, n + 6],
    ], dtype='float64')

    b = np.array([n + 4, n + 6, n + 8], dtype='float64')

    tol = 1e-6

    methods = [richardson, jacobi, seidel]
    colors = 'mgb'
    names = ['Richardson', 'Jacobi', 'Gauss-Seidel']

    for method, color, name in zip(methods, colors, names):
        xs, ys = method(A, b, tol)
        plt.plot(range(len(ys)), get_accuracy(ys, np.zeros_like(ys)), f'{color}.-', label=name)
        assert np.linalg.norm(A@xs[-1] - b) <= tol, f'{name} method failed'

    axes = plt.axis()
    plt.plot(axes[:2], -np.log10([tol, tol]), 'k:', label='tolerance')

    plt.suptitle('Test iterative methods')
    plt.ylabel('Acc')
    plt.xlabel('N iter')
    plt.legend()
    plt.show()


def test_convergence():
    """
    То же, что и в прошлом, но tol = 1e-20
    Q: почему мы не может достичь заданной точности?
    """
    n = 5
    A = np.array([
        [n + 2, 1, 1],
        [1, n + 4, 1],
        [1, 1, n + 6],
    ], dtype='float64')

    b = np.array([n + 4, n + 6, n + 8], dtype='float64')

    tol = 1e-20

    methods = [richardson, jacobi, seidel]
    colors = 'mgb'
    names = ['Richardson', 'Jacobi', 'Gauss-Seidel']

    for method, color, name in zip(methods, colors, names):
        xs, ys = method(A, b, tol)
        plt.plot(range(len(ys)), get_accuracy(ys, np.zeros_like(ys)), f'{color}.-', label=name)

    axes = plt.axis()
    plt.plot(axes[:2], -np.log10([tol, tol]), 'k:', label='tolerance')

    plt.suptitle('Test iterative methods convergence')
    plt.ylabel('Acc')
    plt.xlabel('N iter')
    plt.legend()
    plt.show()
