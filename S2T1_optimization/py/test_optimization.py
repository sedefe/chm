import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from S2T1_optimization.py import optimization
from utils.utils import get_accuracy


default_student = 0


@pytest.mark.parametrize("student", [default_student])  # номер варианта
@pytest.mark.parametrize("n_dim, projection", [  # размерность задачи
    [2, 'rectilinear'],
    [3, '3d'],
])
def test_optimization(student, n_dim, projection):
    N = student

    A = np.array([[4,  1,      1],
                  [1,  6+.2*N, -1],
                  [1,  -1,      8+.2*N]],
                 dtype='float')[:n_dim, :n_dim]
    b = np.array([1, -2, 3], dtype='float').T[:n_dim]
    x0 = np.zeros_like(b)

    x1 = np.linalg.solve(A, -b)
    y1 = (1 / 2 * x1.T @ A @ x1 + b.T @ x1).item()

    eps_y = 1e-6
    eps_x = 1e-3

    methods = ['mngs', 'mps']
    styles = ['go:', 'bo:']

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection=projection)

    for i, method in enumerate(methods):
        optimization.func.calls = 0
        X, Y = getattr(optimization, method)(A, b, x0, eps_y)

        assert np.equal(x0, X[0]).all(), 'X should start with initial point'
        assert np.linalg.norm(x1 - X[-1]) < eps_x, 'last X should be close enough to the optimum'
        assert np.linalg.norm(y1 - Y[-1]) < eps_y, 'last Y should be close enough to the optimum'
        assert optimization.func.calls == len(Y), f'function was called {optimization.func.calls} times, ' \
                                                  f'but there is {len(Y)} point in the output'

        ax1.plot(get_accuracy(Y, y1*np.ones_like(Y)), styles[i], label=method)
        ax2.plot(*list(np.array(X).T), styles[i], label=method)
    ax2.plot(*[[x] for x in x1], 'kp', label='exact')

    ax = ax1.axis()
    ax1.plot(ax[:2], -np.log10([eps_y, eps_y]) , 'k-')
    ax1.set_xlabel('N iter')
    ax1.set_ylabel('acc')
    ax1.legend()
    ax2.legend()
    fig.suptitle(f'Results for dimension {n_dim}')
    plt.show()
