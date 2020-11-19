import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from S2T1_optimization.py import optimization
from utils.utils import get_accuracy


default_student = 0


@pytest.mark.parametrize("eps_y, check_accuracy", [  # допуск
    [1e-06, True],
    [1e-16, False],
])
@pytest.mark.parametrize("student", [default_student])  # номер варианта
@pytest.mark.parametrize("n_dim, projection", [  # размерность задачи
    [2, 'rectilinear'],
    [3, '3d'],
])
def test_optimization(eps_y, check_accuracy, student, n_dim, projection):
    N = student

    A = np.array([[4,   1,      1],
                  [1,   6+.2*N, -1],
                  [1,   -1,     8+.2*N]],
                 dtype='float')[:n_dim, :n_dim]
    b = np.array([1, -2, 3], dtype='float').T[:n_dim]
    x0 = np.zeros_like(b)

    x1 = np.linalg.solve(A, -b)
    y1 = (1 / 2 * x1.T @ A @ x1 + b.T @ x1).item()

    eps_x = np.sqrt(eps_y)

    methods = ['mngs', 'mps', 'newton']
    styles = ['go:', 'bo:', 'mo:']

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection=projection)

    for i, method in enumerate(methods):
        optimization.func.calls = 0
        xs, ys = getattr(optimization, method)(A=A, b=b, x0=x0, eps=eps_y, max_iter=30)

        assert np.equal(x0, xs[0]).all(), 'xs should start with initial point'
        if check_accuracy:
            assert np.linalg.norm(x1 - xs[-1]) < eps_x, 'last xs should be close enough to the optimum'
            assert np.linalg.norm(y1 - ys[-1]) < eps_y, 'last ys should be close enough to the optimum'
        assert optimization.func.calls == len(ys), f'function was called {optimization.func.calls} times, ' \
                                                   f'but there is {len(ys)} point in the output'

        ax1.plot(get_accuracy(ys, y1*np.ones_like(ys)), styles[i], label=method)
        ax2.plot(*list(np.array(xs).T), styles[i], label=method)
    ax2.plot(*[[x] for x in x1], 'kp', label='exact')

    ax = ax1.axis()
    ax1.plot(ax[:2], -np.log10([eps_y, eps_y]), 'k-')
    ax1.set_xlabel('N iter')
    ax1.set_ylabel('accuracy')
    ax1.legend()
    ax2.legend()
    fig.suptitle(f'Results for {n_dim}D')
    plt.show()
