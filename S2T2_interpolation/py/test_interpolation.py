import pytest
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

from S2T2_interpolation.py.interpolation import LaGrange, Spline1, Spline3
from utils.utils import get_accuracy


class NodeType(Enum):
    EQ = auto()
    CHEB = auto()


class TestInterpolation:

    def _get_nodes(self, nodes_type: NodeType, a, b, n_nodes):
        if nodes_type == NodeType.EQ:
            return np.linspace(a, b, n_nodes)
        if nodes_type == NodeType.CHEB:
            return np.sort(1 / 2 * ((b - a) * np.cos(np.pi * (np.arange(n_nodes) + 1 / 2) / n_nodes) + (b + a)))
        raise ValueError(f'Unknown node type {nodes_type}')

    def _test_case(self, fname, func: callable, a, b, n_nodes, interp_params):
        """
        Общий метод проверки
        """
        k_dense = 10
        m = k_dense * n_nodes

        # точки, в которых будем сравнивать с точным значением
        xs_dense = np.array(sorted([*np.linspace(a, b, m),
                                    *self._get_nodes(NodeType.EQ, a, b, n_nodes),
                                    *self._get_nodes(NodeType.CHEB, a, b, n_nodes)]))
        ys_dense = func(xs_dense)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.plot(xs_dense, ys_dense, 'k-', label='exact')

        for (color, interp_name, nodes_type) in interp_params:
            xs = self._get_nodes(nodes_type, a, b, n_nodes)
            ys = func(xs)

            interp = interp_name(xs, ys)

            # проверяем, что попали в узлы интерполяции
            ys_num = interp(xs)
            assert (np.abs(ys - ys_num) < 1e-6).all()

            # смотрим, как у нас дела в остальных точках интервала
            ys_dense_num = interp(xs_dense)

            label = f'{interp.name}-{str(nodes_type.name)}'
            ax1.plot(xs_dense, ys_dense_num, f'{color}:', label=label)
            ax1.plot(xs, ys, f'{color}.')
            ax2.plot(xs_dense, get_accuracy(ys_dense, ys_dense_num), f'{color}-', label=label)

        ax1.set_title(f'{fname}')
        ax1.legend()

        ax2.set_title('accuracy')
        ax2.legend()

        plt.show()

    @pytest.mark.parametrize('fname, func',
                             [
                                 ['exp(sin(x))', lambda x: np.exp(np.sin(x))],
                                 ['x^4', lambda x: x**4],
                             ])
    def test_equidistant(self, fname, func):
        """
        Интерполяция с равноотстоящими узлами
        """
        n_nodes = 15
        a, b = -1, 1
        interp_params = [
            # color, interp_name, nodes_type, label
            ['b', LaGrange, NodeType.EQ],
            ['g', Spline1,  NodeType.EQ],
            ['c', Spline3,  NodeType.EQ],
        ]

        self._test_case(fname, func, a, b, n_nodes, interp_params)

    @pytest.mark.parametrize('fname, func',
                             [
                                 ['exp(sin(x))', lambda x: np.exp(np.sin(x))],
                                 ['cos(exp(x))', lambda x: np.cos(np.exp(x))],
                             ])
    def test_chebyshev(self, fname, func):
        """
        Интерполяция с узлами Чебышёва
        """
        n_nodes = 15
        a, b = -1, 1
        interp_params = [
            # color, interp_name, nodes_type
            ['b', LaGrange, NodeType.EQ],
            ['r', LaGrange, NodeType.CHEB],
            ['g', Spline1,  NodeType.CHEB],
        ]

        self._test_case(fname, func, a, b, n_nodes, interp_params)

    def test_runge(self):
        """
        https://en.wikipedia.org/wiki/Runge%27s_phenomenon
        """
        func = lambda x: 1 / (1 + x**2)
        n_nodes = 15
        a, b = -5, 5
        interp_params = [
            # color, interp_name, nodes_type, label
            ['b', LaGrange, NodeType.EQ],
            ['r', LaGrange, NodeType.CHEB],
            ['c', Spline3,  NodeType.EQ],
        ]

        self._test_case('1 / (1 + x**2)', func, a, b, n_nodes, interp_params)

    def test_convergence(self):
        """
        Проверка сходимости
        """
        func = np.abs
        a, b = -5, 5

        interp_params = [
            # color, interp_name, nodes_type, label
            ['b', LaGrange, NodeType.EQ],
            ['r', LaGrange, NodeType.CHEB],
            ['c', Spline3,  NodeType.EQ],
        ]

        node_params = [
            # n_nodes, style
            [5, '-'],
            [17, '--'],
            [65, ':'],
            # [257, '-.'],
        ]

        # точки, в которых будем сравнивать с точным значением
        m = 1025
        xs_dense = np.linspace(a, b, m)
        ys_dense = func(xs_dense)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.plot(xs_dense, ys_dense, 'k-', label='exact')

        for (color, interp_name, nodes_type) in interp_params:
            for (n_nodes, style) in node_params:
                xs = self._get_nodes(nodes_type, a, b, n_nodes)
                ys = func(xs)

                interp = interp_name(xs, ys)
                ys_dense_num = interp(xs_dense)

                label = f'{interp.name}-{str(nodes_type.name)}-{n_nodes}'
                ax1.plot(xs_dense, ys_dense_num, f'{color}{style}', label=label)
                ax1.plot(xs, ys, f'{color}.')
                ax2.plot(xs_dense, get_accuracy(ys_dense, ys_dense_num), f'{color}{style}', label=label)

        ax1.set_title(f'y(x)')
        ax1.axis([a, b, -1, 6])
        ax1.legend()

        ax2.set_title('accuracy')
        ax2.legend()

        plt.show()
