import pytest
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

from S1T2_solve_linear_system.py.exacts import (qr, lu,
                                                solve_lower_triang,
                                                solve_upper_triang,
                                                solve_lu,
                                                solve_qr)


def test_lu():
    """
    check your LU vs SciPy's LU
    NB: SciPy's returns (P,L,U), where P is a permutational matrix, but here P is identity matrix
    """
    with np.printoptions(precision=3, suppress=True):
        A = np.array([
            [9, 1, 2],
            [0, 8, 1],
            [9, 1, 9],
        ], dtype='float64')
        P, L0, U0 = spla.lu(A)
        L1, U1 = lu(A)

        assert npla.norm(A - L1 @ U1) < 1e-6
        assert npla.norm(L1 - L0) < 1e-6
        assert npla.norm(U1 - U0) < 1e-6


def test_qr():
    """
    check your QR vs NumPy's QR
    NB: signs of Q's and R's can differ
    """
    with np.printoptions(precision=3, suppress=True):
        A = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 9],
        ], dtype='float64')

        Q0, R0 = npla.qr(A)
        Q1, R1 = qr(A)

        # check composition
        assert npla.norm(A - Q1 @ R1) < 1e-6
        # check Q is orthogonal
        assert npla.norm(Q1 @ Q1.T - np.eye(len(Q1))) < 1e-6
        # check R is upper triangular
        assert npla.norm(np.abs(R1) - np.abs(R0)) < 1e-6


def test_triangle_solve():
    A = np.array([
        [1, 0, 0],
        [2, 1, 0],
        [4, 2, 1],
    ], dtype='float64')
    b = np.array([1, 1, 3], dtype='float64')

    x = solve_lower_triang(A, b)
    assert npla.norm(x - npla.solve(A, b)) < 1e-6

    x = solve_upper_triang(A.T, b)
    assert npla.norm(x - npla.solve(A.T, b)) < 1e-6


@pytest.mark.parametrize('n', range(0, 20))
def test_lu_solve(n):
    A = np.array([
        [n+2, 1, 1],
        [1, n+4, 1],
        [1, 1, n+6],
    ], dtype='float64')
    b = n + np.array([4, 6, 8], dtype='float64')

    x = solve_lu(A, b)
    assert npla.norm(x - 1) < 1e-6


@pytest.mark.parametrize('n', range(0, 20))
def test_qr_solve(n):
    A = np.array([
        [n+2, 1, 1],
        [1, n+4, 1],
        [1, 1, n+6],
    ], dtype='float64')
    b = n + np.array([4, 6, 8], dtype='float64')

    x = solve_qr(A, b)
    assert npla.norm(x - 1) < 1e-6
