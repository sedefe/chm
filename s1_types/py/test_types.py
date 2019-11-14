import pytest
import numpy as np
import matplotlib.pyplot as plt

from s1_types.py import types


def test_uint():
    """
    check unsigned int conversions
    """
    print(f'running unsigned conversion test')
    for x in [0, 1, 4, 88, 2**32-1]:
        s = types.dec2bin(x, signed=False)
        assert len(s) == 32
        y = types.bin2dec(s, signed=False)
        assert x == y


def test_int():
    """
    check signed int conversions
    """
    print(f'running signed conversion test')
    for x, s in [
        (0, '0'*32),
        (-1, '1'*32),
        (1, '0'*31+'1'),
        (2**31-1, '0'+31*'1'),
        (-2**31, '1'+31*'0')
    ]:
        s1 = types.dec2bin(x, signed=True)
        x1 = types.bin2dec(s, signed=True)
        assert x1 == x
        assert s1 == s


def test_float():
    """
    check floating point conversions
    """
    print(f'running floating point test')

    def split_float(s: str):
        return s[:1], s[1:9], s[9:32]

    for f in [2.**-120, -2.**120, 1.0, 0., 2.**120]:
        s = types.float2bin(f)
        f1 = types.bin2float(s)
        sign, exp, mantis = split_float(s)
        print(f'{sign}_{exp}_{mantis}')
        assert f == f1


def test_float_log2():
    """
    check linear dependency of log2(x) and x converted to uint
    """
    X = np.linspace(0, 16, 100)[1:]
    Y_log = np.log2(X)
    Y_int = [types.bin2dec(types.float2bin(x), signed=False) for x in X]

    plt.subplot(1, 2, 1)
    plt.plot(X, Y_log, 'r.-', label='log2(x)')
    plt.plot(X, Y_int, 'b.-', label='uint(x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Y_int,  Y_log, 'k.-', label=f'log2(x) vs uint(x)')
    plt.legend()

    plt.show()


def test_fast_inv_sqrt():
    """
    calculate fast inverse square root by floating point trick
    """
    def fast_inv_sqrt(x, n_iter=1):
        c = 0x5f3759df
        s = types.float2bin(x)
        i = c - types.bin2dec(f'0{s[:-1]}', signed=False)
        y = types.bin2float(types.dec2bin(i, signed=False))
        for _ in range(n_iter):
            y *= 1.5 - 0.5 * x * y * y
        return y

    X = np.linspace(0, 16, 100)[1:]
    Y0 = np.power(X, -0.5)

    Y = [
        [fast_inv_sqrt(x, n_iter=i) for x in X]
        for i in range(3)]
    colors = 'rgb'

    plt.subplot(1, 2, 1)
    plt.plot(X, Y0, 'k-', label='exact')
    for i in range(3):
        plt.plot(X, Y[i], f'{colors[i]}.:', label=f'{i}-iterations approximation')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(X, -np.log10(np.abs(Y0 - Y[i])), f'{colors[i]}.:', label=f'{i}-iterations accuracy')
    plt.legend()

    plt.show()
