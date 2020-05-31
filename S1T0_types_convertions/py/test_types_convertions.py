import pytest
import numpy as np
import matplotlib.pyplot as plt

from S1T0_types_convertions.py import types_convertions as tc


def test_uint():
    """
    Проверяем преобразования для unsigned int
    """
    for x in [0, 1, 4, 88, 2**32-1]:
        s = tc.dec2bin(x, signed=False)
        assert len(s) == 32
        y = tc.bin2dec(s, signed=False)
        assert x == y


def test_int():
    """
    Проверяем преобразования для signed int
    """
    for x, s in [
        (0, '0'*32),
        (-1, '1'*32),
        (1, '0'*31+'1'),
        (2**31-1, '0'+31*'1'),
        (-2**31, '1'+31*'0')
    ]:
        s1 = tc.dec2bin(x, signed=True)
        x1 = tc.bin2dec(s, signed=True)
        assert x1 == x
        assert s1 == s


def test_float():
    """
    Проверяем преобразования для  floating point
    """
    def split_float(s: str):
        return s[:1], s[1:9], s[9:32]

    for f in [2.**-120, -2.**120, 1.0, 0., 2.**120]:
        s = tc.float2bin(f)
        f1 = tc.bin2float(s)
        sign, exp, mantis = split_float(s)
        print(f'{sign}_{exp}_{mantis}')
        assert f == f1


def test_float_log2():
    """
    Проверяем линейную зависимости между log2(x) и целочисленной интерпретацией x
    """
    X = np.linspace(0, 16, 100)[1:]
    Y_log = np.log2(X)
    Y_int = [tc.bin2dec(tc.float2bin(x), signed=False) for x in X]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(X, Y_log, 'r.-', label='log2(x)')
    ax1.plot(X, Y_int, 'b.-', label='uint(x)')
    ax1.legend()

    ax2.plot(Y_int,  Y_log, 'k.-', label=f'log2(x) vs uint(x)')
    ax2.legend()

    plt.show()


def test_fast_inv_sqrt():
    """
    Проверяем быстрое вычисление обратного квадратного корня при помощи трюка с плавающей запятой
    https://en.wikipedia.org/wiki/Fast_inverse_square_root
    """
    def fast_inv_sqrt(x, n_iter=1):
        c = 0x5f3759df
        s = tc.float2bin(x)
        i = c - tc.bin2dec(f'0{s[:-1]}', signed=False)
        y = tc.bin2float(tc.dec2bin(i, signed=False))
        for _ in range(n_iter):
            y *= 1.5 - 0.5 * x * y * y
        return y

    X = np.linspace(0, 16, 100)[1:]
    Y0 = np.power(X, -0.5)

    Y = [
        [fast_inv_sqrt(x, n_iter=i) for x in X]
        for i in range(3)]
    colors = 'rgb'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(X, Y0, 'k-', label='exact')
    for i in range(3):
        ax1.plot(X, Y[i], f'{colors[i]}.:', label=f'{i}-iterations approximation')
    ax1.legend()

    for i in range(3):
        ax2.plot(X, -np.log10(np.abs(Y0 - Y[i])), f'{colors[i]}.:', label=f'{i}-iterations accuracy')
    ax2.legend()

    plt.show()
