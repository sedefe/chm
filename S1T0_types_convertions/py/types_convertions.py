import ctypes


def dec2bin(x, signed) -> str:
    """
    get binary representation of int32/uint32
    you should rewrite the code to avoid built-in f-string conversions
    """
    if signed:
        return f'{x & 0xffffffff:032b}'
    else:
        return f'{x:032b}'


def bin2dec(x, signed) -> int:
    """
    get integer from it's binary representation
    you should rewrite the code to avoid built-in int(x, 2) conversion
    """
    return int(x, 2) - 2**32 * int(x[0]) * signed


def float2bin(x) -> str:
    """
    get binary representation of float32
    you may leave it as is
    """
    return f'{ctypes.c_uint32.from_buffer(ctypes.c_float(x)).value:>032b}'


def bin2float(x) -> float:
    """
    get float from it's binary representation
    you should rewrite the code to avoid ctypes conversions
    """
    return ctypes.c_float.from_buffer(ctypes.c_uint32(int(x, 2))).value
