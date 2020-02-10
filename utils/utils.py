import numpy as np


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


def get_log_error(x, y, axis=1):
    return -np.log10(np.linalg.norm(np.array(x) - np.array(y), axis=axis))
