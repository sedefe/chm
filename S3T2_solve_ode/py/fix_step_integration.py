def fix_step_integration(method, func, y_start, T, **kwargs):
    """
    performs fix-step integration using one-step method
    """
    Y = [y_start]

    for i, t in enumerate(T[:-1]):
        y = Y[-1]

        y1 = method(func, t, y, T[i+1] - t, **kwargs)
        Y.append(y1)

    return T, Y
