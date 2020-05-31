import enum
import numpy as np

from S3T2_solve_ode.py.one_step_methods import OneStepMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, func, y_start, ts):
    """
    Выполняем интегрирование одношаговм метдом с фиксированным шагом
    ts: набор значений t
    returns: list of t, list of y
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(func, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, func, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    Выполняем интегирование одношаговым методом с адаптивным шагом
    t_span: (t0, t1)
    adapt_type: правило Рунге (AdaptType.RUNGE) или вложенная схема (AdaptType.EMBEDDED)
    допуски контролируют локальную погрешность:
        err <= atol
        err <= |y| * rtol
    returns: list of t, list of y
    """
    y = y_start
    t, t_end = t_span

    ys = [y]
    ts = [t]

    raise NotImplementedError
    return ts, ys
