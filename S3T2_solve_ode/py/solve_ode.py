import enum
import numpy as np

from utils.ode_collection import ODE
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, ode: ODE, y_start, ts):
    """
    Интегрирование одношаговым методом с фиксированным шагом

    :param method:  одношаговый метод
    :param ode:     СОДУ
    :param y_start: начальное значение
    :param ts:      набор значений t
    :return:        список значений t (совпадает с ts), список значений y
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(ode, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, ode: ODE, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    Интегрирование одношаговым методом с адаптивным выбором шага.
    Допуски контролируют локальную погрешность:
        err <= atol
        err <= |y| * rtol

    :param method:      одношаговый метод
    :param ode:         СОДУ
    :param y_start:     начальное значение
    :param t_span:      интервал интегрирования (t0, t1)
    :param adapt_type:  правило Рунге (AdaptType.RUNGE) или вложенная схема (AdaptType.EMBEDDED)
    :param atol:        допуск на абсолютную погрешность
    :param rtol:        допуск на относительную погрешность
    :return:            список значений t (совпадает с ts), список значений y
    """
    y = y_start
    t, t_end = t_span

    ys = [y]
    ts = [t]

    raise NotImplementedError
    return ts, ys
