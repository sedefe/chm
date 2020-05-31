import numpy as np
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


#  Коэффиенты методов Адамса
adams_coeffs = {
    1: [1],
    2: [-1 / 2, 3 / 2],
    3: [5 / 12, -4 / 3, 23 / 12],
    4: [-3 / 8, 37 / 24, -59 / 24, 55 / 24],
    5: [251 / 720, -637 / 360, 109 / 30, -1387 / 360, 1901 / 720]
}


def adams(func, y_start, T, coeffs, one_step_method: OneStepMethod):
    """
    T: список точек, по которым мы шагаем (шаг постоянный)
    coeffs: список коэффициентов метода Адамса
    one_step_method: одношаговый метод для разгона
    returns: list of t (same as T), list of y
    """
    raise NotImplementedError
