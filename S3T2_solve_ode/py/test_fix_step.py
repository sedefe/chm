import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic
from utils.utils import get_log_error
from S3T2_solve_ode.py.fix_step_integration import fix_step_integration
from S3T2_solve_ode.py.explicit_one_step_methods import euler, runge_kutta, rk4_coeffs, dopri_coeffs
from S3T2_solve_ode.py.explicit_multistep_methods import adams, adams_coeffs


def test_one_step():
    """
    test Euler and RK methods
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = np.pi/2
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    T = np.arange(t0, t1+dt, dt)

    exact = f[T].T
    plt.subplot(1, 2, 1)
    plt.plot(T, [e[0] for e in exact], 'k', label='Exact')

    colors = 'rgbcmyk'
    for i, (method, label, kwargs) in enumerate(
            [
                [euler, 'Euler', {}],
                [runge_kutta, 'RK4', {'rk_coeffs': rk4_coeffs}],
                [runge_kutta, 'DoPri', {'rk_coeffs': dopri_coeffs}],
            ]
    ):
        _, y = fix_step_integration(method, f, y0, T, **kwargs)
        print(f'len(Y): {len(y)}')
        print(f'Function calls: {f.get_call_counter()}')

        plt.subplot(1, 2, 1), plt.plot(T, [_y[0] for _y in y], f'{colors[i]}.--', label=label)
        plt.subplot(1, 2, 2), plt.plot(T, get_log_error(exact, y), f'{colors[i]}.--', label=label)

    plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
    plt.suptitle('one-step methods')
    plt.show()


def test_multi_step():
    """
    test Adams method
    """
    y0 = np.array([0, 1])
    t0 = 0
    t1 = 1.
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    T = np.arange(t0, t1+dt, dt)
    exact = f[T].T

    plt.figure()
    plt.subplot(1, 2, 1), plt.plot(T, [e[0] for e in exact], 'k', label='Exact')
    for p, c in adams_coeffs.items():
        t_adams, y_adams = adams(f, y0, T, c,
                                 one_step_method=runge_kutta,
                                 rk_coeffs=rk4_coeffs)
        print(f'Function calls: {f.get_call_counter()}')

        err = get_log_error(exact, y_adams)

        label = f"Adams's order {p}"
        plt.subplot(1, 2, 1), plt.plot(t_adams, [y[0] for y in y_adams], '.--', label=label)
        plt.subplot(1, 2, 2), plt.plot(t_adams, err, '.--', label=label)

    plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
    plt.suptitle('multi-step methods')
    plt.show()
