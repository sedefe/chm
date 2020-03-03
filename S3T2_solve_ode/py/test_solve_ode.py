import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic, HarmExp
from utils.utils import get_log_error
from S3T2_solve_ode.py.solve_ode import fix_step_integration, adaptive_step_integration, AdaptType
from S3T2_solve_ode.py.one_step_methods import EulerMethod, RungeKuttaMethod, EmbeddedRungeKuttaMethod
import S3T2_solve_ode.py.coeffs_collection as collection
from S3T2_solve_ode.py.multistep_methods import adams, adams_coeffs


def test_one_step():
    """
    test Euler and RK methods
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = np.pi/2
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)

    exact = f[ts].T
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(ts, [e[0] for e in exact], 'k', label='Exact')

    colors = 'rgbcmyk'
    for i, method in enumerate(
            [
                EulerMethod(),
                RungeKuttaMethod(collection.rk4_coeffs),
                RungeKuttaMethod(collection.dopri_coeffs),
            ]
    ):
        _, y = fix_step_integration(method, f, y0, ts)
        print(f'len(Y): {len(y)}')
        print(f'Function calls: {f.get_call_counter()}')

        plt.subplot(1, 2, 1), plt.plot(ts, [_y[0] for _y in y], f'{colors[i]}.--', label=method.name)
        plt.subplot(1, 2, 2), plt.plot(ts, get_log_error(exact, y), f'{colors[i]}.--', label=method.name)

    plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
    plt.suptitle('one-step methods')
    plt.show()


def test_multi_step():
    """
    test Adams method
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = 1.
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)
    exact = f[ts].T

    plt.figure()
    plt.subplot(1, 2, 1), plt.plot(ts, [e[0] for e in exact], 'k', label='Exact')
    for p, c in adams_coeffs.items():
        t_adams, y_adams = adams(f, y0, ts, c,
                                 one_step_method=RungeKuttaMethod(collection.rk4_coeffs))
        print(f'Function calls: {f.get_call_counter()}')

        err = get_log_error(exact, y_adams)

        label = f"Adams's order {p}"
        plt.subplot(1, 2, 1), plt.plot(t_adams, [y[0] for y in y_adams], '.--', label=label)
        plt.subplot(1, 2, 2), plt.plot(t_adams, err, '.--', label=label)

    plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
    plt.suptitle('multi-step methods')
    plt.show()


@pytest.mark.parametrize('f,y0', (
        (Harmonic(np.array([1., 1.]), 1, 1), np.array([1., 1.])),
        (HarmExp(), np.exp([1, 0])),
))
def test_adaptive(f, y0):
    """
    test adaptive step algorithms
    """
    t0, t1 = 0, 4*np.pi

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        (EulerMethod(),                                             AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs=collection.rk4_coeffs),            AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs=collection.dopri_coeffs),  AdaptType.EMBEDDED),
    )

    for method, adapt_type in methods:
        f.clear_call_counter()
        ts, ys = adaptive_step_integration(method=method,
                                           func=f, y_start=y0, t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        print(f'{method.name} took {f.get_call_counter()} function calls')

        tss.append(np.array(ts))
        yss.append(ys)

    ts = np.array(sorted([t for ts in tss for t in ts]))
    exact = f[ts].T
    y0 = np.array([y[0] for y in exact])

    # plots
    plt.figure('y(t)'), plt.suptitle('y(t)'), plt.xlabel('t'), plt.ylabel('y')
    plt.plot(ts, y0, 'ko-', label='exact')

    plt.figure('dt(t)'), plt.suptitle('step sizes'), plt.xlabel('t'), plt.ylabel('dt')
    plt.figure('dy(t)'), plt.suptitle('accuracies'), plt.xlabel('t'), plt.ylabel('accuracy')

    for (m, _), ts, ys in zip(methods, tss, yss):
        plt.figure('y(t)'), plt.plot(ts, [y[0] for y in ys], '.', label=m.name)
        plt.figure('dt(t)'), plt.plot(ts[:-1], ts[1:] - ts[:-1], '.-', label=m.name)
        plt.figure('dy(t)'), plt.plot(ts, get_log_error(f[ts].T, ys), '.-', label=m.name)

    plt.figure('y(t)'), plt.legend()
    plt.figure('dt(t)'), plt.legend()

    plt.show()


def test_adaptive_order():
    """
    test adaptive algorithms convergence
    """
    t0, t1 = 0, 2*np.pi
    y0 = np.array([1., 1.])
    f = Harmonic(y0, 1, 1)

    methods = (
        (EulerMethod(),                                            AdaptType.RUNGE),
        (RungeKuttaMethod(coeffs=collection.rk4_coeffs),           AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs=collection.dopri_coeffs), AdaptType.EMBEDDED),
    )
    tols = 10. ** -np.arange(3, 9)

    plt.figure()
    for i, (method, adapt_type) in enumerate(methods):
        print(method.name)
        fcs = []
        errs = []
        for tol in tols:
            f.clear_call_counter()
            ts, ys = adaptive_step_integration(method=method,
                                               func=f, y_start=y0, t_span=(t0, t1),
                                               adapt_type=adapt_type,
                                               atol=tol, rtol=tol*1e3)
            err = np.linalg.norm(ys[-1] - f[t1])
            fc = f.get_call_counter()
            print(f'{method.name}: {fc} RHS calls, err = {err:.5f}')
            errs.append(err)
            fcs.append(fc)

        x = np.log10(fcs)
        y = -np.log10(errs)
        k, b = np.polyfit(x, y, 1)
        plt.plot(x, k*x+b, 'k:')
        plt.plot(x, y, 'p', label=f'{method.name} ({k:.2f})')
    plt.suptitle('check RHS evals')
    plt.xlabel('log10(function_calls)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
