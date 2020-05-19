from Newton.newton import JFNK
from Newton.preconditioners import multigrid_primer
import matplotlib.pyplot as plt
from time import time
from Project.project_matricies import *


def calculate_pseudo_stepsize(a, dt, L, c1):
    return lambda N: c1 * L / (a * dt * N)


def one_spec(n, pre, post):
    L = 2
    x = interval(n, length=L)
    u0 = func_u0(x)
    dt = 0.1

    pseudo_step = calculate_pseudo_stepsize(max(u0), dt, L, 0.99)
    t1 = time()
    residuals = JFNK(lambda u: F(u, dt, u0), u0, multigrid_primer(0.33, pseudo_step, pre, post))
    return time() - t1, residuals


def spec():
    N = 2 ** np.array([8, 9, 10])
    p = 1

    times = []
    residualss = []
    for n in N:
        t, residuals = one_spec(n, p, p)
        times.append(t)
        residualss.append(residuals)

    times = np.array(times) * 1000
    times = np.floor(times)
    fig, ax = plt.subplots()
    lines = plt.semilogy(residualss[0], 'bs-', residualss[1], 'rs-', residualss[2], 'gs-')
    plt.legend(lines, ('256: {}ms'.format(times[0]), '512: {}ms'.format(times[1]), '1024: {}ms'.format(times[2])))
    plt.ylabel('Residual')
    plt.xlabel('Iteration number')
    plt.title('Plot over residual for 3 different numbers of inner points')
    plt.show()


def runem():
    L = 2
    n = 2 ** 8
    x = interval(n, length=L)
    u0 = func_u0(x)
    Nt = 300
    u = np.zeros((n, Nt + 1))
    u[:, 0] = u0

    dt = 0.2

    pseudo_step = calculate_pseudo_stepsize(max(u0), dt, L, 0.99)
    for k in range(0, Nt):
        uk = u[:, k]
        u[:, k + 1] = JFNK(lambda u: F(u, dt, uk), uk, multigrid_primer(0.33, pseudo_step, 1, 1))
        print(k)
    np.save('OkBurger2.npy', u)
    animation('OkBurger2.npy')


def animation(name):
    L = 2
    u = np.load(name)
    n = u[:, 1].shape[0]
    x = interval(n, length=L)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 4))
    line, = ax.plot([], [])
    for uk in u.T[:, ]:
        plt.cla()
        plt.plot(x, uk)
        plt.axis([0, L, 0, 4])
        plt.pause(0.01)
        plt.show()


def problem_a():
    n = 2 ** 10
    dt = 1


    calculate_pseudo_stepsize()
runem()
