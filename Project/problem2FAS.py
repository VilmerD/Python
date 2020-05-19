from Linear_Solvers.multigrid import *
from Project.project_matricies import *
from Linear_Solvers.smoothers import *
import scipy.sparse.linalg as splin
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from time import time


def F_u(u, h):
    return u + h / 2 * (u ** 2 - np.roll(u, -1) ** 2)


def F_linop(h):
    def F_wrapper(n):
        return splin.LinearOperator((n, n), lambda u: F_u(u, h))
    return F_wrapper


def run_spec(n, dt, pre, cons):
    L = 2
    x = interval(n, length=L)
    u0 = func_u0(x)
    u = u0.copy()

    pseudo_step = lambda x: cons[1]

    fnorm = np.linalg.norm(u)
    residual0 = np.linalg.norm(F_u(u, dt) - u) / fnorm
    residuals = [residual0]

    t1 = time()
    while residuals[-1] > 10 ** -9:
        u = FAS(F_linop(dt), u, u0, RungeKutta(cons[0], pseudo_step), pre=pre)
        residual = np.linalg.norm(F_u(u, dt) - u0) / fnorm
        residuals.append(residual)
        if abs(residuals[-1]/residuals[-2]) > 0.50:
            print("Stagnated at: {}".format(residuals[-1]))
            break
        print(np.log10(residual))

    fig, ax = plt.subplots()
    plt.semilogy(residuals)
    plt.show()


def animate(u):
    L = 2
    n = 2 ** 8
    x = interval(n, length=L)

    Nt = 100
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 4))
    line, = ax.plot([], [])

    for k in range(0, Nt):
        plt.cla()
        plt.plot(x, u[:, k])
        plt.axis([0, L, 0, 4])
        plt.pause(0.1)
        plt.show()


run_spec(2 ** 8, 0.2, 12, (0.3, 0.8))
