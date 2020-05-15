import numpy as np
from Newton.newton import JFNK
from Project.project_matricies import F
from Newton.preconditioners import multigrid_primer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def interval(n, length=1):
    dx = length / n
    return np.arange(0, n) * dx


def eval(func):
    def eval_wrapper(x):
        return (func(x) + func(np.roll(x, -1))) / 2

    return eval_wrapper


@eval
def func_u0(x):
    return 2 + 2*np.sin(np.pi * x)


def scaling():
    n = 2 ** 8
    JFNK(lambda u: F(u, dt, uk), uk, multigrid_primer(max(uk), dt, L))


def runem():
    L = 2
    n = 2 ** 8
    x = interval(n, length=L)
    u0 = func_u0(x)
    Nt = 4
    u = np.zeros((n, Nt + 1))
    u[:, 0] = u0

    dt = 2
    for k in range(0, Nt):
        uk = u[:, k]
        u[:, k + 1] = JFNK(lambda u: F(u, dt, uk), uk, multigrid_primer(max(uk), dt, L))
        print(k)
    np.save('Burger2.npy', u)


def ani():
    L = 2
    n = 2 ** 8
    x = interval(n, length=L)

    Nt = 30
    u = np.load('Burger.npy')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, L), ylim=(0, 4))
    line, = ax.plot([], [])

    for k in range(0, Nt):
        plt.cla()
        plt.plot(x, u[:, k])
        plt.axis([0, L, 0, 4])
        plt.pause(0.1)
        plt.show()


ani()