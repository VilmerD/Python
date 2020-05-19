import numpy as np
import scipy.sparse as sp


def J(udt):
    return sp.diags((1 + udt, -udt[:-1], -udt[-1]), (0, -1, udt.shape[0] - 1))


def F(u, dt, uold):
    return u - uold + dt / 2 * (u ** 2 - np.roll(u, -1) ** 2)


def interval(n, length=1):
    dx = length / n
    return np.arange(0, n) * dx


def evalu0(func):
    def eval_wrapper(x):
        return (func(x) + func(np.roll(x, -1))) / 2

    return eval_wrapper


@evalu0
def func_u0(x):
    return 2 + 2 * np.sin(np.pi * x)


