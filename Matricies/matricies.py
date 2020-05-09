import numpy as np
import scipy.sparse as sp


def interval(n, L=1):
    dx = L / (n + 1)
    return (np.arange(1, n + 1) * dx).reshape((n, ))


def interpolator(v):
    n = len(v)
    u = np.zeros(n + 2)
    u[1: -1] = v.reshape(n)
    v_new = np.zeros(2 * n + 1)

    for k in np.arange(0, n):
        v_new[2 * k] = (u[k + 1] + u[k]) / 2
        v_new[2 * k + 1] = u[k + 1]
    v_new[-1] = v[-1] / 2
    return v_new


def restrictor(v):
    n = len(v)
    if n > 1:
        v_new = np.zeros(int((n - 1) / 2))
        for k in np.arange(0, int((n - 1) / 2)):
            v_new[k] = (v[2 * k] + 2 * v[2 * k + 1] + v[2 * k + 2]) / 4
        return v_new


def discrete_second(n, L=1):
    dx = L / (n + 1)
    return sp.csr_matrix(dx ** -2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))


