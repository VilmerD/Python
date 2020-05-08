import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
from Matricies.matricies import *
from Multigrid.smoothers import *
import scipy.linalg as slin
import matplotlib.pyplot as plt


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
        vnew = np.zeros(int((n - 1) / 2))
        for k in np.arange(0, int((n - 1) / 2)):
            vnew[k] = (v[2 * k] + 2 * v[2 * k + 1] + v[2 * k + 2]) / 4
        return vnew


def twogrid(Afun, b, x0=None, pre=2, post=2):
    n = len(b)
    n_l = int((n - 1) / 2)
    A = Afun(n)
    D = sp.csr_matrix(sp.diags(A.diagonal()))
    if x0 is None:
        x0 = np.zeros((n, ))

    xtilde = njacobi(A, D, x0, b, 1, pre)
    x_biss = A.dot(xtilde)
    r = x_biss - b
    rl_1 = restrictor(r)
    el_1 = splin.spsolve(Afun(n_l), rl_1)
    xtilde = xtilde - interpolator(el_1)
    xtilde = njacobi(A, D, xtilde, b, 1, post)

    return xtilde


def discrete_second(n, L=1):
    dx = L / (n + 1)
    return sp.csr_matrix(dx ** -2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))


def source(n):
    x = interval(n)
    return 4 * np.pi ** 2 * np.sin(np.pi * x ** 2).reshape((n,))
