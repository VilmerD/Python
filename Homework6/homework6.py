from Multigrid.multigrid import v_cycle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from time import time
from Matricies.matricies import *
import statistics as st


def a(n):
    if n > 1:
        return sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([[-8]])


def timeit(func, n=100):
    times = []
    for l in range(0, n):
        t1 = time()
        func()
        times.append((time() - t1))
    return times


def multigrid_tol(len, pre, post):
    tol = 10 ** -8
    f = 4 * np.pi ** 2 * np.sin(np.pi * interval(len) ** 2)
    u = np.zeros((len, ))
    func = lambda n: -a(n)

    residuals = []
    residual = np.linalg.norm(func(len).dot(u) - f)
    while residual > tol:
        u = v_cycle(func, u, f, pre=pre, post=post)
        residual = np.linalg.norm(func(len).dot(u) - f)
        residuals.append(residual)
    return residuals, u


def figs():
    nn = [31, 63]
    pp = [1, 2, 3]

    fig, ax = plt.subplots()
    axes = []
    for k in [0, 1]:
        n = nn[k]

        for j in range(0, len(pp)):
            p = pp[j]
            r, u = multigrid_tol(n, p, p)
            axes.append(plt.subplot(2, 3, 3 * k + j + 1))
            plt.semilogy(r)
            t = timeit(lambda: multigrid_tol(n, p, p), n=10)
            print("St.Dev: {} for n: {} and p: {}".format(st.pstdev(t), n, p))
            plt.title("Total time {}ms".format(np.round(st.mean(t)*1000)))

    axes[0].set_ylabel("n = {}".format(nn[0]))
    axes[3].set_ylabel("n = {}".format(nn[1]))
    for k in range(3, 6):
        axes[k].set_xlabel("{} pre and post".format(pp[k-3]))
    plt.show()


def fun():
    n = 7
    A = sp.eye(n)
    precon = v_cycle(lambda N: -a(N), np.zeros((n, n)), A)
    precon.dot(-a(n).toarray())

fun()

