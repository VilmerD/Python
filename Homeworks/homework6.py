from Linear_Solvers.multigrid import v_cycle
import matplotlib.pyplot as plt
from time import time
from Matricies.matricies import *
import statistics as st
import scipy.sparse.linalg as splin


def a(n):
    if n > 1:
        return sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([[-8]])


def multigrid_tol(f, pre=1, post=3):
    tol = 10 ** -8
    n = f.shape[-1]
    u = np.zeros((n, ))
    A = -a(n)

    residual = np.linalg.norm(A.dot(u) - f)
    residuals = [residual]
    while residual/residuals[0] > tol:
        u = v_cycle(A, u, f, pre, post, mode="agg")
        residual = np.linalg.norm(A.dot(u) - f)
        print(np.log10(residual))
        residuals.append(residual)
    return residuals, u


def test_size():
    n = 2**12 - 1
    f = 4 * np.pi ** 2 * np.sin(np.pi * interval(n) ** 2)

    t1 = time()
    multigrid_tol(f)
    print(time() - t1)

    t2 = time()
    A = -a(n)
    splin.spsolve(A, f)
    print(time() - t2)


def test_agg():
    n = 128
    f = 4 * np.pi ** 2 * np.sin(np.pi * interval(n) ** 2)

    res, u = multigrid_tol(f)
    fig, ax = plt.subplots()
    plt.plot(interval(n), u)
    plt.show()


test_agg()
