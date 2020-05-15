import numpy as np
import scipy.sparse as sp
from scipy.sparse import extract
from scipy.sparse import linalg as splin


def smoothing_decorator(smoother):
    def smoothing_wrapper(A, level, n):
        for k in range(0, n):
            level.v = smoother(A, level.v, level.f)
    return smoothing_wrapper


@smoothing_decorator
def gauss_seidel(a, x, b):
    dl = extract.tril(a, 0)                             # lower plus diagonal matrix
    u = sp.csr_matrix(- extract.triu(a, 1))             # upper matrix
    return splin.spsolve_triangular(dl, u.dot(x) + b)


@smoothing_decorator
def jacobi(a, x, b, w=2/3):
    d = a.diagonal()[0]
    return x - w * (a.dot(x) - b) / d


def RungeKutta(a, dt, L):
    @smoothing_decorator
    def RK2(A, x, b):
        a1, c1 = 0.33, 0.99
        N = b.shape[0]
        h = c1 * L / (a * dt * N)

        x0 = x
        x1 = x + a1 * h * (-A(N) * x0 + b)

        return x + h * (-A(N) * x1 + b)
    return RK2