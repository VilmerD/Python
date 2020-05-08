import numpy as np
import scipy.sparse as sp
from scipy.sparse import extract
from scipy.sparse import linalg as splin


def gauss_seidel(a, v0, f):
    dl = extract.tril(a, 0)                                  # lower plus diagonal matrix
    u = - extract.triu(a, 1)                                 # upper matrix
    return splin.spsolve_triangular(dl, u.dot(v0) + f)


def n_gauss_seidel(a, v0, f, n):
    for k in np.arange(0, n):
        v0 = gauss_seidel(a, v0, f)
    return v0


def jacobi(a, x, b, w):
    d = sp.diags(a.diagonal())
    return x - w * splin.spsolve(d, a.dot(x) - b)


def n_jacobi(a, x, b, n, w=1):
    for k in range(0, n):
        x = jacobi(a, x, b, w)
    return x
