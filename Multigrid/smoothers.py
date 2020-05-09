import numpy as np
import scipy.sparse as sp
from scipy.sparse import extract
from scipy.sparse import linalg as splin


def gauss_seidel(a, x, b):
    dl = sp.csr_matrix(extract.tril(a, 0))                                 # lower plus diagonal matrix
    u = sp.csr_matrix(- extract.triu(a, 1))                                # upper matrix
    return splin.spsolve_triangular(dl, u.dot(x) + b)


def n_gauss_seidel(a, x, b, n):
    for k in np.arange(0, n):
        v0 = gauss_seidel(a, x, b)
    return v0


def jacobi(a, x, b, w):
    d = sp.csr_matrix(sp.diags(a.diagonal()))
    return x - w * splin.spsolve(d, a.dot(x) - b)


def n_jacobi(a, x, b, n, w=2/3):
    for k in range(0, n):
        x = jacobi(a, x, b, w)
    return x
