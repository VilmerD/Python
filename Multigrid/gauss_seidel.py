import numpy as np
from scipy.sparse import extract
from scipy.sparse import linalg as splin


def gauss_seidel(a, v0, f):
    if a.shape[0] > 1:
        dl = extract.tril(a, 0)                                  # lower plus diagonal matrix
        u = - extract.triu(a, 1)                                 # upper matrix

        return splin.spsolve_triangular(dl, u.dot(v0) + f)
    else:
        return splin.spsolve(a, f)


def n_gauss_seidel(a, v0, f, n):
    for k in np.arange(0, n):
        v0 = gauss_seidel(a, v0, f)
    return v0
