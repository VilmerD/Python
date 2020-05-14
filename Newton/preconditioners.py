import scipy.sparse as sp
import scipy.sparse.linalg as splin
import scipy.sparse.extract as ex
from Linear_Solvers.multigrid import v_cycle
import numpy as np


def gauss_seidel(A):
    L_D = ex.tril(A, 1)
    return lambda u: splin.spsolve(L_D, u)


def ilu(A):
    inv_approx = splin.spilu(A, fill_factor=4)
    return lambda u: inv_approx.solve(u)


def nothing(A):
    return lambda u: u


def multigrid(A, n=1):
    def n_multigrid(b):
        v0 = np.zeros(b.shape)
        for k in range(0, n):
            v0 = v_cycle(A, v0, b, 1, 1)
        return v0
    return n_multigrid
