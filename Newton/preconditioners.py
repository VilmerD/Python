import scipy.sparse as sp
import scipy.sparse.linalg as splin
import scipy.sparse.extract as ex
from Linear_Solvers.multigrid import v_cycle
import numpy as np
import Linear_Solvers.smoothers as smooth


def gauss_seidel(A):
    L_D = ex.tril(A, 1)
    return lambda u: splin.spsolve(L_D, u)


def ilu(A):
    inv_approx = splin.spilu(A, fill_factor=4)
    return lambda u: inv_approx.solve(u)


def multigrid_primer(a, dt, L):
    smoother = smooth.RungeKutta(a, dt, L)

    def multigrid(A):
        def multigrid_wrapper(n):
            return splin.LinearOperator((n, n), lambda x: v_cycle(A, np.zeros(x.shape), x, smoother))
        return multigrid_wrapper
    return multigrid
