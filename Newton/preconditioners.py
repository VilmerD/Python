import scipy.sparse.linalg as splin
import scipy.sparse.extract as ex
from Linear_Solvers.multigrid import v_cycle
import numpy as np
import Linear_Solvers.smoothers as smooth


def gauss_seidel(A):
    L_D = ex.tril(A, 1)
    return lambda u: splin.spsolve(L_D, u)


def ilu(A, n):
    inv_approx = splin.spilu(A, fill_factor=4)
    return splin.LinearOperator((n, n), lambda u: inv_approx.solve(u))


def multigrid_primer(a1, pseudo_time_step):
    smoother = smooth.RungeKutta(a1, pseudo_time_step)
    n = 1

    def n_multigrid(A, v):
        x = np.zeros(v.shape)
        for k in range(0, n):
            x = v_cycle(A, x, v, smoother)
        return x

    def multigrid(A, s):
        return splin.LinearOperator((s, s), lambda x: n_multigrid(A, x))
    return multigrid
