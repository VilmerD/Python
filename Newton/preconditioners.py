import scipy.sparse as sp
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


def multigrid_primer(a1, pseudo_timestep, pre, post):
    smoother = smooth.RungeKutta(a1, pseudo_timestep)

    def multigrid(A, s):
        v = np.zeros((s, ))
        return splin.LinearOperator((s, s), lambda x: v_cycle(A, v, x, smoother, pre=pre, post=post))
    return multigrid
