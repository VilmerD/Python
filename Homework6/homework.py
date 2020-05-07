from Multigrid.multigrid import v_cycle
import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp


def a(level):
    n = level * 2 + 1
    if level > 1:
        return sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([-8])


