import numpy as np
import scipy.sparse as s
import numpy.linalg as lin
from Multigrid.smoothers import *
from Multigrid.matricies import interpolator1d, restrictor1d


def matrix_to_func(a):
    lookup = {a.shape[0]: a.toarray()}

    def a_func(n):
        t = a.toarray()
        if n not in lookup:
            t_prev = lookup[n * 2 + 1]
            lookup[n] = restrictor(2 * restrictor(t_prev.T).T)
        return lookup[n]
    return a_func


def v_cycle(a, v0, f, pre=2, post=2, gamma=2):
    grid = Grid(v0, f)
    A = matrix_to_func(a) if s.isspmatrix(a) else a
    v = v_cycle_recursive(A, grid, pre, post, grid.n_levels - 1, gamma)
    return v


def v_cycle_recursive(a, grid, n1, n2, level, gamma):
    current_level = grid.levels[level]
    v0, f = current_level.unpack()
    A = a(2 ** (level + 1) - 1)
    v_tilde = None

    if level > 0:
        # v_tilde = n_jacobi(A, v0, f, n1)
        v_tilde = n_gauss_seidel(A, v0, f, n1)
        f_next = restrictor(A.dot(v_tilde) - f)

        grid.levels[level - 1].f = f_next

        v_previous = np.nan
        for g in range(0, gamma):
            v_previous = v_cycle_recursive(a, grid, n1, n2, level - 1, gamma)
        v_tilde = v_tilde - interpolator(v_previous)
        # v_tilde = n_jacobi(A, v_tilde, f, n2)
        v_tilde = n_gauss_seidel(A, v_tilde, f, n2)
    else:
        v_tilde = lin.solve(A, f)

    current_level.v = v_tilde
    return v_tilde


def full_multigrid(a, f, n0):
    return full_multigrid_recursive(a, f, f, n0)


def full_multigrid_recursive(a, v0, f, n0):
    n = len(f)
    if n > 1:
        f_next = restrictor1d(f)
        v_previous = full_multigrid_recursive(a, v0, f_next, n0)
        v0 = interpolator1d(v_previous)
    else:
        v0 = np.array([0])
    # noinspection PyUnusedLocal
    for k in np.arange(0, n0):
        v0 = v_cycle(a, v0, f)
    return v0


def restrictor(v):
    if v.ndim == 1:
        return restrictor1d(v)
    else:
        n, m = v.shape
        n_new = int((n - 1) / 2)
        u = np.zeros((n_new, m))
        for k in range(0, m):
            u[:, k] = restrictor1d(v[:, k])
        return u


def interpolator(v):
    if v.ndim == 1:
        return interpolator1d(v)
    else:
        n, m = v.shape
        n_new = n * 2 + 1
        u = np.zeros((n_new, m))
        for k in range(0, m):
            u[:, k] = interpolator1d(v[:, k])
        return u


class Grid:

    def __init__(self, v0, f0):
        n_levels = int(np.log2(len(v0) + 1))
        self.levels = []
        for l in range(0, n_levels - 1):
            n = 2 ** (l + 1) - 1
            v = np.zeros((n, ))
            self.levels.append(self.Level(v, v))
        self.n_levels = n_levels
        self.levels.append(self.Level(v0, f0))

    class Level:

        def __init__(self, v0, f0):
            self.v = v0
            self.f = f0

        def unpack(self):
            return self.v, self.f
