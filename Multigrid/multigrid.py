import numpy as np
import numpy.linalg as lin
from Multigrid.gauss_seidel import n_gauss_seidel
from Multigrid.matricies import interpolator1d, restrictor1d


def v_cycle(a, v0, f):
    n = len(v0)
    n_levels = int((n - 1) / 2)
    grid = Grid(n_levels + 1)
    U = v_cycle_recursive(a, grid, 2, 2, n_levels)
    return U


def v_cycle_recursive(a, grid, n1, n2, level):
    level = grid.levels[level]
    v0, f = level.unpack()
    A = a(level)
    v_tilde = None

    if level > 0:
        v_tilde = n_gauss_seidel(A, v0, f, n1)
        f_next = restrictor1d(a.dot(v0) - f)
        grid.levels[level - 1].f = f_next

        v_previous = v_cycle_recursive(a, grid, n1, n2, level - 1)

        v_tilde = v_tilde + interpolator1d(v_previous)
        v_tilde = n_gauss_seidel(A, v_tilde, f, n2)
    else:
        v_tilde = lin.solve(A.toarray(), v0)

    level.v = v_tilde
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


class Grid:

    def __init__(self, n_levels):
        self.levels = []
        for n in range(0, n_levels):
            self.levels.append(self.Level(n))
            self.n_levels = n_levels

    def n_levels(self):
        return self.n_levels

    class Level:

        def __init__(self, level):
            self.v = np.zeros(level * 2 + 1)
            self.f = np.zeros(level * 2 + 1)

        def unpack(self):
            return self.v, self.f
