import numpy as np
import numpy.linalg as lin
from Multigrid.smoothers import *
from Multigrid.matricies import interpolator1d, restrictor1d


def v_cycle(a, v0, f, pre=3, post=3):
    grid = Grid(v0, f)
    v = v_cycle_recursive(a, grid, pre, post, grid.n_levels - 1)
    return v


def v_cycle_recursive(a, grid, n1, n2, level):
    current_level = grid.levels[level]
    v0, f = current_level.unpack()
    A = a(level)
    v_tilde = None

    if level > 0:
        v_tilde = n_jacobi(A, v0, f, n1)
        v_tilde = n_gauss_seidel(A, v_tilde, f, n1)
        f_next = restrictor1d(A.dot(v_tilde) - f)

        grid.levels[level - 1].f = f_next

        v_previous = v_cycle_recursive(a, grid, n1, n2, level - 1)

        v_tilde = v_tilde - interpolator1d(v_previous)
        v_tilde = n_jacobi(A, v_tilde, f, n2)
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


class Grid:

    def __init__(self, v0, f0):
        n_levels = int(np.log2(len(v0) + 1))
        self.levels = []
        for l in range(0, n_levels - 1):
            n = 2 ** (l + 1) - 1
            self.levels.append(self.Level(np.zeros(n, ), np.zeros(n, )))
        self.n_levels = n_levels
        self.levels.append(self.Level(v0, f0))

    def get_solutions(self):
        solutions = []
        for level in self.levels:
            solutions.append(level.v)
        return solutions

    class Level:

        def __init__(self, v0, f0):
            self.v = v0
            self.f = f0

        def unpack(self):
            return self.v, self.f
