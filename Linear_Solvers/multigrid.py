from Linear_Solvers.smoothers import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from functools import lru_cache


def v_cycle(a, v0, f, smoother, gamma=1, level0=0):
    grid = Grid(v0, f, level0)
    pre = 1
    post = 2

    def v_cycle_recursive(level):
        current_level = grid.levels[level]
        next_level = grid.levels[level - 1]
        n = current_level.f.shape[0]

        if level > 0:
            smoother(a, current_level, pre)
            next_level.f = R(n)*(a(n) * current_level.v - current_level.f)

            for g in range(0, gamma):
                v_cycle_recursive(level - 1)

            current_level.v = current_level.v - P(n)*next_level.v
            smoother(a, current_level, post)
        else:
            amat = a(1) * np.array([1, ]).T
            current_level.v = splin.spsolve(amat, current_level.f)

    v_cycle_recursive(grid.n_levels - 1)
    return grid.levels[-1].v


def R(n):
    return splin.LinearOperator((int(n / 2), n), agg_res)


def P(n):
    return splin.LinearOperator((n, int(n / 2)), agg_pro)


def galerkin(a):
    @lru_cache
    def galerkin_wrapper(n):
        if n == 0:
            return a
        else:
            return R(n)*(R(n)*galerkin_wrapper(n - 1)).T * 2
    return galerkin_wrapper


def default_res(v):
    if v.ndim != 1:
        s = v.shape
        if s[1] == 1:
            v = v.reshape((s[0],))
        else:
            return (v[:, 0:-2:2] + 2 * v[:, 1:-1:2] + v[:, 2::2]) / 4
    return (v[0:-2:2] + 2 * v[1:-1:2] + v[2::2]) / 4


def default_pro(v):
    if v.ndim != 1:
        s = v.shape
        if s[1] == 1:
            v = v.reshape((s[0], ))
        else:
            return 2 * default_res(v.T)
    u = np.zeros(len(v) * 2 + 1)
    u[1:-1:2] = v
    return (2 * u + np.roll(u, 1) + np.roll(u, -1)) / 2


def agg_res(v):
    if v.ndim != 1:
        s = v.shape
        if s[1] == 1:
            v = v.reshape((s[0],))
        else:
            return (v[:, :-1:2] + v[:, 1::2]) / 2
    return (v[:-1:2] + v[1::2]) / 2


def agg_pro(v):
    if v.ndim != 1:
        s = v.shape
        if s[1] == 1:
            v = v.reshape((s[0],))
        else:
            return 2 * agg_res(v.T)
    return np.repeat(v, 2)


class Grid:

    def __init__(self, v0, f0, level0):
        self.n_levels = int(np.log2(len(v0))) - level0 + 1
        self.levels = []
        for level in range(0, self.n_levels - 1):
            n = 2 ** (level + level0)
            v = np.zeros((n, ))
            self.levels.append(self.Level(v, v))

        self.levels.append(self.Level(v0, f0))

    class Level:

        def __init__(self, v0, f0):
            self.v = v0
            self.f = f0
