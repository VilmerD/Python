import numpy as np
import scipy.sparse.linalg as splin


def v_cycle(a, v0, f0, smoother, gamma=1, l0=3, pre=1, post=1):
    n0 = v0.shape[0]
    level_max = int(np.log(n0) / np.log(4))
    v = np.zeros(int(4 * n0 * (1 - 4 ** (- level_max + l0 - 1)) / 3))
    f = v.copy()
    v[-n0:] = v0
    f[-n0:] = f0

    def v_cycle_recursive(level):
        n = int(4 ** (level + l0))
        i_max = int(4 ** l0 * (4 ** (level + 1) - 1) / 3)
        i_min = i_max - n

        if i_min > 0:
            for k in range(0, pre):
                v[i_min: i_max] = smoother(a, v[i_min: i_max], f[i_min: i_max])
            f[int(i_min - n / 4): i_min] = R(n) * (a(n) * v[i_min: i_max] - f[i_min: i_max])

            for g in range(0, gamma):
                v_cycle_recursive(level - 1)

            v[i_min: i_max] = v[i_min: i_max] - P(int(n / 4)) * v[int(i_min - n / 4): i_min]
            for k in range(0, post):
                v[i_min: i_max] = smoother(a, v[i_min: i_max], f[i_min: i_max])
        else:
            v[i_min: i_max] = splin.spsolve(a(n) * np.eye(n), f[i_min: i_max])
    v_cycle_recursive(level_max - l0)
    return v[-n0:]


def FAS(A, v0, f, smoother, gamma=3, level0=5, pre=30):
    grid = Grid(v0, f, level0)
    epsilon = 0.9

    def FAS_recursive(level):
        current_level = grid.levels[level]
        next_level = grid.levels[level - 1]

        for k in range(0, pre):
            current_level.v = smoother(A, current_level.v, current_level.f)

        if level > 0:
            n = current_level.v.shape[0]
            u_tilde = R(n) * current_level.v
            next_level.f = A(int(n / 2)) * u_tilde + epsilon * R(n) * (current_level.f - A(n) * current_level.v)
            for k in range(0, gamma):
                FAS_recursive(level - 1)
            current_level.v += P(int(n / 2)) * (next_level.v - u_tilde) / epsilon

    FAS_recursive(grid.n_levels - 1)
    return grid.levels[-1].v


def R(n):
    return splin.LinearOperator((int(n / 4), n), aggres2d)


def P(n):
    return splin.LinearOperator((4 * n, n), aggpro2d)


def aggres2d(v):
    n = int(v.shape[0] ** 0.5)
    vmat = v.reshape((n, n))
    mat = 0.25 * (vmat[0::2, 0::2] + vmat[1::2, 1::2] + vmat[0::2, 1::2] + vmat[1::2, 0::2])
    return mat.reshape((int(n ** 2 / 4), ))


def aggpro2d(v):
    n = int(v.shape[0] ** 0.5)
    vs = v.reshape((n, n))
    vnew = np.zeros((n * 2, n * 2))
    vnew[0::2, 0::2] = vs
    vnew[1::2, 1::2] = vs
    vnew[1::2, 0::2] = vs
    vnew[0::2, 1::2] = vs
    return vnew.reshape((4 * n ** 2, ))


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
            v = v.reshape((s[0],))
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
            v = np.zeros((n,))
            self.levels.append(self.Level(v, v))

        self.levels.append(self.Level(v0, f0))

    class Level:

        def __init__(self, v0, f0):
            self.v = v0
            self.f = f0
