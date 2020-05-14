import numpy as np
import scipy.sparse as sp


def J(udt):
    return sp.diags((1 + udt, -udt[:-1], -udt[-1]), (0, -1, udt.shape[0] - 1))


def F(u, dt, uold):
    return u - uold + dt / 2 * (u ** 2 - np.roll(u, 1) ** 2)

