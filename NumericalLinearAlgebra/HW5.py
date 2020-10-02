import numpy as np
from scipy.linalg import *


class WrongIntervalException(Exception):
    pass


def A(p):
    A0 = np.array([[5, 0, 0, -1], [1, 0, -1, 1], [-1.5, 1, -2, 1], [-1, 1, 3, -3]])
    if p == 0:
        return np.diag(A0)
    elif p == 1:
        return A0
    elif p > 1 or p < 0:
        raise WrongIntervalException("p in wrong interval, should be [0, 1].")
    d = np.diag(A0)
    scaled = A0*p
    return scaled + (1 - p)*np.diag(d)


def task1():
    Ap = 0

A(4)