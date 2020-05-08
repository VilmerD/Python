import numpy as np


def interval(n, L=1):
    dx = L / (n + 1)
    return (np.arange(1, n + 1) * dx).reshape((n, ))

