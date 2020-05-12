import random as r
import numpy as np


def n_steps(n):
    """Simulates n steps of the montecarlo situation"""
    x, y = r.randint(-1, 1), r.randint(-1, 1)
    for k in range(0, n):
        if x == 0 and y == 0:
            dx, dy = r.choice(((1, 0), (0, 1), (-1, 0), (0, -1)))
            x += dx
            y += dy

        elif x == 0 or y == 0:
            dx, dy = r.choice(((y, x), (-y, -x), (-x, -y)))
            x += dx
            y += dy
        else:
            dx, dy = r.choice(((-x, 0), (0, -y)))
            x += dx
            y += dy
    return x, y


def simulate(n, m=100000):
    """Simulates n steps, m times and displays result as matrix"""
    R = np.zeros((3, 3))

    for k in range(0, m):
        x, y = n_steps(n)
        R[x + 1, y + 1] += 1
    print(R/m)


simulate(11)
