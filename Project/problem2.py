from Matricies.matricies import xrange
import numpy as np
import matplotlib.pyplot as plt


def interval(n, L=1):
    dx = L / n
    return np.arange(0, n) * dx


def eval(func):
    def eval_wrapper(x):
        return (func(x) + func(np.roll(x, -1))) / 2
    return eval_wrapper


@eval
def u0(x):
    return 2 + np.sin(np.pi*x)


def matvec(F, epsilon):
    def wrapper_matvec(y, q):
        return

n = 256
x = interval(n, L=2)
fig, ax = plt.subplots()
plt.plot(x, u0(x))
plt.show()

