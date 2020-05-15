import numpy as np
from Newton.newton import JFNK
from Project.project_matricies import F
from Newton.preconditioners import multigrid_primer
import matplotlib.pyplot as plt


def interval(n, length=1):
    dx = L / n
    return np.arange(0, n) * dx


def eval(func):
    def eval_wrapper(x):
        return (func(x) + func(np.roll(x, -1))) / 2
    return eval_wrapper


@eval
def func_u0(x):
    return 2 + np.sin(np.pi*x)


L=2
n = 2 ** 8
x = interval(n, length=L)
dt = 10 ** -3
u0 = func_u0(x)
u1 = JFNK(lambda u: F(u, dt, L), u0, multigrid_primer(max(u0), dt, L))
