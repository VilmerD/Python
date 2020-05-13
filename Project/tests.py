from Homework3.GMRES import gmres
from Matricies.matricies import *
from Homework2.matricies import *
from Homework2.newton import *
from Multigrid.multigrid import *
import matplotlib.pyplot as plt
import numpy as np


def multigrid_preconditioner(A, n=2):
    def n_multigrid(b):
        v0 = np.zeros(b.shape)
        for k in range(0, n):
            v0 = v_cycle(A, v0, b)
        return v0
    return n_multigrid


n = 127
residuals, sols, etas, nits = newton(F, J, n, M=multigrid_preconditioner)
fix, ax = plt.subplots()
plt.semilogy(abs(residuals))
plt.show()
