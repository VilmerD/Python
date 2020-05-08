from Multigrid.multigrid import v_cycle
from Matricies.matricies import *
import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
import matplotlib.pyplot as plt


def a(level):
    n = 2 ** (level + 1) - 1
    if level > 0:
        return sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([[-8]])


n = 63
n_level = int(np.log2(n + 1) - 1)
x = np.arange(1, n + 1) / (n + 1)

f = 4 * np.pi ** 2 * np.sin(np.pi * x ** 2)
v = np.zeros(n, )

for k in range(0, 6):
    v = v_cycle(lambda N: -a(N), v, f)
    res = np.linalg.norm(a(n_level).dot(v) + f)
    print(res)

fig, ax = plt.subplots()
plt.plot(x, v)
plt.show()
