from Multigrid.multigrid import v_cycle
import numpy as np
import scipy.sparse.linalg as splin
import scipy.sparse as sp
import matplotlib.pyplot as plt


def a(level):
    n = 2 ** (level + 1) - 1
    if level > 0:
        return - sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([[8]])

n = 63
x = np.arange(1, n + 1) / (n + 1)
tol = 10 ** -2
f = np.pi ** 2 * np.sin(np.pi*x)
v = np.zeros(63, )
for k in range(0, 3):
    v = v_cycle(a, v, f)
    print(np.linalg.norm(v))

fig, ax = plt.subplots()
plt.plot(x, v - np.sin(np.pi*x), 'r')
plt.show()
