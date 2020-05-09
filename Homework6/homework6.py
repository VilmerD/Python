from Multigrid.multigrid import v_cycle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from time import time

def a(n):
    if n > 1:
        return sp.csr_matrix((n + 1) ** 2 * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)))
    else:
        return np.array([[-8]])


nn = [31, 63]
pp = [1, 2, 3]
tol = 10 ** -9

fig, ax = plt.subplots()
axes = []
for k in [0, 1]:
    n = nn[k]
    x = np.arange(1, n + 1) / (n + 1)
    f = 4 * np.pi ** 2 * np.sin(np.pi * x ** 2)
    for j in range(0, len(pp)):
        r = []
        p = pp[j]
        v = np.zeros(n, )
        res = np.linalg.norm(a(n).dot(v) + f)
        t1 = time()
        while res > tol:
            v = v_cycle(lambda N: -a(N), v, f, pre=p, post=p)
            res = np.linalg.norm(a(n).dot(v) + f)
            r.append(res)
        t = (time() - t1) * 1000
        axes.append(plt.subplot(2, 3, 3 * k + j + 1))
        plt.semilogy(r)
        plt.title("Total time {}ms".format(np.round(t)))

axes[0].set_ylabel("n = {}".format(nn[0]))
axes[3].set_ylabel("n = {}".format(nn[1]))
for k in range(3, 6):
    axes[k].set_xlabel("{} pre and post".format(pp[k-3]))
plt.show()
