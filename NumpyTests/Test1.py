import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

N = 1000
diagonals = np.array([np.ones(N-1), -2*np.ones(N), np.ones(N-1)])
pos = np.array([-1, 0, 1])

T = sparse.diags(diagonals, pos)
T = (N+1)**2*T
T = sparse.csr_matrix(T)

x = np.arange(0,1,1/(1+N))
x = x[1:]

f = np.e**(np.sin(9*np.pi*x)) - 1

sol = spsolve(T,f)
sol = np.insert(sol, [0, N], [0, 0])
x = np.insert(x, [0, N], [0, 1])

print(sol)
fig, ax = plt.subplots()
ax.plot(x, sol)
plt.show()