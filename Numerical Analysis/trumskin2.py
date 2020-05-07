from numpy import array, ones, zeros, arange, linspace, concatenate
from scipy.sparse import diags, linalg
import numpy as np
import matplotlib.pyplot as plt


def grid(N):
    dx = 1 / (N + 0.5)
    return linspace(dx / 2, 1 - dx, N)


def matrix(N):
    center_diagonal = concatenate(([-1], - 2 * ones(N-1)))
    t = (N + 0.5) ** 2 * diags((ones(N-1), center_diagonal, ones(N-1)), (-1, 0, 1))

    center_s = concatenate(([-1], zeros(N-1)))
    s = (N + 0.5) * diags((-ones(N-1), center_s, ones(N-1)), (-1, 0, 1)) / 2
    r = diags(1 / grid(N), 0)
    return - r * s - t


N = 67
eigenvalues, eigenvectors = np.linalg.eig(matrix(N).toarray())

print(eigenvalues)
fig, ax = plt.subplots()
ax.plot(grid(N), eigenvectors[:, 1])
plt.show()