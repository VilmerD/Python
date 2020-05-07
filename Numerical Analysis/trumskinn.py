import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt


def grid(N):
    return np.linspace(0, 1, N + 1)[:-1]


def matrix(N):
    x_n = np.linspace(0, 1, N + 1)
    super_diagonal = np.concatenate(([0], x_n[:-1]))
    sub_diagonal = np.concatenate((x_n[1:], [0]))
    u_dblprime = N ** 2 * sparse.spdiags([super_diagonal, -2*x_n, sub_diagonal], [1, 0, -1], N, N)
    u_prime = N / 2 * sparse.spdiags([np.concatenate(([0, 0], np.ones(N - 2))), - np.ones(N)], [1, -1], N, N)
    return - u_dblprime - u_prime


def plotter(N, k):
    m = matrix(N)
    eigenvalues, eigenvectors = linalg.eigs(m, k + 1)
    fig, ax = plt.subplots()
    ax.plot(grid(N), (eigenvectors[:, N - (k-1)]))
    plt.show()

N = 300
plotter(N, N - 3)
