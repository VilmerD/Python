from numpy import array, ones, zeros, arange, linspace, concatenate
from scipy.sparse import diags, linalg, csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def x(N):
    dx = 1 / (N + 0.5)
    return linspace(dx / 2, 1 - dx, N)


def x_minus(N):
    dx = 1 / (N + 0.5)
    return linspace(0, 1 - 3 * dx / 2, N)


def x_plus(N):
    dx = 1 / (N + 0.5)
    return concatenate((x_minus(N)[1:], [1 - dx / 2]))


def matrix(N):
    center = concatenate(([1 / (N + 0.5)], (x_minus(N) + x_plus(N))[1:]))
    m = -  diags((1 / x(N)), 0) * (N + 0.5) ** 2 * diags((x_minus(N)[1:], - center, x_plus(N)[:-1]), (-1, 0, 1))
    return m


def iterations(uold, M):
    dt = 1 / M
    u = np.copy(uold).reshape(N, 1)
    for k in linspace(1, M, M):
        uold = u[:, int(k - 1)]
        unew = iteration(uold, dt).reshape(N, 1)
        u = np.concatenate((u, unew), axis=-1)
    return u


def iteration(uold, dt):
    A_plus = csr_matrix(np.eye(N) + dt / 2 * matrix(N))
    A_minus = csr_matrix(np.eye(N) - dt / 2 * matrix(N))
    return linalg.spsolve(A_minus, A_plus * uold)


def plotter(x, y, data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _ = ax.plot_surface(x, y, data)
    plt.show()


N = 67
M = 10
uold = np.cos(np.pi*x(N))
u = iterations(uold, M)
x = x(N)
t = linspace(0, 1, M + 1)
T, X = np.meshgrid(t, x)

plotter(X, T, u)