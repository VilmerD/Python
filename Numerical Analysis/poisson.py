import numpy as np
from scipy.sparse import linalg
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def big_t(n, bt):
    diagonal = 4*np.ones(n**2)
    sub_diagonal = np.tile(np.concatenate([np.ones(n-1), np.array([0])]), n)
    super_diagonal = np.concatenate([[1], sub_diagonal])[:-1]
    superduper_diagonal = np.ones(n ** 2)
    suberduperduper_diagonal = np.ones(n ** 2)

    boundary_matrix = get_boundary_matrix(n, bt)

    return n**2*(sparse.spdiags([diagonal, -sub_diagonal, -super_diagonal, -suberduperduper_diagonal,
                                 -superduper_diagonal], [0, -1, 1, -n, n], n**2, n**2).tocsr() + boundary_matrix)


def get_boundary_matrix(n, bt):
    onev = np.ones(n)
    west_diagonal = n * np.arange(0, n)
    west_m = sparse.csr_matrix((bt[0]*onev, (west_diagonal, west_diagonal)), shape=(n**2, n**2))
    north_diagonal = np.arange(0, n)
    north_m = sparse.csr_matrix((bt[2]*onev, (north_diagonal, north_diagonal)), shape=(n ** 2, n ** 2))
    east_diagonal = n * np.arange(1, n + 1) - 1
    east_m = sparse.csr_matrix((bt[1]*onev, (east_diagonal, east_diagonal)), shape=(n ** 2, n ** 2))
    south_diagonal = np.arange(n ** 2 - n, n ** 2)
    south_m = sparse.csr_matrix((bt[3]*onev, (south_diagonal, south_diagonal)), shape=(n ** 2, n ** 2))

    return west_m + north_m + east_m + south_m


def get_boundary_columns(data, b):
    west, north, east, south = data
    n = len(west)

    westc = sparse.csr_matrix((west*b[0], (n * np.arange(0, n), np.zeros(n))), shape=(n ** 2, 1))
    northc = sparse.csr_matrix((north*b[2], (np.arange(0, n), np.zeros(n))), shape=(n ** 2, 1))
    eastc = sparse.csr_matrix((east*b[1], (n * np.arange(1, n + 1) - 1, np.zeros(n))), shape=(n ** 2, 1))
    southc = sparse.csr_matrix((south*b[3], (np.arange(n ** 2 - n, n**2), np.zeros(n))), shape=(n ** 2, 1))

    return (westc + northc + eastc + southc)*n**2



def f(x, y):
    x = np.reshape(x, (len(x)**2, 1))
    y = np.reshape(y, (len(y)**2, 1))
    return 100*np.exp(-100*(x - 0.5)**2 - 100*(y - 0.5)**2)*0


def solver(matrix, f):
    return linalg.spsolve(matrix, f)


def plotter(x, y, data):
    N = len(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _ = ax.plot_surface(x, y, sol)
    plt.show()


def meshgrid(N):
    nx, ny = (N, N)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv_inner, yv_inner = np.meshgrid(x, y)
    return xv_inner, yv_inner


def b_constants(alpha, beta, N):
    beta = beta * -1 ** np.arange(0,4)
    print(beta)
    b = 2 / (alpha - N * beta)
    bt = b * (alpha + N * beta) / 2
    return b, bt


N = 200
boundaries = np.sin(3*np.pi*np.linspace(0, 1, N)) + 1, 2*np.linspace(1, -1, N), -np.sin(3*np.pi*np.linspace(0, 1, N)) + 1, 2*np.linspace(1, -1, N)
b, bt = b_constants(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), N)

x, y = meshgrid(N)
f_on_grid = sparse.csr_matrix(f(x, y)) + get_boundary_columns(boundaries, b)
matrix = big_t(N, bt)

sol = solver(matrix, f_on_grid).reshape(N, N)
plotter(x, y, sol)
