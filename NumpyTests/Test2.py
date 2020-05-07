import numpy as np
from scipy.sparse import linalg
from scipy import sparse
import matplotlib.pyplot as plt


N = 100


def newton_f(u):
    n = len(u)
    u_n_plus_1 = u[-1] + 2 / n
    u_zero = 0

    u = np.concatenate([[u_zero], u, [u_n_plus_1]])
    u_super = u[2:]
    u_sub = u[:-2]
    u_center = u[1:-1]
    return (u_sub - 2 * u_center + u_super) * (n ** 2) - u_center * (u_sub - u_super) * n / 2 + u_center


def newton_f_prime(u):
    n = len(u)
    u_n_plus_1 = u[-2] + 2 / n
    u_zero = 0

    u = np.concatenate([[u_zero], u, [u_n_plus_1]])
    u_super = u[2:]
    u_sub = u[:-2]
    u_center = u[1:-1]

    f_super = n ** 2 + u_center / 2 * n
    f_sub = n ** 2 - u_center / 2 * n
    f_diag = -2 * n ** 2 - (u_sub - u_super) * n / 2 + 1
    f_sub[-2] = 2*n**2
    return sparse.diags([f_diag, f_sub, f_super], [0, -1, 1])


def newton(f, f_prime, uold):
    return uold - linalg.spsolve(f_prime(uold), f(uold))


def newton_iterate(f, f_prime, first, n):
    uold = first
    for i in range(1, n):
        uold = newton(f, f_prime, uold)
    return uold


x = np.arange(0, 1, 1/N)
first_guess = np.dot(-2 * np.exp(-0.5*x + 0.5), np.sin(np.pi/2*x).discrete_second)
print(first_guess.size)
print((first_guess[-1] - first_guess[-3])/2*N)

u = newton_iterate(newton_f, newton_f_prime, first_guess, 200)
u = np.concatenate([[0], u])

last_elements = u[-3:]
print((last_elements[2] - last_elements[0])/2*N)

fig, ax = plt.subplots()
ax.plot(np.arange(0, 1 + 1/N, 1/N), u)
plt.show()
