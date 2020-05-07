import scipy.sparse as sp
import numpy as np

# Tar fram vektor som motsvarar x i intervallet (0, 1)
def interval(n):
    return (np.arange(1, n + 1) / (n + 1)).reshape((n, 1))


# Matrisen som motsvarar diskretisering av andraderivatan
def T(n):
    diagonal = - 2 * np.ones(n)
    sub_diagonal = np.ones(n - 1)
    return (n + 1) ** 2 * sp.csr_matrix(sp.diags((sub_diagonal, diagonal, sub_diagonal), (-1, 0, 1), (n, n)))


# Källtermen , dvs f(x) = 2 - sin(x(1-x))
def source(n):
    x = interval(n)
    return 2 * np.ones((n, 1)) - 9*np.sin(x*(1 - x))


# F, vilket är vektorn som sätts till noll i newton
def F(u):
    n = len(u)
    u = u.reshape((n, 1))
    return T(n).dot(u) + source(n) + np.sin(u)


# Jacobian-matrisen
def jacobian(u):
    n = len(u)
    u = u.reshape((n, 1))
    return T(n) + sp.csr_matrix(sp.diags(np.cos(u).reshape(n, )))