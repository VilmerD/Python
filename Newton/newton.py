import numpy as np
import scipy.linalg as slin
import scipy.sparse.linalg as splin
import scipy.sparse as sp
from Linear_Solvers.GMRES import gmres
from Linear_Solvers.multigrid import R, P


def set_eta(eta_max, gamma, epsilon):
    def calculate_eta(last_eta, last_norm, current_norm):
        eta_a = gamma * (current_norm / last_norm) ** 2

        if gamma * last_eta ** 2 <= 0.1:
            eta_c = min(eta_max, eta_a)
        else:
            eta_c = min(eta_max, max(eta_a, gamma * last_eta ** 2))
        return min(eta_max, max(eta_c, epsilon / (2 * current_norm)))
    return calculate_eta


def NK(F, J, n, eta_max=0.1, max_it=10, M=lambda _, __: lambda x: x):
    tol = 10 ** - 10
    gamma = 0.5
    epsilon = tol
    next_eta = set_eta(eta_max, gamma, epsilon)

    u_0 = np.zeros((n, ))
    u = u_0.copy()

    first_norm = slin.norm(F(u_0))
    last_norm = np.nan
    current_norm = first_norm
    r = current_norm
    eta = np.nan

    etas = []
    residuals = [first_norm]
    nits = []
    sols = [u_0]

    k = 0
    while r/first_norm > tol and k < max_it:
        J_ = J(u)
        F_ = F(u)
        M_ = M(J_, n)

        if k == 0:
            eta = eta_max
        else:
            eta = next_eta(eta, last_norm, current_norm)

        s, i, r = gmres(splin.aslinearoperator(J_), -F_, M_, tol=eta)
        u += s

        last_norm = current_norm
        current_norm = slin.norm(F(u))

        residuals.append(r/first_norm)
        etas.append(eta)
        nits.append(i)
        sols.append(u)
        k += 1

    return residuals, sols, etas, nits


def JFNK(Ab, u0, eta_max=0.1, max_it=10):
    n = u0.shape[0]
    tol = 10 ** - 9
    epsilon = tol
    gamma = 0.5
    next_eta = set_eta(eta_max, gamma, epsilon)

    u = u0.copy()

    first_norm: float = slin.norm(F(u0))
    last_norm: float = np.nan
    current_norm: float = first_norm
    eta: float = eta_max

    nits = [0]
    residuals = [first_norm]
    j = 0
    while current_norm/first_norm > tol and j < max_it:
        s, i, r = gmres(Ab, tol=eta, k_max=30)
        u += s

        Ab.setRhs(u)
        current_norm = slin.norm(F(u))
        eta = next_eta(eta, last_norm, current_norm)
        last_norm = current_norm
        nits.append(i)
        residuals.append(r)
        j += 1
        print("Newton({}), nGMRES({}), q = {}".format(j, i, int(np.log10(current_norm/first_norm))))

    return u


class LinearSystem(object):

    def __init__(self, E, h, dimensions=2, M=None):
        self.n = h.shape[0]
        self.b = h

        if callable(E):
            self.Eh = E(self.b)
            self.E = E
            self.Matrix = False

        elif sp.issparse(E) or isinstance(E, np.ndarray):
            self.E = splin.aslinearoperator(E)
            self.Matrix = True

        self.M = M
        self.dimensions = dimensions

    def setRHS(self, h):
        self.b = h
        if not self.Matrix:
            self.Eh = self.E(h)

    def jacobian(self, k):
        s = self.n
        d = self.dimensions

        def derivative(q):
            norm_q = slin.norm(q)
            j_eps = 10 ** (-7) / norm_q if norm_q != 0 else 1
            return (self.E(self.b + j_eps * q.reshape((s, ))) - self.Eh) / j_eps

        if k == s:
            return splin.LinearOperator((s, s), derivative)
        else:
            return R(d * 2 * k) * self.jacobian(d * 2 * k) * P(k)

    def residual(self, x):
        return self.E * x - self.b

    def __mul__(self, other):
        if self.Matrix:
            return self.E * other
        else:
            return self.jacobian(other.shape[0]) * other

    def fNorm(self, b):
        return

    def AM(self, other):
        b = other
        if self.M is not None:
            b = self.M * other
        return self * b
