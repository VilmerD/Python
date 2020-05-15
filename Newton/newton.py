import numpy as np
import scipy.linalg as slin
import scipy.sparse.linalg as splin
from Linear_Solvers.GMRES import gmres
from Linear_Solvers.multigrid import R, P
import Newton.preconditioners as precon
from functools import lru_cache


def set_eta(eta_max, gamma, epsilon):
    def calculate_eta(last_eta, last_norm, current_norm):
        eta_a = gamma * (current_norm / last_norm) ** 2

        if gamma * last_eta ** 2 <= 0.1:
            eta_c = min(eta_max, eta_a)
        else:
            eta_c = min(eta_max, max(eta_a, gamma * last_eta ** 2))
        return min(eta_max, max(eta_c, epsilon / (2 * current_norm)))
    return calculate_eta


def NK(F, J, n, eta_max=0.1, max_it=10, M=lambda _: lambda x: x):
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
    residuals = []
    nits = []
    sols = [u_0]

    k = 0
    while r/first_norm > tol and k < max_it:
        J_ = J(u)
        F_ = F(u)
        M_ = M(J_)

        if k == 0:
            eta = eta_max
        else:
            eta = next_eta(eta, last_norm, current_norm)

        s, i, r = gmres(J_, -F_, M_, tol=eta)
        u += s

        last_norm = current_norm
        current_norm = slin.norm(F(u))

        residuals.append(r/first_norm)
        etas.append(eta)
        nits.append(i)
        sols.append(u)
        k += 1

    return residuals, sols, etas, nits


def JFNK(F, u0, M, eta_max=0.1, max_it=10):
    n = u0.shape[0]
    tol = 10 ** - 9
    gamma = 0.5
    epsilon = tol
    jacobian_epsilon = 10 ** -6
    next_eta = set_eta(eta_max, gamma, epsilon)

    u = u0.copy()

    first_norm = slin.norm(F(u0))
    last_norm = np.nan
    current_norm = first_norm
    r = current_norm
    eta = np.nan

    etas = []
    residuals = []
    nits = []
    sols = [u0]

    def approx_J(Fy, y):
        s = y.shape[0]

        def deriv(q):
            return (F(y + jacobian_epsilon * q.reshape((s, ))) - Fy) / jacobian_epsilon

        @lru_cache
        def wrapper(n):
            if n == s:
                return splin.LinearOperator((s, s), deriv)
            else:
                return R(2*n) * wrapper(2*n) * P(2*n)                       # 256 pkt, 64pkr i multigrid beräkna för 128pkt och 256
        return wrapper

    k = 0
    while r/first_norm > tol and k < max_it:
        F_ = F(u)
        J = approx_J(F_, u)

        if k == 0:
            eta = eta_max
        else:
            eta = next_eta(eta, last_norm, current_norm)

        s, i, r = gmres(J, -F_, M(J), tol=eta)
        u += s

        last_norm = current_norm
        current_norm = slin.norm(F(u))

        residuals.append(r/first_norm)
        etas.append(eta)
        nits.append(i)
        sols.append(u)
        k += 1

    return sols[-1]
