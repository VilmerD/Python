import numpy as np
import scipy.linalg as slin
from Linear_Solvers.GMRES import gmres


def set_eta(eta_max, gamma, epsilon):
    def calculate_eta(last_eta, last_norm, current_norm):
        eta_a = gamma * (current_norm / last_norm) ** 2

        if gamma * last_eta ** 2 <= 0.1:
            eta_c = min(eta_max, eta_a)
        else:
            eta_c = min(eta_max, max(eta_a, gamma * last_eta ** 2))
        return min(eta_max, max(eta_c, epsilon / (2 * current_norm)))
    return calculate_eta


def newton(F, J, n, eta_max=0.1, max_it=10, M=lambda _: lambda x: x):
    tol = 10 ** - 9
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


def jacobian_free(F, n, eta_max=0.1, max_it=10, M=lambda _: lambda x: x):
    tol = 10 ** - 9
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
        F_ = F(u)
        # M_ = M(J_)

        if k == 0:
            eta = eta_max
        else:
            eta = next_eta(eta, last_norm, current_norm)

        s, i, r = gmres(-F_, M, tol=eta)
        u += s

        last_norm = current_norm
        current_norm = slin.norm(F(u))

        residuals.append(r/first_norm)
        etas.append(eta)
        nits.append(i)
        sols.append(u)
        k += 1

    return residuals, sols, etas, nits
