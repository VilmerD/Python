import numpy as np
import scipy.linalg as splin


def gmres(A, b, M, tol=10 ** -9, x0=None, k_max=None):
    m = int(b.shape[0])
    nits = 0
    if x0 is None:
        x0 = np.zeros((m, ))
    x = x0

    if k_max is None:
        k_max = 40
    else:
        k_max = min(m, k_max)

    norm0 = splin.norm(b)
    r0 = b - A * x
    gamma = [splin.norm(r0)]
    v = [r0 / gamma[0]]
    h = np.zeros((k_max + 1, k_max))

    c = np.zeros(k_max)
    s = np.zeros(k_max)
    nits = 0
    if splin.norm(r0) == 0:
        return x
    else:
        for j in range(0, k_max):
            wj = A * M * v[j]
            for i in range(0, j + 1):
                h[i, j] = v[i].T.dot(wj)
                wj -= h[i, j] * v[i]

            h[j + 1, j] = splin.norm(wj)
            for i in range(0, j):
                h[i:i + 2, j] = np.array([[c[i], s[i]], [-s[i], c[i]]]).dot(h[i:i + 2, j])

            beta = np.sqrt(h[j, j] ** 2 + h[j + 1, j] ** 2)
            s[j] = h[j + 1, j] / beta
            c[j] = h[j, j] / beta
            h[j, j] = beta

            gamma.append(-s[j]*gamma[j])
            gamma[j] = c[j]*gamma[j]

            if np.abs(gamma[j + 1]/norm0) >= tol:
                v.append(wj / h[j + 1, j])
            else:
                nits = j + 1
                alpha = np.zeros((j + 1, 1))
                for i in range(j, -1, -1):
                    alpha[i] = (gamma[i] - h[i, i + 1: j + 1].dot(alpha[i + 1:j + 1])) / h[i, i]
                    x += alpha[i] * v[i]
                nits = j + 1
                break
        if nits == 0:
            j = -1

    return M*x, nits, abs(gamma[-1])
