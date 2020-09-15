import numpy as np
import numpy.linalg as linalg
import scipy.linalg as splin
from scipy.sparse.linalg import LinearOperator


def project(q, a):
    return np.inner(q, a) / splin.norm(q)


def givens_matrix(a):
    r = linalg.norm(a)
    c = a[0] / r
    s = a[1] / r
    return c, s


def QT_dot_x(Q, s):
    def inner(u):
        m, n = s
        for k in np.arange(n, 0, step=-1):
            vk = Q[k - 1:, k - 1].reshape((m - k + 1, 1))
            u[k - 1:] = u[k - 1:] - 2 * vk.dot(vk.T.dot(u[k - 1:]))
        return u
    return inner


class Orthogonalization:
    def __init__(self, A):
        self.A = A
        self.s = A.shape
        self.Q = np.zeros((self.s[0], self.s[0]))
        self.R = np.zeros(self.s)
        self.G = None

    def grahm_Schmidt(self):
        m, n = self.A.shape
        for k in range(0, n):
            self.Q[:, k] = self.orthogonalize(k)

    def orthogonalize(self, k):
        a = self.A[:, k]
        vk = a
        for j in range(0, k):
            qj = self.Q[:, j]
            self.R[j, k] = project(qj, a)
            vk = vk - self.R[j, k] * qj
        self.R[k, k] = splin.norm(vk)
        return vk / self.R[k, k]

    def householder(self):
        m, n = self.s
        self.R = self.A.copy()
        for k in range(0, n):
            a = self.R[k:, k]
            a_hat = np.array([[linalg.norm(a)] + (m - k - 1) * [0.]]).reshape((m - k,))
            v_hat = a - a_hat
            vk = v_hat / linalg.norm(v_hat)

            self.Q[k:, k] = vk

            vk = vk.reshape((m - k, 1))
            self.R[k:, k:] = self.R[k:, k:] - 2 * vk.dot(vk.T.dot(self.R[k:, k:]))
        Q_hat = LinearOperator((m, m), QT_dot_x(self.Q, (m, n)))
        return self.R, Q_hat

    def givens_rotations(self):
        m, n = self.s
        self.G = [[0, 0] * (m*n - m - n + 1)]
        self.R = self.A.copy()
        for i in range(0, n - 1):
            for j in range(m - 2, i - 1, -1):
                print(j, i)
                ak = self.R[j:j + 2, i]
                c, s = givens_matrix(ak)
                self.G = None
                self.R[j:j + 2, i] = givens_matrix(ak).dot(ak)
        return self.R

