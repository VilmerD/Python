import numpy as np
import numpy.linalg as linalg
import scipy.linalg as splin


def project(q, a):
    return np.inner(q, a) / splin.norm(q)


def givens_matrix(a):
    r = linalg.norm(a)
    c = a[0] / r
    s = a[1] / r
    return c, s


def house_step(A):
    a = A[:, 0]
    m, n = A.shape
    a_hat = np.array([[linalg.norm(a)] + (m - 1) * [0.]]).reshape((m,))
    v = np.sign(a[0]) * a_hat + a
    return v / linalg.norm(v)


class Orthogonalization:
    def __init__(self, A):
        self.A = A
        self.s = A.shape
        m, n = self.s
        self.Q = np.zeros(self.s)
        self.R = np.zeros((n, n))
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
        for k in range(0, m):
            Rk = self.R[k:, k:]
            vk = house_step(Rk)
            self.Q[k:, k] = vk
            self.R[k:, k:] = Rk - 2 * vk.reshape(m-k, 1).dot(vk.reshape((1, m-k)).dot(Rk))
        return self.R, self.Q

    def givens_rotations(self):
        m, n = self.s
        self.G = [[0, 0] * (m*n - m - n + 1)]
        self.R = self.A.copy()
        for i in range(0, n - 1):
            for j in range(m - 2, i - 1, -1):
                print(j, i)
                ak = self.R[j:j + 2, i]
                c, s = givens_matrix(ak)
                self.G =
                self.R[j:j + 2, i] = givens_matrix(ak).dot(ak)
        return self.R

    def QT_dot(self, u):
        m, n = self.s
        for k in np.arange(m-1, -1, step=-1):
            vk = self.Q[k:, k]
            u[k:] = u[k:] - 2 * vk.dot(vk.dot(u[k:]))
        return u


A = np.random.rand(4, 4)
OR = Orthogonalization(A)
R = OR.givens_rotations()
print(R)
