import numpy as np
import scipy.linalg as scl
import time

class Otrhogonalization:
    def __init__(self, matrix):
        self.type = 'float32' # 'float64'
        matrix = matrix.astype(self.type)
        self.matrix = np.array(matrix)

    def __testOrth(f):
        def wrap(*args, **kwargs):
            print("-- TestOrth: {:s} --".format(f.__name__))
            time1 = time.time()

            Q = f(*args, **kwargs)
            if(any(np.isnan(x) for x in Q.flatten())):
                print("Found nan in Q.")
                return None
            if(any(np.isinf(x) for x in Q.flatten())):
                print("Found inf in Q.")
                return None

            print('{:s} took {:.3f} ms'.format(f.__name__, (time.time() - time1) * 1000.0))
            print("n =", len(Q.T))
            print("m =", len(Q))
            print("2norm =", scl.norm(Q, ord=2))
            QTQ = np.dot(Q.T, Q)
            print("2norm(Q.T Q - I) =", scl.norm(QTQ - np.identity(len(QTQ)), ord=2))
            ev = scl.eigvals(QTQ)
            print("1st eigval of Q.T Q =", ev[0], " - Last eigval of Q.T Q =", ev[len(ev) - 1])
            print("2nomr([1,...,1] - eigvals) =", scl.norm(np.ones(len(ev),) - ev, ord=2))
            print("Det(Q.T Q) =", scl.det(QTQ))
            print("--  --")
            return Q
        return wrap

    def __testQR(f):
        def wrap(*args, **kwargs):
            print("-- TestQR: {:s} --".format(f.__name__))
            time1 = time.time()

            Q, R = f(*args, **kwargs)
            M = args[0].matrix

            print('{:s} took {:.3f} ms'.format(f.__name__, (time.time() - time1) * 1000.0))
            print()
            print("Q:\n", Q)
            print()
            print("R:\n", R)
            print()
            QR = Q.dot(R)
            print("QR:\n", QR)
            print()
            print(M)
            print()
            print("2norm(Q.R - M) =", scl.norm(QR - M, ord=2))

            print("--  --")
            return Q, R
        return wrap

    def gramschmidt_cl_mute(self):
        columns = np.copy(self.matrix).T
        for i in range(len(columns)):
            sum = 0
            for j in range(0, i):
                sum += np.dot(columns[j], columns[i]) * columns[j]
            columns[i] -= sum
            columns[i] = self.__norm(columns[i])
        return columns.T

    @__testOrth
    def gramschmidt_cl(self):
        return self.gramschmidt_cl_mute()

    def gramschmidt_mute(self):
        columns = np.copy(self.matrix).T
        for i in range(len(columns)):
            for j in range(0, i):
                columns[i] -= np.dot(columns[j], columns[i]) * columns[j]
            columns[i] = self.__norm(columns[i])
        return columns.T

    @__testOrth
    def gramschmidt(self):
        return self.gramschmidt_mute()

    def QR_mute(self):
        return scl.qr(self.matrix)

    @__testQR
    def QR(self):
        return self.QR_mute()

    @__testOrth
    def QR_orth(self):
        return self.QR_mute()[0]

    def householder_mute(self):
        R = np.copy(self.matrix)
        m, n = R.shape
        Q = np.identity(m)
        for k in range(n):
            x = R[k: m, k]
            v_k = self.__norm(np.sign(x[0]) * scl.norm(x, ord=2) * self.__makeE(len(x)) + x)
            F_hat = 2 * np.outer(v_k, v_k)
            R[k: m, k: n] = R[k: m, k: n] - F_hat.dot(R[k: m, k: n])
            Q_k = np.identity(m)
            Q_k[k: m, k: m] = Q_k[k: m, k: m] - F_hat
            Q = Q.dot(Q_k)
        Q.astype(self.type)
        R.astype(self.type)
        return Q, R

    @__testQR
    def householder(self):
        return self.householder_mute()

    @__testOrth
    def householder_orth(self):
        return self.householder_mute()[0]

    def givens_mute(self):
        R = np.copy(self.matrix)
        m, n = R.shape
        Q = np.identity(m)
        for j in range(n):
            k = n - j
            for i in range(j + 1, m):
                O = np.identity(m)
                theta = np.arctan(- R[i,j] / R[j,j])
                c = np.cos(theta)
                s = np.sin(theta)
                O[j, j] = c
                O[i, j] = s
                O[j, i] = -s
                O[i, i] = c
                R = O.dot(R)
                Q = O.dot(Q)
        return Q.T, R

    @__testQR
    def givens(self):
        return self.givens_mute()

    @__testOrth
    def givens_orth(self):
        return self.givens_mute()[0]

    def __makeE(self, n):
        e = np.zeros(n)
        e[0] = 1
        return e

    def __norm(self, v):
        return v / scl.norm(v, ord=2)

# M = np.array([[3,2],[1,2]])
# print(M)
# orth = Otrhogonalization(M)
# print(orth.QR())

def testOrthogonalizations():
    for n in [1, 10, 20, 50, 100, 200, 500, 1000]:
        print("================",n,"================")
        print()
        M = np.random.rand(n + 2, n)
        orth = Otrhogonalization(M)
        # Q1 = orth.QR_orth()
        # print()
        # Q2 = orth.gramschmidt_cl()
        # print()
        # Q3 = orth.gramschmidt()
        # print()
        Q4 = orth.householder_orth()
        print()
        Q5 = orth.givens_orth()
        print()

def testQR():
    M = np.random.rand(100, 85)
    print(M)
    orth = Otrhogonalization(M)
    Q1, R1 = orth.QR()
    Q2, R2 = orth.householder()
    Q3, R3 = orth.givens()

# testOrthogonalizations()
# testQR()

import scipy.misc as sm
openpath = 'F:\Computer\Desktop\kvinnagr.jpg'
savepath = 'F:\Computer\Desktop\kvinnany.jpg'
data = np.array(sm.imread(openpath, True))

U, s, V = scl.svd(data)
Sigma = scl.diagsvd(s, len(U), len(V))
data2 = U.dot(Sigma).dot(V)

# print(scl.norm(data - data2, ord=2))

tolerance = 3000
s_neg = [si if si > tolerance else 0 for si in s]
Sigma_neg = scl.diagsvd(s_neg, len(U), len(V))

# print(s)
# print(s_neg)
# print(len(s))

data3 = U.dot(Sigma_neg).dot(V)

# sm.imsave(savepath, data3)
















