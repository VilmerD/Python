import numpy as np
from scipy import linalg as splin
from NumericalLinearAlgebra.Orthogonalization import Orthogonalization
from matplotlib import pyplot as plt
import sympy as sy


def problem1():
    data = []
    with open('signal.dat', 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([float(i) for i in k])
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]

    A = np.stack((np.sin(x), np.cos(x), np.sin(2 * x), np.cos(2 * x)), axis=-1)
    m, n = A.shape
    OR = Orthogonalization(A)
    R, Q = OR.householder()
    R_hat = R[:n, :]
    a = splin.solve_triangular(R_hat, Q.dot(y.copy())[:n])

    f = A.dot(a)
    print(np.linalg.norm(f - y))
    fig, ax = plt.subplots()
    plt.plot(x, y, 'r')
    plt.plot(x, f, 'g-')
    plt.show()


def problem4():
    n = 50
    H = splin.hilbert(n)
    Hi = splin.invhilbert(n)
    U, S, V = splin.svd(H)
    b = U[:, 0] * S[0]
    db = U[:, -1] * S[-1]
    print(S[-1])
    x = Hi.dot(b)
    dx = Hi.dot(db)
    k = S[0]/S[-1]
    print('Condition number of H50: {}'.format(k))
    dxx = np.linalg.norm(dx) / np.linalg.norm(x)
    dbb = np.linalg.norm(db) / np.linalg.norm(b)
    q = dxx/dbb
    print('Quotient of pertubations: {}'.format(q))
    print('Quotient: {}'.format(q / k))

    maxq = 0
    for j in range(0, 100):
        num_vecs = 1000
        b_collection = np.random.rand(n, num_vecs)
        db_collection = np.random.rand(n, num_vecs) * 1e-16
        x_collection = Hi.dot(b_collection)
        dx_collection = Hi.dot(db_collection)
        q_collection = np.zeros((num_vecs, 1))
        for k in range(0, num_vecs):
            q_collection[k] = np.linalg.norm(dx_collection[:, k]) * np.linalg.norm(b_collection[:, k])
            q_collection[k] /= np.linalg.norm(db_collection[:, k]) * np.linalg.norm(x_collection[:, k])
        maxq = max(maxq, max(q_collection))
        print(j)
    print(max(maxq))

problem4()