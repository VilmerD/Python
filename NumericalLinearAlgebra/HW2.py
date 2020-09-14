import numpy.linalg as linalg
import numpy as np
from NumericalLinearAlgebra.Orthogonalization import Orthogonalization


def problem2(A):
    OR = Orthogonalization(A)
    OR.grahm_Schmidt()
    Q = OR.Q
    R = OR.R
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q)) - 1))


def problem3(A):
    Q, R = linalg.qr(A)
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q))- 1))


def problem5(A):
    OR = Orthogonalization(A)
    OR.householder()
    Q = OR.Q
    R = OR.R
    print("Norm of Q: {}".format(linalg.norm(Q)))
    print("How unorthogonal they are: {}".format(linalg.norm(np.identity(n) - Q.T.dot(Q))))
    print("det: 1 + {}".format(np.linalg.det(Q.T.dot(Q))- 1))


def problem6(A):
    print(A)
    OR = Orthogonalization(A)
    R, Q = OR.householder()
    print("R: {}".format(R))
    print("Q: {}".format(Q))
    u = np.array([3, 4, 9])
    print(A.dot(u))
    print("___")
    print(OR.QT_dot(R.dot(u)))

m, n = (1002, 1000)
A = np.random.rand(m, n)
problem5(A)