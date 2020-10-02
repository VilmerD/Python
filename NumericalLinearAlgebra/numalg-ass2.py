
# coding: utf-8

# In[ ]:

import numpy as np
from numpy import linalg as linalg

#Gram-Schmidt with matrix A as input, returning Q with columns as orthogonal basis.
def gramschmidt(A):
    Q = np.zeros(A.shape)
    #Begin with the orthogonlization process through projection
    for i in range(A.shape[1]):
        a = A[:, i]
        q = Q[:, :i]
        proj = np.dot(a, q)
        a = a - np.sum(proj * q, axis=1)
        # Continue by normalizing the resulting vectors
        norm = np.sqrt(np.dot(a, a))
        a /= abs(norm) < 1e-8 and 1 or norm
        Q[:, i] = a
    return Q
        

#Make the Householder transformation
def householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H    

#Perform the QR using the transformation
def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
    

