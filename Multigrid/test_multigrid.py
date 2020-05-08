import unittest
import numpy as np
from numpy import linalg as lin
import scipy.sparse.linalg as splin
import scipy.sparse as sp
from Multigrid.multigrid import *


class TestMultigrid(unittest.TestCase):
    def test_eye(self):
        level = 7
        n = 2 ** level - 1

        def eye(level):
            n = level ** 2 - 1
            return sp.eye(n)
        v = np.zeros(63, )
        f = np.sin(np.pi*np.arrange(1, n + 1) / (n + 1))
        u = v_cycle(eye, v, f)

        self.assertEqual(0, lin.norm(f - eye(level)))
        self.assertEqual(0, lin.norm(u - f))