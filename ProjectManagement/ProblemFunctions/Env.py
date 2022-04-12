import numpy as np
import copy


class Project:
    def __init__(self, m, inst_num=0):
        self.m = m
        self.N = 3*self.m + 1
        self.xi_dim = self.m
        self.y_dim = self.N
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.xi_dim)

        # make A
        A = np.zeros([self.N, self.N])
        for l in np.arange(m):
            for p in [2, 3]:
                A[3*l + 1, 3*l + p] = 1
                A[3*l + p, 3*l + 4] = 1

    def duration(self, i, j, xi):
        return None

