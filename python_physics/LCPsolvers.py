import numpy as np
from abc import ABC, abstractproperty
# from functools import cached_property

def pivoting_methods(A, b, method="default"):
    solver = IncrementalPivoting(A, b) # Default
    if method == "principal":
        solver =  PrincipalPivoting(A, b)
    solver.solve()
    return solver.x


def get_block(A, II, JJ):
    li, lj = len(II), len(JJ)
    B = np.empty([li, lj])

    for ii in range(li):
        for jj in range(lj):
            B[ii, jj] = A[II[ii], JJ[jj]]

    return B


def compute_alpha(vj, delta_vj, F, x, delta_xf, L, v, delta_vl):
    xf = x[F]
    vl = v[L]

    if abs(delta_vj) < 1e-5:
        delta_vj = 0.0
    aj = -vj/delta_vj
    af = np.inf
    al = np.inf
    ind_f = -1
    ind_l = -1
    alphas = [aj, af, al]

    for xi, delta_xi, f in zip(xf, delta_xf, F):
        if delta_xi < 0:
            alphas[1] = np.min([alphas[1], (-xi/delta_xi).squeeze()])
            ind_f = f

    for vi, delta_vi, l in zip(vl, delta_vl, L):
        if delta_vi < 0:
            alphas[2] = np.min([alphas[2], (-vi/delta_vi).squeeze()])
            ind_l = l

    a_out = np.min(alphas)
    indices = [-1, ind_f, ind_l]
    ind_block = indices[alphas.index(a_out)]

    return a_out, ind_block


def find_j(v: np.ndarray, P: list):
    if not P or np.min(v[P]) >= 0:
        return -1
    return P[int(np.where(v[P]==np.min(v[P]))[0][0])]

class Splitting(ABC):
    
    def __init__(self, A, b, num_iters=10):
        self.A = A
        self.b = b.reshape(-1,1)
        self.num_iters = num_iters
        self.x = np.zeros_like(self.b)
        self.solve()
        self.M

    @abstractproperty
    def M(self):
        pass
    
    @property
    def N(self):
        return self.M - self.A    

    def solve(self):
        o = np.zeros_like(self.x)
        for _ in range(self.num_iters):
            z = np.linalg.inv(self.M)@(self.N@self.x+self.b)
            self.x = np.maximum(z, o)

class Jacobi(Splitting):

    @property
    def M(self):
        return np.diag(np.diag(self.A))

class PGS(Splitting):

    @property
    def M(self):
        return np.tril(self.A, k=0)


class IncrementalPivoting(object):
    def __init__(self, A, b):
        self.A = np.copy(A)
        self.b = np.copy(b)

        self.F = []
        self.L = []
        self.x = np.zeros_like(self.b)
        self.v = np.copy(b)

    def solve(self):
        print("Incremantal Solver")
        condition1 = True

        while condition1:
            P = [a for a in range(len(self.A)) if a not in self.F and a not in self.L]
            j = find_j(self.v, P)
            if j < 0:
                break

            while True:
                Aff = get_block(self.A, self.F, self.F)
                Afj = get_block(self.A, self.F, [j])
                Afl_T = get_block(self.A, self.F, self.L).T
                Alj = get_block(self.A, self.L, [j])
                Afj_T = get_block(self.A, self.F, [j]).T
                Ajj = self.A[j, j]

                delta_xf = np.linalg.solve(Aff, -Afj)
                # if delta_xf.shape[0] > 0:
                delta_vl = Afl_T @ delta_xf + Alj
                delta_vj = (Afj_T @ delta_xf + Ajj).squeeze()
                # else:
                #     delta_vl = Alj
                #     delta_vj = Ajj

                alpha, ll = compute_alpha(self.v[j],
                                          delta_vj,
                                          self.F,
                                          self.x,
                                          delta_xf,
                                          self.L,
                                          self.v,
                                          delta_vl)

                if delta_xf.shape[0] > 0:
                    self.x[self.F] += alpha * delta_xf.squeeze()
                if delta_vl.shape[0] > 0:
                    self.v[self.L] += alpha * delta_vl.squeeze()
                self.x[j] = alpha

                if ll < 0:
                    ll = j
                if ll in self.F:
                    self.F.pop(self.F.index(ll))
                    self.L.append(ll)
                elif ll in self.L:
                    self.L.pop(self.L.index(ll))
                    self.F.append(ll)
                else:
                    self.F.append(ll)

                if ll == j:
                    break

class PrincipalPivoting(object):
    def __init__(self, A, b):
        self.A = np.copy(A)
        self.b = np.copy(b)
        self.F = list(np.arange(np.shape(A)[0]))

        self.L = []
        self.U = []
        self.x = np.zeros_like(self.b)
        self.v = np.copy(b)
        self.lh = np.inf
        self.ll = 0
        self.N = 10

    def solve(self):
        print("Principal Solver")
        for _ in range(3):
            print(self.A)
            Aff = get_block(self.A, self.F, self.F)
            Afl = get_block(self.A, self.F, self.L)
            AfU = get_block(self.A, self.F, self.U)
            bf  = self.b[self.F]
            xl  = self.x[self.L]
            xu  = self.x[self.U]

            self.x[self.F] = np.linalg.solve(Aff, -bf - Afl @ xl - AfU @ xu)
            self.v = self.A @ self.x + self.b

            for i in range(len(self.x)):
                if i in self.F:
                    if self.x[i] < self.ll:
                        self.F.pop(self.F.index(i))
                        self.L.append(i)
                    elif self.x[i] > self.lh:
                        self.F.pop(self.F.index(i))
                        self.U.append(i)
                elif i in self.U and self.v[i] >= 0:
                    self.U.pop(self.U.index(i))
                    self.F.append(i)
                elif i in self.L and self.v[i] <= 0:
                    self.L.pop(self.L.index(i))
                    self.F.append(i)


        # if delta_xf.shape[0] > 0:

