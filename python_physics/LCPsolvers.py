import numpy as np


def incremental_pivoting(A, b):
    ip = IncrementalPivoting(A, b)
    ip.solve()
    return ip.x


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

    aj = -vj/delta_vj
    af = np.inf
    al = np.inf
    ind_f = -1
    ind_l = -1
    alphas = [aj, af, al]

    for xi, delta_xi, f in zip(xf, delta_xf, F):
        if delta_xi < 0:
            alphas[1] = np.min([alphas[1], -xi/delta_xi])
            ind_f = f

    for vi, delta_vi, l in zip(vl, delta_vl, L):
        if delta_vi < 0:
            alphas[2] = np.min([alphas[2], -vi/delta_vi])
            ind_l = l

    a_out = np.min(alphas)
    indices = [-1, ind_f, ind_l]
    ind_block = indices[alphas.index(a_out)]

    return a_out.squeeze(), ind_block


def find_j(v, P):
    for j in P:
        if v[j] < 0:
            return j

    return -1


class IncrementalPivoting(object):
    def __init__(self, A, b):
        self.A = np.copy(A)
        self.b = np.copy(b)

        self.F = []
        self.L = []
        self.x = np.zeros_like(self.b)
        self.v = np.copy(b)

    def solve(self):
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
                if delta_xf.shape[0] > 0:
                    delta_vl = Afl_T @ delta_xf + Alj
                    delta_vj = Afj_T @ delta_xf + Ajj
                else:
                    delta_vl = Alj
                    delta_vj = Ajj

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
                lambda_j = alpha

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
