import numpy as np
import networkx as nx


def MNDP(G: np.ndarray, c: int, F: np.ndarray = None, T: int = 1, L: int = 1000):
    # T = 1
    # L = 1000
    epcl = 1.0e-3
    mi = 1.0e20
    n = len(G[0])
    sm = np.sum(G)
    for t in range(T):
        # 这里如果T>1的话，第二次循环 X=F，而不是随机初始化了
        if F is None:
            X = np.random.uniform(0, 1, size=(n, c))
        else:
            X = F
        gg = X.dot(X.T)
        tmp = sm / np.sum(gg)
        tmp = tmp ** 0.5
        X *= tmp
        gg = X.dot(X.T)
        sav = np.sum(abs(G - gg) ** 2) ** 0.5
        for i in range(L):
            X *= (0.5 + G.dot(X) / (2 * (X.dot(X.T)).dot(X)))
            gg = X.dot(X.T)
            cur = np.sum(abs(G - gg) ** 2) ** 0.5
            if abs(cur - sav) < epcl:
                sav = cur
                break
            sav = cur
        if sav < mi:
            mi = sav
            F = X
    Prob = F.dot(F.T)
    return Prob, F


def MNDP_EM(g: nx.Graph, c: int, F: np.ndarray = None, w: np.ndarray = None, q: np.ndarray = None, epoch: float = 1):
    # thr = 1.0e-4
    import math
    thr = 200 / math.exp(epoch + 1)
    L = 20000
    T = 1
    M = list(g.edges)
    n_e = get_n_e_Index(g)
    num_edge = g.number_of_edges()
    num_node = g.number_of_nodes()
    mx = -1.0e20
    sseta = 0
    sw = 0
    LH = 0

    for tt in range(T):
        if F is None:
            F = np.random.uniform(0, 1, size=(num_node, c))
            for z in range(c):
                sum_ = np.sum(F[:, z])
                F[:, z] /= sum_

        if w is None:
            w = np.random.uniform(0, 1, size=(c,))
            w = w / np.sum(w) * num_edge * 2
        if q is None:
            q = np.zeros(shape=(num_edge, c))
        sav = np.random.random()

        for ll in range(L):
            # E step
            for x in range(num_edge):
                e = M[x]
                i = e[0]
                j = e[1]
                vec = F[i] * F[j] * w
                q[x] = vec / (np.sum(vec) + 1e-8)
            # E end
            # M step
            for z in range(c):
                sumz = np.sum(q[:, z]) * 2

                for i in range(num_node):
                    di = n_e[i]
                    fz = np.sum(q[di, z])
                    F[i, z] = fz / sumz

                w[z] = sumz
            # M end
            LH = computeLH(M, F, w)

            if abs(LH - sav) < thr:
                break
            sav = LH
        print("Likelihood:  ", LH)
        if LH > mx:
            mx = LH
            sseta = F
            sw = w

    return F


def get_n_e_Index(g: nx.Graph):
    edges = list(g.edges)
    n_e = {i: [] for i in range(g.number_of_nodes())}
    for i, e in enumerate(edges):
        u, v = e[0], e[1]
        n_e[u].append(i)
        n_e[v].append(i)
    return n_e


def computeLH(M, F, w):
    from math import log
    LH = 0
    m = len(M)
    for i in range(m):
        i, j = M[i]
        LH = LH + log(np.sum(F[i] * F[j] * w) + 1e-8, 2) * 2

    return LH - np.sum(F.dot(F.T))



