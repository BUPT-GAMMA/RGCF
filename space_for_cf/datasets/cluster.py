

import itertools
import networkx as nx
from time import time
__all__ = [
    "clustering",
    "latapy_clustering",
    "robins_alexander_clustering",
]


def cc_dot(nu, nv):
    return float(len(nu & nv)) / len(nu | nv)


def cc_max(nu, nv):
    return float(len(nu & nv)) / max(len(nu), len(nv))


def cc_min(nu, nv):
    return float(len(nu & nv)) / min(len(nu), len(nv))


def latapy_clustering(G):
    nodes = G
    ccs_dot = {}
    ccs_max = {}
    ccs_min = {}
    for v in nodes:
        ccdot = 0.0
        ccmax = 0.0
        ccmin = 0.0
        nbrs2 = {u for nbr in G[v] for u in G[nbr]} - {v}
        for u in nbrs2:
            ccdot += cc_dot(set(G[u]), set(G[v]))
            ccmax += cc_max(set(G[u]), set(G[v]))
            ccmin += cc_min(set(G[u]), set(G[v]))
        if ccdot > 0.0:  # len(nbrs2)>0
            ccdot /= len(nbrs2)
        if ccmax > 0.0:
            ccmax /= len(nbrs2)
        if ccmin > 0.0:
            ccmin /= len(nbrs2)
        ccs_dot[v] = ccdot
        ccs_max[v] = ccmax
        ccs_min[v] = ccmin
    return ccs_dot, ccs_max, ccs_min


clustering = latapy_clustering



def robins_alexander_clustering(G):
    if G.order() < 4 or G.size() < 3:
        return 0
    L_3 = _threepaths(G)
    if L_3 == 0:
        return 0
    C_4 = _four_cycles(G)
    return (4.0 * C_4) / L_3


def _four_cycles(G):
    cycles = 0
    for v in G:
        for u, w in itertools.combinations(G[v], 2):
            cycles += len((set(G[u]) & set(G[w])) - {v})
    return cycles / 4


def _threepaths(G):
    paths = 0
    count = 0
    t1 = time()
    for v in G:
        for u in G[v]:
            for w in set(G[u]) - {v}:
                paths += len(set(G[w]) - {v, u})
        count+=1
        if count%10==0:
            print(f'{time()-t1}')
    # Divide by two because we count each three path twice
    # one for each possible starting point
    return paths / 2
