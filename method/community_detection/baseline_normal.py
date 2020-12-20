from karateclub.community_detection.overlapping import BigClam
from community import community_louvain
import networkx as nx
from networkx.algorithms import community
import numpy as np
from EM_Poisson.main import display_result, Modularity_Q
from karateclub.community_detection.overlapping import NNSED, DANMF, MNMF, BigClam, SymmNMF
from karateclub.community_detection.non_overlapping import GEMSEC, EdMot


def louvain(g, labels, isUnknow):
    print('===========    louvain   ===============')
    res = community_louvain.best_partition(g)
    res = np.array(list(res.values())) + 1

    if isUnknow:
        Q = Modularity_Q(res, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(res, labels)


def girvan_newman(g, labels, isUnknow):
    print('===========    girvan_newman   ===============')
    communities_generator = community.girvan_newman(g)
    pre_label = dict()
    for com in communities_generator:
        if len(com) >= max(labels):
            for i, nodes in enumerate(com):
                for node in nodes:
                    pre_label.update({node: i})
            pre_label = sorted(pre_label.items(), key=lambda x: x[0])
            pre_label = [i[1] for i in pre_label]
            pre_label = np.array(pre_label) + 1

            if isUnknow:
                Q = Modularity_Q(pre_label, g)
                print("Modularity_Q: {:.4f}".format(Q))
            else:
                display_result(pre_label, labels)
            break


def greedy_modularity_communities(g, labels, isUnknow):
    print('===========    greedy_modularity   ===============')
    com_res = list(community.greedy_modularity_communities(g))
    pre_label = dict()
    for i, com in enumerate(com_res):
        for node in list(com):
            pre_label.update({node: i})
    pre_label = sorted(pre_label.items(), key=lambda x: x[0])
    pre_label = [i[1] for i in pre_label]
    pre_label = np.array(pre_label) + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)


def train_byMNMF(g, labels, isUnknow):
    print('===========  Method:  MNMF  ===============')
    num_c = max(labels)
    model = MNMF(dimensions=64, clusters=num_c, lambd=0.2, alpha=0.05,
                 beta=0.05, iterations=200, lower_control=10 ** -15, eta=5.0, seed=42)
    model.fit(g)
    F_ = model.get_memberships()
    F_ = np.array(list(F_.values()))
    pre_label = F_ + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)


def train_byGEMSEC(g, labels, isUnknow):
    print('===========  Method:  GEMSEC  ===============')
    num_c = max(labels)
    model = GEMSEC(walk_number=5, walk_length=10, dimensions=32, negative_samples=10,
                   window_size=4, learning_rate=0.1, clusters=num_c, gamma=0.1, seed=42)
    model.fit(g)
    F_ = model.get_memberships()
    F_ = np.array(list(F_.values()))
    pre_label = F_ + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)


def train_byEdMot(g, labels, isUnknow):
    print('===========  Method:  EdMot  ===============')
    num_c = max(labels)
    model = EdMot()
    model.fit(g)
    F_ = model.get_memberships()
    F_ = np.array(list(F_.values()))
    pre_label = F_ + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)


def train_byBigClam(g, labels, isUnknow):
    print('===========  Method:  BigClam  ===============')
    num_c = max(labels)
    num_nodes = len(g.nodes)
    model = BigClam(num_of_nodes=num_nodes, dimensions=4, iterations=200, learning_rate=0.002, seed=42)
    model.fit(g)
    F_ = model.get_memberships()
    F_ = np.array(list(F_.values()))
    pre_label = F_ + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)


def train_byDANMF(g, labels, isUnknow):
    print('===========  Method:  DANMF  ===============')
    num_c = max(labels)
    num_nodes = len(g.nodes)
    model = DANMF(layers=[16, 4], pre_iterations=320,
                  iterations=320, seed=42, lamb=0.01)
    model.fit(g)
    F_ = model.get_memberships()
    F_ = np.array(list(F_.values()))
    pre_label = F_ + 1

    if isUnknow:
        Q = Modularity_Q(pre_label, g)
        print("Modularity_Q: {:.4f}".format(Q))
    else:
        display_result(pre_label, labels)
