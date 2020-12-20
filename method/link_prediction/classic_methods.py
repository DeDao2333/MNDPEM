import networkx as nx
from networkx.algorithms import link_prediction


def resource_allocation_index(g, edges, alpha=0.5):
    preds = link_prediction.resource_allocation_index(g, edges)
    infer = []
    for u, v, p in preds:
        if p > alpha:
            infer.append((u, v))
    return infer


def jaccard_coefficient(g, edges, alpha=0.5):
    preds = link_prediction.jaccard_coefficient(g, edges)
    infer = []
    for u, v, p in preds:
        if p > alpha:
            infer.append((u, v))
    return infer


def adamic_adar_index(g, edges, alpha=1.5):
    preds = link_prediction.adamic_adar_index(g, edges)
    infer = []
    for u, v, p in preds:
        if p > alpha:
            infer.append((u, v))
    return infer


def preferential_attachment(g, edges, alpha=10):
    preds = link_prediction.adamic_adar_index(g, edges)
    infer = []
    for u, v, p in preds:
        if p > alpha:
            infer.append((u, v))
    return infer
