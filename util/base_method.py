from community import community_louvain
import numpy as np
import networkx as nx
import pandas as pd


def Modularity_Q(label, g):
    partition = {i: index for i, index in enumerate(label)}
    Q = community_louvain.modularity(partition, g)
    return Q


def mean_variance(path):
    df = pd.read_csv(path, usecols=[1, 2, 3])

    print(df.describe())


def display_result(F_argmax, target):
    from sklearn.metrics.cluster import adjusted_mutual_info_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score

    def purity(cluster, label):
        cluster = np.array(cluster)
        label = np.array(label)
        indedata1 = {}
        for p in np.unique(label):
            indedata1[p] = np.argwhere(label == p)
        indedata2 = {}
        for q in np.unique(cluster):
            indedata2[q] = np.argwhere(cluster == q)
        count_all = []
        for i in indedata1.values():
            count = []
            for j in indedata2.values():
                a = np.intersect1d(i, j).shape[0]
                count.append(a)
            count_all.append(count)

        return sum(np.max(count_all, axis=0)) / len(cluster)

    nmi = normalized_mutual_info_score(target, F_argmax)
    ari = adjusted_rand_score(target, F_argmax)
    pur = purity(F_argmax, target)
    # f1_macro = f1_score(target, F_argmax, average='macro')
    # f1_micro = f1_score(target, F_argmax, average='micro')
    # print('NMI: {:.4f}, f1_macro: {:.4f},   f1_micro: {:.4f} '
    #       .format(nmi, f1_macro, f1_micro))

    print(f'NMI: {nmi:.4f}')
    print(f'ARI: {ari:.4f}')
    print(f'Pur: {pur:.4f}')
    return nmi, ari, pur, F_argmax  # , f1_macro, f1_micro


def get_train_test(graph: nx.Graph, fraction: float = 0.2, del_list: list = None, isShuffle: bool = True) -> (
np.ndarray, list):
    # 提前 copy，为了让 obs 里面保持着原图中所有结点，只是边缺失而已
    observe_graph = graph.copy()

    if del_list is None:
        all_edges = list(graph.edges)
        num_total_edges = graph.number_of_edges()
        del_list = []

        if fraction == 0:
            return graph, []
        num_del = int(num_total_edges * fraction)
        # 随机删除原图中的边，按照比例
        # del_edges_index = np.random.choice([i for i in range(len(all_edges))], size=num_del, replace=False)
        if isShuffle:
            np.random.shuffle(all_edges)
            del_list = all_edges[:num_del + 1]
            observe_graph.remove_edges_from(del_list)
        else:
            count = 0
            i = 0
            while count < num_del:
                i += 1
                np.random.seed(i)
                index = np.random.random_integers(0, num_total_edges - 1)
                edge = all_edges[index]
                if edge in del_list:
                    continue
                observe_graph.remove_edge(edge[0], edge[1])
                if nx.is_connected(observe_graph):
                    del_list.append(edge)
                    count += 1
                else:
                    observe_graph.add_edge(edge[0], edge[1])
    else:
        observe_graph.remove_edges_from(del_list)  # 此时虽然删除了些边，但是即使结点孤立，也保存在图中

    test_edge_list = del_list
    # print("deleted edges: ", sorted(test_edge_list))
    return observe_graph, test_edge_list


def get_Z_init(graph) -> tuple:
    """
    在执行 E 步之前进行，删除了一些边之后，获取所有缺失边
    包含两部分：无边，预测的边
    :param N:
    :param graph:  如 -20% 边的 graph
    :return: 缺失边列表
    """
    # 任意结点所有可能边， i < j
    nodes = sorted(graph.nodes)
    start_index, end_index = nodes[0], nodes[-1]
    all_Edges = [(i, j) for i in range(start_index, end_index + 1) for j in range(i + 1, end_index + 1)]
    g_Z_noEdge = nx.Graph()
    g_Z_noEdge.add_edges_from(set(all_Edges) - set(graph.edges))  # 去掉已经存在的边，剩下是缺失边
    return nx.Graph(), g_Z_noEdge


if __name__ == '__main__':
    mean_variance('../res/karate_byGEMSEC.csv')
