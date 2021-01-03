import numpy as np
import scipy.io as io
import sys, os
import pandas as pd
import networkx as nx
from collections import defaultdict
from networkx.generators import ego_graph
from networkx.algorithms.community import greedy_modularity_communities
from community import community_louvain
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.abspath('..'))
pre_path_com_known = sys.path[-1] + '/dataset/complete_network/known_/'
pre_path_com_unknown = sys.path[-1] + '/dataset/complete_network/unknown_/'
pre_path_time = sys.path[-1] + '/dataset/time_network/'


def read_karate_club():
    print("Dataset : karate ---------------------------------------------")
    g_ = nx.karate_club_graph()
    B = nx.to_numpy_array(g_, nodelist=sorted(g_.nodes))
    g = nx.from_numpy_array(B)
    data = io.loadmat(pre_path_com_known + 'karate_rlabels.mat')
    labels = np.array(data['labels'])[0]
    is_unknown = False

    res = dict()
    res['g_'] = g_
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 30
    return res


def read_dolphins():
    # k : 2
    print("Dataset : dolphins ---------------------------------------------")
    g, B, labels = _read_txt_graph('dolphins')
    is_unknown = False

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 100
    return res


def read_polbooks():
    print("Dataset : polbooks ---------------------------------------------")
    g, B, labels = _read_txt_graph('polbooks')
    is_unknown = False

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 10
    return res


def read_polblogs():
    print("Dataset : polblogs ------------------------------------------")
    g, B, labels = _read_txt_graph('polblogs')
    is_unknown = False

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 100
    return res


def read_football():
    print("Dataset : football ---------------------------------------------")
    g, B, labels = _read_txt_graph('football')
    is_unknown = False

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 100
    return res


def read_wisconsin():
    # K : 5
    print("Dataset : wisconsin ---------------------------------------------")
    data = io.loadmat(pre_path_com_known + 'wisconsin.mat')
    B = np.array(data['B'], dtype=float)
    g = nx.from_numpy_array(B)
    content = np.array(data['F'])
    target = np.array(data['labels'])
    target = target.reshape((target.size,))

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = target
    res['is_unknown'] = False
    res['num_del_edges'] = 100
    return res


def _read_txt_graph(name):
    with open(pre_path_com_known + name + '.txt', 'r') as file:
        lines = file.readlines()
        new_lines = []
        for line in lines:
            line = line.split(' ')
            new_line = [int(line[0]), int(line[1].split('\n')[0])]
            new_lines.append(new_line)
        # print(new_lines)
    res = np.array(new_lines)
    pd_res = pd.DataFrame(res)
    g = nx.from_pandas_edgelist(pd_res, 1, 0)
    g.add_nodes_from([i for i in range(1, max(g.nodes) + 1)])  # 因为这里的点有的是孤立的，没在edge列表里，所以需要将缺失的孤立点补全
    B = nx.to_numpy_array(g, nodelist=sorted(g.nodes))
    g = nx.from_numpy_array(B)
    print(f"g.size: {len(g.nodes)}")
    data = io.loadmat(pre_path_com_known + name + '_rlabels.mat')
    labels = np.array(data['labels'])
    return g, B, labels[0]


def read_adjnoun():
    print('----------- unknown com data : adjnoun ------------------')
    return _read_unknown('adjnoun')


def read_celegansneural():
    print('----------- unknown com data : celegansneural ------------------')
    return _read_unknown('celegansneural')


def read_email():
    print('----------- unknown com data : email ------------------')
    return _read_unknown('email')


def read_jazz():
    print('----------- unknown com data : jazz ------------------')
    return _read_unknown('jazz')


def read_lesmis():
    print('----------- unknown com data : lesmis ------------------')
    return _read_unknown('lesmis')


def _read_unknown(name):
    # 需要对gml文件里内容的开头，标注上是有向图还是无向图
    g = nx.read_gml(pre_path_com_unknown + name + '.gml', label='id')
    B = nx.to_numpy_array(g, nodelist=sorted(g.nodes))
    B[B > 1] = 1
    g = nx.from_numpy_array(B)
    partition = community_louvain.best_partition(g)
    label = np.array(list(partition.values()), dtype=int) + 1

    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = label
    res['is_unknown'] = True
    res['num_del_edges'] = 100
    return res


def read_dblp_time1():
    G = nx.Graph()
    paper_label = []
    name_labels = defaultdict(list)

    with open(pre_path_time + 'dblp/label1.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            paper_label.append(int(line.strip()))

    with open(pre_path_time + 'dblp/name1.txt', 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            author = line.strip()
            name_labels[author].append(paper_label[j])
            G.add_node(author, label=paper_label[j])

    with open(pre_path_time + 'dblp/edge1.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            G.add_edge(line[0].strip(), line[1].strip())

    G = nx.to_undirected(G)  # node name
    B = nx.to_numpy_array(G)  # network adj
    G1 = nx.from_numpy_array(B)  # node index

    # take the first label as author label
    author_label = []
    for name, labels in name_labels.items():
        author_label.append(labels[0])

    res = dict()
    res['graph_real'] = G1
    res['adj_real'] = B
    res['labels'] = author_label
    res['is_unknown'] = False
    res['name_labels'] = name_labels
    res['paper_label'] = paper_label
    res['graph_ori'] = G

    return res


def read_dblp_time1_subgraph():
    def ori_label_to_index_label(labels):
        # convert 1, 1, 3, 4 ---> 1, 1, 2, 3
        tmp_labels = set(labels)
        ori2index = dict()
        index = 1
        for i in tmp_labels:
            ori2index[str(i)] = index
            index += 1
        return ori2index

    res = read_dblp_time1()
    g_ = res['graph_ori']
    graph = ego_graph(g_, 'Thomas S. Huang', 5)
    labels = []
    for name, attr in graph.nodes.data():
        labels.append(attr['label'])

    label_dict = ori_label_to_index_label(labels)
    sorted_label = []
    for i in labels:
        sorted_label.append(label_dict[str(i)])
    B = nx.to_numpy_array(graph)  # network adj
    G1 = nx.from_numpy_array(B)  # node index
    res_ = dict()
    res_['graph_real'] = G1
    res_['adj_real'] = B
    res_['labels'] = sorted_label
    res_['ori_labels'] = labels
    res_['is_unknown'] = False
    return res_


def main():
    import matplotlib.pyplot as plt

    data = read_dblp_time1_subgraph()
    g = data['graph_real']
    labels = data['labels']
    ori_labels = data['ori_labels']
    print(ori_labels)
    print(labels)
    print(g.nodes.data())
    plt.subplots(1, 1, figsize=(15, 13))
    nx.draw_spring(g, with_labels=True, node_color=labels)
    plt.show()


if __name__ == '__main__':
    # res = read_dblp_time1()
    # g = res['graph_real']
    # print(g)
    main()
