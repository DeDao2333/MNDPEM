import numpy as np
import scipy.io as io
import sys, os
import pandas as pd
import networkx as nx
from collections import defaultdict
from networkx.generators import ego_graph
from networkx.algorithms.community import greedy_modularity_communities
from community import community_louvain
import graph_tool.all as gt
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


def read_pubmed():
    print("Dataset : pubmed ---------------------------------------------")
    with open(pre_path_com_known + 'pubmed' + '.txt', 'r') as file:
        lines = file.readlines()
        new_lines = []
        for line in lines:
            line = line.split('\t')
            new_line = [int(line[0]), int(line[1].split('\n')[0])]
            new_lines.append(new_line)
        # print(new_lines)
    res_ = np.array(new_lines)
    pd_res = pd.DataFrame(res_)
    g = nx.from_pandas_edgelist(pd_res, 1, 0)
    g.add_nodes_from([i for i in range(1, max(g.nodes) + 1)])  # 因为这里的点有的是孤立的，没在edge列表里，所以需要将缺失的孤立点补全
    B = nx.to_numpy_array(g, nodelist=sorted(g.nodes))
    g = nx.from_numpy_array(B)

    with open(pre_path_com_known + 'pubmed_labels.txt', 'r') as f:
        line = f.readline()
        line = line[1:-1].split(',')
        labels = list(map(int, line))

    is_unknown = False
    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = labels
    res['is_unknown'] = is_unknown
    res['num_del_edges'] = 100
    return res

def read_pubmed_sub():
    import re
    with open(pre_path_com_known + 'pubmed_sub.txt', 'r') as f:
        lines = f.readlines()
        edges_list = []
        for line in lines:
            e = line.strip().split(' ')
            edges_list.append((int(e[0]), int(e[1])))
    g = nx.Graph()
    g.add_edges_from(edges_list)
    B = nx.to_numpy_array(g, nodelist=sorted(g.nodes))
    g = nx.from_numpy_array(B)

    with open(pre_path_com_known + 'pubmed_sub_labels.txt', 'r') as f:
        res = re.findall(r"\d+", f.readline())
        vals = list(map(int, res))

    is_unknown = False
    res = dict()
    res['graph_real'] = g
    res['adj_real'] = B
    res['labels'] = vals
    res['is_unknown'] = is_unknown
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

    data = read_pubmed()
    g = data['graph_real']
    labels = data['labels']
    with open('pubmed_labels.txt', 'w') as f:
        for i, j in enumerate(labels):
            f.write(str(i + 1) + '\t' + str(j) + '\n')


def main2():
    from model.Strategy import Strategy
    data = Strategy.prepare_data(read_pubmed)
    graph_real = data['graph_real']
    print(f'is connected : {nx.is_connected(graph_real)}')
    print(f'node num: {len(graph_real.nodes)}, edges num: {len(graph_real.edges)}')
    # main()
    obser_g = data['observe_graph']
    largest_ = max(nx.connected_components(obser_g), key=len)
    miss_g_sub = nx.subgraph(obser_g, largest_)
    print(f'is connected : {nx.is_connected(miss_g_sub)}')
    print(f'node num: {len(obser_g.nodes)}, edges num: {len(obser_g.edges)}')
    print(f'node num: {len(miss_g_sub.nodes)}, edges num: {len(miss_g_sub.edges)}')


def main3():
    import graph_tool.all as gt
    import numpy as np

    g = gt.collection.data["dolphins"].copy()
    N = g.num_vertices()
    E = g.num_edges()
    q = g.new_ep("double", 0.8)

    q_default = (E - q.a.sum()) / ((N * (N - 1)) / 2 - E)
    state = gt.UncertainBlockState(g, q=q, q_default=q_default, nested=True)
    gt.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=1))

    u = None
    bs = []
    cs = []

    def collect_marginals(s):
        global bs, u, cs
        u = s.collect_marginal(u)
        bstate = s.get_block_state()
        bs.append(bstate.levels[0].b.a.copy())
        cs.append(gt.local_clustering(s.get_graph()).fa.mean())

    gt.mcmc_equilibrate(state, force_niter=2000, mcmc_args=dict(niter=7), callback=collect_marginals)
    eprob = u.ep.eprob

    print(state.get_block_state().get_bs()[0])
    data = read_dolphins()
    labels = data['labels']


def main_pubmed():
    res = read_pubmed()
    g_real = res['graph_real']
    # g_sub = nx.subgraph(g_real, [i for i in range(500)])
    g_sub = nx.ego_graph(g_real, 0, 4)
    print(len(g_sub.edges))
    print(len(g_sub.nodes))

    with open('pubmed_sub.txt', 'w') as f:
        for i, j in g_sub.edges:
            f.write(str(i))
            f.write(' ')
            f.write(str(j))
            f.write('\n')

    with open('pubmed_sub_labels.txt', 'w') as f:
        labels = res['labels']
        sub_labels = []
        for i in g_sub.nodes:
            sub_labels.append(labels[i])
        f.write(str(sub_labels))


if '__main__' == __name__:
    # res = read_dolphins()
    # print(res['labels'])
    # main_pubmed()
    # read_pubmed_sub()
    res = read_pubmed_sub()
    print(len(res['graph_real'].nodes))