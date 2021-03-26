from karateclub.community_detection.overlapping import BigClam
import os, sys

sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from util import draw_graph as Draw
from util import base_method as tools
from method.community_detection import MNDP
from collections import defaultdict
from util import read_data as Read
from model.MNDPEM_model import MNDPEM_model
from karateclub.community_detection.overlapping import NNSED, DANMF, MNMF, BigClam, SymmNMF
from karateclub.community_detection.non_overlapping import GEMSEC, EdMot
import model.conf as CONF


class Strategy(object):
    def __init__(self):
        pass

    @staticmethod
    def prepare_data(read_data, missing_rate=0.2, del_list=None):
        res: dict = read_data()
        # 删除一定比例的边数，得到观测图
        observe_graph, test_edges = tools.get_train_test(res['graph_real'], missing_rate, del_list)
        num_del_edges = len(test_edges)

        # 在删除一定比例边后，得到 Z 区域所有可能边列表
        Z_Edge_ori, Z_noEdge_ori = tools.get_Z_init(observe_graph)

        N = res['graph_real'].number_of_nodes()
        C = len(set(res['labels']))  # 有多少社团，不能用max，比如1，1，3，3

        res['observe_graph'] = observe_graph
        res['Z_noEdge_ori'] = Z_noEdge_ori
        res['Z_Edge_ori'] = Z_Edge_ori
        res['del_edges'] = test_edges
        if num_del_edges != 0:
            res['num_del_edges'] = num_del_edges

        res['N'] = N
        res['C'] = C
        return res

    @classmethod
    def paint_color(cls, g_observe_: nx.Graph, g_predicted: nx.Graph, test_edges: list, label: list) -> nx.Graph:
        """
        paint color for node label, edges.
        :param g_observe_: observe graph
        :param g_predicted: graph produced from algorithm
        :param test_edges: deleted edges from real network
        :param label: label for final graph
        :return: painted graph
        """
        g_observe = g_observe_.copy()
        predicted_edges: set = set(g_predicted.edges) - set(g_observe.edges)
        predicted_edges_true: set = predicted_edges & set(test_edges)
        predicted_edges_wrong: set = predicted_edges - set(test_edges)
        g_observe.add_edges_from(test_edges)
        g_observe.add_edges_from(predicted_edges)
        for i, j in g_observe.edges:
            g_observe.edges[i, j]['color'] = CONF.LINK_COLORs['exist']
        for i, j in test_edges:
            g_observe.edges[i, j]['color'] = CONF.LINK_COLORs['removed']
        for i, j in predicted_edges_true:
            g_observe.edges[i, j]['color'] = CONF.LINK_COLORs['predict_true']
        for i, j in predicted_edges_wrong:
            g_observe.edges[i, j]['color'] = CONF.LINK_COLORs['predict_wrong']
        print(f'nodes: {g_observe.nodes}')
        for i in g_observe.nodes:
            g_observe.nodes[i]['color'] = CONF.NODE_LABELs[label[i]]
        return g_observe

    @classmethod
    def res_display(cls, is_unknown, F_argmax, graph, labels):
        res = dict()
        res['graph_res'] = graph
        if is_unknown:
            Q = tools.Modularity_Q(F_argmax, graph)
            print("Modularity_Q: {:.4f}".format(Q))
            res['Q'] = Q
        else:
            tmp = tools.display_result(F_argmax, labels)
            res['nmi'] = tmp[0]
            res['ari'] = tmp[1]
            res['pur'] = tmp[2]
            res['F_argmax'] = tmp[3]
        return res

    @classmethod
    def res2csv(cls, res: dict, path):
        if 'nmi' in res.keys():
            # represent nmi, ari, pur
            df = pd.DataFrame(res, columns=['nmi', 'ari', 'pur'])
            df.to_csv(path, mode='a', index=False)
        elif 'Q' in res.keys():
            # represent Q
            df = pd.DataFrame(res, columns=['Q'])
            df.to_csv(path, mode='a', index=False)
        else:
            assert False, "Not right index"

    @staticmethod
    def train_byMNDPEM(data, num_EM_iter=15, alpha=0.5):
        print('===========  Method:  MNDPEM  ===============')
        C = data['C']
        N = data['N']
        ob = data['observe_graph']
        labels = data['labels']
        Z_noEdge_ori = data['Z_noEdge_ori']
        Z_Edge_ori = data['Z_Edge_ori']
        num_del_edges = data['num_del_edges']
        is_unknown = data['is_unknown']
        our_method = MNDPEM_model(N, C)
        our_method.train_mode = 2
        our_method.num_EM_iter = num_EM_iter
        our_method.alpha = alpha
        return our_method.train(ob, labels, Z_noEdge_ori, Z_Edge_ori, num_del_edges, is_unknown)

    @classmethod
    def train_byMNDP_Missing(cls, data):
        print('===========  Method:  MNDP-Missing  ===============')
        C = data['C']
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']
        F_ = MNDP.MNDP_EM(observe_graph, C, epoch=16)
        # prob, F_ = MNDP.MNDP(data['adj_real'], C)
        F_argmax = np.argmax(F_, axis=1) + 1
        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byGEMSEC(cls, data):
        print('===========  Method:  GEMSEC  ===============')
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']

        num_c = max(labels)
        model = GEMSEC(walk_number=5, walk_length=10, dimensions=64, negative_samples=30,
                       window_size=6, learning_rate=0.1, clusters=num_c, gamma=0.1, seed=42)
        model.fit(observe_graph)
        F_ = model.get_memberships()
        F_ = np.array(list(F_.values()))
        F_argmax = F_ + 1

        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byDANMF(cls, data):
        print('===========  Method:  DANMF  ===============')
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']
        model = DANMF(layers=[16, 2], pre_iterations=320,
                      iterations=320, seed=42, lamb=0.01)
        model.fit(observe_graph)
        F_ = model.get_memberships()
        F_ = np.array(list(F_.values()))
        F_argmax = F_ + 1
        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byLouvain(cls, data):
        print('===========    louvain   ===============')
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']

        res = community_louvain.best_partition(observe_graph, resolution=2)
        F_argmax = np.array(list(res.values())) + 1

        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byBigClam(cls, data):
        print('===========  Method:  BigClam  ===============')
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']
        num_nodes = len(observe_graph.nodes)
        model = BigClam(num_of_nodes=num_nodes, dimensions=64, iterations=500, learning_rate=0.003, seed=42)
        model.fit(observe_graph)
        F_ = model.get_memberships()
        F_ = np.array(list(F_.values()))
        F_argmax = F_ + 1

        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byAllMethod(cls, data):
        cls.train_byMNDP_Missing(data)
        cls.train_byMNDPEM(data)
        cls.train_byDANMF(data)
        cls.train_byGEMSEC(data)
        cls.train_byBigClam(data)
        cls.train_byLouvain(data)

    @classmethod
    def Experiment_intro_case(cls, mode=1):
        """

        :param mode: =1 -> search suitable cases; =2 -> display the specified case
        :return:
        """
        data = Read.read_karate_club()
        g_ori = data['graph_real']
        true_labels = data['labels']
        g_ori_painted = cls.paint_color(g_ori, g_ori, [], true_labels)
        Draw.draw_karate(g_ori_painted, labels=true_labels, fig_title='karate club network with ground trues')

        if mode != 1:
            del_edges = [(0, 6), (0, 8), (1, 7), (1, 17), (1, 30), (2, 3), (2, 32), (29, 33)]
            g_obs = g_ori.copy()
            g_obs.remove_edges_from(del_edges)
            g_painted = cls.paint_color(g_obs, g_obs, del_edges, true_labels)
            Draw.draw_karate(g_painted, true_labels, fig_title='Karate network with 10% missing edges')

            # CLMC on 10% edges removed network
            # link prediction
            # predicted true edges: (1, 7), (29, 33), (1, 17)
            # predicted false edges: (10, 16)
            # node with wrong label: 2
            clmc_labels = [
                1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1,
                2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ]
            clmc_g_res = g_obs.copy()
            clmc_g_res.add_edges_from([(1, 7), (29, 33), (1, 17), ])
            clmc_g_painted = cls.paint_color(g_obs, clmc_g_res, del_edges, clmc_labels)
            # clmc_g_painted.nodes[2]['color'] = CONF.NODE_LABELs[-1]
            Draw.draw_karate(clmc_g_painted, clmc_labels, 'A. CLMC on missing-edges network')

            # MNDPEM
            predicted_edges = [(0, 33), (2, 3), (13, 12), (5, 3)]
            em_g = g_obs.copy()
            em_g.add_edges_from(predicted_edges)
            em_labels = [
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1,
                1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ]
            em_g_painted = cls.paint_color(g_obs, em_g, del_edges, em_labels)
            Draw.draw_karate(em_g_painted, em_labels, 'B. Our model on missing-edges network')

    @classmethod
    def Experiment_known_network(cls):
        dataset = [
            # Read.read_karate_club,
            # Read.read_dolphins,
            Read.read_football,
            # Read.read_polbooks,
            Read.read_polblogs,
        ]

        methods = [
            # cls.train_byMNDP_Missing,
            # cls.train_byDANMF,
            cls.train_byGEMSEC,
            # cls.train_byLouvain,
            cls.train_byBigClam,
            # cls.train_byMNDPEM,
        ]

        for data_ in dataset:
            for method_ in methods:
                res = defaultdict(list)
                for i in range(15):
                    np.random.seed(i)
                    data = cls.prepare_data(data_, missing_rate=0.2)
                    tmp = method_(data)
                    print(f'time -> {i}')
                    if method_.__name__ == 'train_byMNDPEM':
                        res['nmi'].extend(tmp['nmi'])
                        res['ari'].extend(tmp['ari'])
                        res['pur'].extend(tmp['pur'])
                    else:
                        res['nmi'].append(tmp['nmi'])
                        res['ari'].append(tmp['ari'])
                        res['pur'].append(tmp['pur'])
                cls.res2csv(res,
                            path=f'../res/{data_.__name__.split("_")[1] + "_" + method_.__name__.split("_")[1]}.csv')

    @classmethod
    def Experiment_unknown_network(cls):
        dataset = [
            # Read.read_adjnoun,
            # Read.read_celegansneural,
            Read.read_email,
            # Read.read_jazz,
            # Read.read_lesmis,
        ]

        methods = [
            # cls.train_byMNDP_Missing,
            # cls.train_byMNDPEM,
            # cls.train_byDANMF,
            # cls.train_byGEMSEC,
            # cls.train_byLouvain,
            cls.train_byBigClam
        ]

        for data_ in dataset:
            for method_ in methods:
                res = defaultdict(list)
                for i in range(15):
                    np.random.seed(i)
                    data = cls.prepare_data(data_, missing_rate=0.2)
                    tmp = method_(data)
                    print(f'time -> {i}')
                    print(tmp['Q'])
                    if method_.__name__ == 'train_byMNDPEM':
                        res['Q'].extend(tmp['Q'])
                    else:
                        res['Q'].append(tmp['Q'])
                cls.res2csv(res,
                            path=f'../res/{data_.__name__.split("_")[1] + "_" + method_.__name__.split("_")[1]}.csv')

    @classmethod
    def Experiment_different_missing_rate(cls, dataset):
        missing_rate = [i * 0.05 for i in range(0, 8)]
        for data_read in dataset:
            for rate in missing_rate:
                print("Missing rate -> {}".format(rate))
                data = cls.prepare_data(data_read, rate)
                cls.train_byMNDP_Missing(data)

    @classmethod
    def Experiment_DBLP_case(cls):
        data = cls.prepare_data(Read.read_dblp_time1_subgraph, missing_rate=0)
        data['num_del_edges'] = 300
        cls.train_byMNDP_Missing(data)

    @classmethod
    def Experiment_case_study(cls, method, network, draw_network, epoch=1):
        from Experiment_case_study import case_study
        clmc_del_edges = case_study.read_del_edges_CLMC_dolph()
        data = cls.prepare_data(network, missing_rate=0.0, del_list=clmc_del_edges)
        observe_graph = data['observe_graph']
        del_edges = data['del_edges']
        print(f'del_edges: {del_edges}')
        res = method(data)
        # cls.res2csv(res, '../res/case_study_' + str(epoch) +'.csv')
        F_argmax = res['F_argmax']
        with open('../res/case_study_metrics.csv', 'a') as f:
            f.write(f'{epoch} {res["nmi"][-1]}\n')
        with open(f'../res/case_study_F_argmax.txt', 'a') as f:
            f.write(f'{epoch}-----------\n')
            f.write(str(F_argmax))
            f.write('\n')
            f.write(str(del_edges))
            f.write('\n')
        g_res = res['graph_res']
        res_g_painted = cls.paint_color(observe_graph, g_res, del_edges, F_argmax)
        draw_network(res_g_painted, F_argmax, fig_title='Our model', save_path=f'../res/case_study_{epoch}.png')


def main(stg_model):
    dataset = [
        # Read.read_dolphins,
        Read.read_polblogs
    ]
    stg_model.Experiment_different_missing_rate(dataset)


def main_case_study():
    import os
    stg_model = Strategy()
    if os.path.exists('../res/case_study_metrics.csv'):
        os.remove('../res/case_study_metrics.csv')
    if os.path.exists('../res/case_study_F_argmax.txt'):
        os.remove('../res/case_study_F_argmax.txt')
    for i in range(30):
        stg_model.Experiment_case_study(
            method=lambda data: stg_model.train_byMNDPEM(data, num_EM_iter=20, alpha=0.4),
            network=Read.read_dolphins,
            draw_network=Draw.display_dolphins, epoch=i)


def main_test_nothing():
    res = Read.read_polbooks()
    labels = res['labels']
    F_argmax = [2, 1, 1, 3, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 2, 2, 2,
                2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3,
                3, 3, 3,
                1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1]
    print(tools.display_result(F_argmax, labels))


def main_intro_case():
    Strategy.Experiment_intro_case(mode=2)


def main_test():
    data = Strategy.prepare_data(Read.read_pubmed)
    res = Strategy.train_byLouvain(data)
    print(res['nmi'])


if __name__ == '__main__':
    # main_case_study()
    # main_test_nothing()
    # main_intro_case()
    # Strategy.Experiment_known_network()
    # Strategy.Experiment_unknown_network()
    main_case_study()
