from karateclub.community_detection.overlapping import BigClam
import numpy as np
import pandas as pd
import networkx as nx
import random
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


class Strategy(object):
    def __init__(self):
        pass

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

    @staticmethod
    def prepare_data(read_data, missing_rate=0.2):
        res: dict = read_data()
        # 删除一定比例的边数，得到观测图
        observe_graph, test_edges = tools.get_train_test(res['graph_real'], missing_rate)
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
    def train_byMNDPEM(data, num_EM_iter=15):
        print('===========  Method:  MNDPEM  ===============')
        C = data['C']
        N = data['N']
        ob = data['observe_graph']
        labels = data['labels']
        Z_noEdge_ori = data['Z_noEdge_ori']
        Z_Edge_ori = data['Z_Edge_ori']
        num_del_edges = data.get('num_del_edges', 20)
        is_unknown = data['is_unknown']
        our_method = MNDPEM_model(N, C)
        our_method.train_mode = 2
        our_method.num_EM_iter = num_EM_iter
        return our_method.train(ob, labels, Z_noEdge_ori, Z_Edge_ori, num_del_edges, is_unknown)

    @classmethod
    def train_byMNDP_Missing(cls, data):
        print('===========  Method:  MNDP-Missing  ===============')
        C = data['C']
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']
        # F_ = MNDP.MNDP_EM(observe_graph, C, epoch=16)
        prob, F_ = MNDP.MNDP(data['adj_real'], C)
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
        model = BigClam(num_of_nodes=num_nodes, dimensions=8, iterations=300, learning_rate=0.003, seed=42)
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
    def Experiment_intro_case(cls):
        from util.draw_graph import display_intro_case
        display_intro_case(mode=1)

    @classmethod
    def Experiment_known_network(cls):
        dataset = [
            Read.read_karate_club,
            Read.read_dolphins,
            Read.read_football,
            Read.read_wisconsin,
            Read.read_polbooks,
            Read.read_polblogs,
        ]

        methods = [
            # cls.train_byMNDP_Missing,
            # cls.train_byDANMF,
            # cls.train_byGEMSEC,
            # cls.train_byLouvain,
            # cls.train_byBigClam,
            cls.train_byMNDPEM,
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
            Read.read_adjnoun,
            Read.read_celegansneural,
            Read.read_email,
            Read.read_jazz,
            Read.read_lesmis,
        ]

        methods = [
            # cls.train_byMNDP_Missing,
            cls.train_byMNDPEM,
            # cls.train_byDANMF,
            # cls.train_byGEMSEC,
            # cls.train_byLouvain,
            # cls.train_byBigClam
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
    def Experiment_case_study(cls, network, draw_network, epoch=1):
        LINK_COLORs = {
            'exist': '#5E5E5E',
            'removed': '#D4D4D4',
            'predict': '#9ACD32'}
        data = cls.prepare_data(network)
        observe_graph = data['observe_graph']
        del_edges = data['del_edges']
        res = cls.train_byMNDPEM(data, num_EM_iter=23)
        cls.res2csv(res, '../res/case_study_' + str(epoch) +'.csv')
        F_argmax = res['F_argmax']
        with open(f'../res/case_study_F_argmax.txt', 'a') as f:
            f.write(f'---------------------{epoch}\n')
            f.write(str(F_argmax))
            f.write('\n')
            f.write(str(del_edges))
            f.write('\n')
        g_res = res['graph_res']

        for i, j in observe_graph.edges:
            observe_graph.edges[i, j]['color'] = LINK_COLORs['exist']

        observe_graph.add_edges_from(del_edges)
        for i, j in del_edges:
            observe_graph.edges[i, j]['color'] = LINK_COLORs['removed']

        predicted_edges = list(set(g_res.edges) - set(observe_graph.edges))
        observe_graph.add_edges_from(predicted_edges)
        for i, j in predicted_edges:
            observe_graph.edges[i, j]['color'] = LINK_COLORs['predict']

        draw_network(observe_graph, F_argmax, fig_title='Our model', epoch=epoch)
        # draw_network(isGround_trues=True)


def main(stg_model):
    dataset = [
        # Read.read_dolphins,
        Read.read_polblogs
    ]
    stg_model.Experiment_different_missing_rate(dataset)


def main_case_study(stg_model: Strategy):
    for i in range(30):
        stg_model.Experiment_case_study(
            network=Read.read_dolphins,
            draw_network=Draw.display_dolphins, epoch=i)


def main_test_nothing(stg_model: Strategy):
    data = Read.read_dolphins()
    g = data['graph_real']
    labels = data['labels']
    node_color = Draw.get_color(labels)
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, with_labels=True, node_color=node_color,  node_size=100, font_size=6)
    plt.show()
    print(pos)


if __name__ == '__main__':
    stg_model = Strategy()
    # main(stg_model)
    # main2(stg_model)
    # main_test_nothing(stg_model)
    # main3(stg_model)
    # main4(stg_model)
    # stg_model.Experiment_unknown_network()
    # stg_model.Experiment_known_network()
    main_case_study(stg_model)
