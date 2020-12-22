from karateclub.community_detection.overlapping import BigClam
import numpy as np
import pandas as pd
import networkx as nx
import random
from community import community_louvain
from util import draw_graph
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
        if is_unknown:
            Q = tools.Modularity_Q(F_argmax, graph)
            print("Modularity_Q: {:.4f}".format(Q))
            return Q
        else:
            return tools.display_result(F_argmax, labels)

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
            df.to_csv(path)
        elif 'Q' in res.keys():
            # represent Q
            df = pd.DataFrame(res, columns=['Q'])
            df.to_csv(path)
        else:
            assert False, "Not right index"

    @staticmethod
    def train_byMNDPEM(data):
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
        ## 绘制 karate 示例专用，其他数据集报错
        # draw_karate(observe_graph_, F_argmax)
        return cls.res_display(is_unknown, F_argmax, observe_graph, labels)

    @classmethod
    def train_byGEMSEC(cls, data):
        print('===========  Method:  GEMSEC  ===============')
        observe_graph = data['observe_graph']
        labels = data['labels']
        is_unknown = data['is_unknown']

        num_c = max(labels)
        model = GEMSEC(walk_number=5, walk_length=10, dimensions=32, negative_samples=10,
                       window_size=4, learning_rate=0.1, clusters=num_c, gamma=0.1, seed=42)
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
                      iterations=220, seed=42, lamb=0.01)
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

        res = community_louvain.best_partition(observe_graph)
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
    def Experiment_known_network(cls):
        dataset = [Read.read_karate_club,
                   Read.read_dolphins,
                   Read.read_football
                   ]

        methods = [
                   cls.train_byMNDP_Missing,
                   cls.train_byMNDPEM,
                   cls.train_byDANMF,
                   cls.train_byGEMSEC,
                   cls.train_byLouvain,
                   # cls.train_byBigClam
                   ]

        for data_ in dataset:
            for method_ in methods:
                res = defaultdict(list)
                for i in range(15):
                    data = cls.prepare_data(data_, missing_rate=0.2)
                    tmp = method_(data)
                    print(f'time -> {i}')
                    if method_.__name__ == 'train_byMNDPEM':
                        res['nmi'].extend(tmp['nmi'])
                        res['ari'].extend(tmp['ari'])
                        res['pur'].extend(tmp['pur'])
                    else:
                        res['nmi'].append(tmp[0])
                        res['ari'].append(tmp[1])
                        res['pur'].append(tmp[2])
                cls.res2csv(res, path=f'../res/{data_.__name__.split("_")[1] + "_" + method_.__name__.split("_")[1]}.csv')

    @classmethod
    def Experiment_different_missing_rate(cls, dataset):
        missing_rate = [i * 0.05 for i in range(0, 2)]
        for data_read in dataset:
            for rate in missing_rate:
                print("Missing rate -> {}".format(rate))
                data = cls.prepare_data(data_read, rate)
                cls.train_byLouvain(data)

    @classmethod
    def Experiment_DBLP_case(cls):
        data = cls.prepare_data(Read.read_dblp_time1_subgraph, missing_rate=0)
        data['num_del_edges'] = 300
        cls.train_byMNDP_Missing(data)


def main(stg_model):
    dataset = [
        # Read.read_dolphins,
        Read.read_football
    ]
    stg_model.Experiment_different_missing_rate(dataset)


def main2(stg_model: Strategy):
    stg_model.Experiment_DBLP_case()


def main3(stg: Strategy):
    print(stg.train_byMNDPEM.__name__)


def main4(stg):
    stg.Experiment_known_network()


if __name__ == '__main__':
    stg_model = Strategy()
    # main(stg_model)
    # main2(stg_model)
    # main3(stg_model)
    main4(stg_model)
