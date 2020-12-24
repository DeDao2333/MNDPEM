import networkx as nx
import time
import numpy as np
import random
import re
from method.community_detection import MNDP
from util import base_method as tools
from collections import defaultdict


class MNDPEM_model(object):

    def __init__(self,
                 N: int, C: int, F: np.ndarray = None,
                 num_EM_iter=15, num_sample=500, num_warm_up=4000):
        self.F = F
        self.C = C
        self.N = N
        self.num_EM_iter = num_EM_iter
        self.num_sample = num_sample
        self.num_warm_up = num_warm_up

    def E_init(self, Z_noEdge, Z_Edge, num_del_edge):
        """
        对删除部分的边数进行初始化
        :param num_del_edge:
        :param Z_Edge:
        :param num_del_edges: 删除边的数量
        :param Z_noEdge：所有没有边的区域，保存的是边列表 list [ tuple ]
        :return: 初始化边列表  形式：[(1,2), (2,3)]
        """
        assert num_del_edge != 0, "num_del_edge can not be 0"
        all_prob = self.get_allProb()
        prob_in_miss_edges = []
        index_edges = {}

        for i, e in enumerate(Z_noEdge.edges):
            prob_in_miss_edges.append(all_prob[e[0], e[1]])
            index_edges.update({i: e})

        # 按照概率，对所有无边的部分进行边的采样
        samples_index = np.random.choice(
            list(index_edges.keys()),
            num_del_edge,
            replace=False,
            p=prob_in_miss_edges / sum(prob_in_miss_edges))

        # 将 初始化 得到的采样边, 添加到 Z_edge 中，同时删除Z_noEdge中对应的边
        for sample in samples_index:
            e = index_edges[sample]
            Z_Edge.add_edge(e[0], e[1])
            Z_noEdge.remove_edge(e[0], e[1])

        return Z_Edge, Z_noEdge

    def E_step(self, Z_noEdge: nx.Graph, Z_Edge: nx.Graph):
        # 要区分 Z 区域和 graph  中正确边，是要对 Z 区域进行采样，而不是对整个图采样
        # Z 区域包含：缺失的边，预测的边
        start = time.time()

        all_prob = self.get_allProb()
        samples_Z_edge = []  # 每个元素是一个 Z_Edge

        # 在 Z—noEdge 区域根据 F 选一条边
        Z_noEdge_allProb = {}
        for _, e in enumerate(Z_noEdge.edges):
            Z_noEdge_allProb[str(e)] = all_prob[e[0], e[1]]
            # 以下需要迭代
        for epoch in range(self.num_warm_up + self.num_sample):
            if epoch >= self.num_warm_up:
                samples_Z_edge.append(Z_Edge.copy())  # M-H 采样预热结束后，开始正式采样
            if epoch % 1000 == 0:
                print("warm-up: ", epoch)

            # 在 Z—edge 区域随机选一条边去掉
            edges_z = list(Z_Edge.edges)
            rdm = random.randint(0, len(edges_z) - 1)
            pre_edge = edges_z[rdm]
            Z_Edge.remove_edge(pre_edge[0], pre_edge[1])
            Z_noEdge.add_edge(pre_edge[0], pre_edge[1])
            Z_noEdge_allProb[str(pre_edge)] = all_prob[pre_edge[0], pre_edge[1]]

            sample_index = np.random.choice(
                list(Z_noEdge_allProb.keys()),
                1,
                replace=True,
                p=np.array(list(Z_noEdge_allProb.values())) / sum(list(Z_noEdge_allProb.values())))

            new_edge = sample_index[0]

            # 判断是否是去掉的那条边, 如果是，将删除的边恢复，进行下一次采样
            if new_edge == str(pre_edge):
                Z_Edge.add_edge(pre_edge[0], pre_edge[1])
                Z_noEdge.remove_edge(pre_edge[0], pre_edge[1])
                Z_noEdge_allProb.pop(str(pre_edge))
                continue

            # 如果不是，计算接受率
            res = re.findall(r"\d+", new_edge)  # 获取字符串中的数字，也就是结点的序号，从而构成边，比如：res : ['47', '12']
            e1, e2 = int(res[0]), int(res[1])
            x_prob = all_prob[pre_edge[0], pre_edge[1]]
            y_prob = all_prob[e1, e2]
            accept_rate = self.calculate_Accept(x_prob, y_prob)
            val = np.random.uniform(0, 1)
            if val < accept_rate:
                # 如果接受，添加到 Z—edge 中，在 Z—noEdge 中删除
                Z_Edge.add_edge(e1, e2)
                Z_noEdge.remove_edge(e1, e2)
                Z_noEdge_allProb.pop(new_edge)
            else:
                # 拒绝
                Z_Edge.add_edge(pre_edge[0], pre_edge[1])
                Z_noEdge.remove_edge(pre_edge[0], pre_edge[1])
                Z_noEdge_allProb.pop(str(pre_edge))
        end = time.time()
        print("Time -> E_Gibbs_every_step : {:.4f}".format(end - start))
        return samples_Z_edge

    def M_step(self, observe_graph: nx.Graph, samples_Z_edge, cur_epoch):
        # 融合 observe_graph + Z_edge = H
        B = self.mergeAll_G_Z(observe_graph, samples_Z_edge)
        G = nx.from_numpy_array(B)
        all_prob = self.get_allProb()

        def find_lonely_node(g):
            # check outline, and produce an edge with prob between outline node and other node
            degree_list = nx.degree(g)
            for n_d in degree_list:
                if n_d[1] is 0:
                    u = n_d[0]
                    v = np.argmax(all_prob[u])
                    g.add_edge(u, v)

        find_lonely_node(G)  # 检测是否有孤立点

        F = np.random.uniform(0, 1, size=(self.N, self.C))
        self.F = MNDP.MNDP_EM(G, self.C, F, epoch=cur_epoch / 2)
        F_argmax = np.argmax(self.F, axis=1) + 1  # 注意 twitter 数据不需要加1
        return F_argmax, G

    def train(self, observe_graph, label, Z_noEdge_ori, Z_Edge_ori, num_del_edge,
              is_unknown_data):
        # M步不同，整合了E步所有样本到一起
        res = defaultdict(list)
        if self.F is None:
            self.F = np.random.uniform(0, 1, size=(self.N, self.C))
        for epoch in range(self.num_EM_iter):
            print(f'epoch: {epoch}')
            Z_Edge = Z_Edge_ori.copy()
            Z_noEdge = Z_noEdge_ori.copy()
            self.E_init(Z_noEdge, Z_Edge, num_del_edge)
            # E 步Gibbs采样
            samples_Z_edge = self.E_step(Z_noEdge, Z_Edge)
            F_argmax, G = self.M_step(observe_graph, samples_Z_edge, epoch)

            # 社团发现结果
            res['F_argmax'] = F_argmax
            res['graph_res'] = G
            if is_unknown_data:
                Q = tools.Modularity_Q(F_argmax, G)
                print("Modularity_Q: {:.4f}".format(Q))
                res['Q'].append(Q)
            else:
                tmp = tools.display_result(F_argmax, label)
                res['nmi'].append(tmp[0])
                res['ari'].append(tmp[1])
                res['pur'].append(tmp[2])

        return res

    @staticmethod
    def mergeAll_G_Z(G: nx.Graph, samples_Z_edge: list, alpha: float = 0.3):
        # r = 0.025 * epoch_ * 0.7 + 0.2
        res = 0
        i = 0
        for Z_ in samples_Z_edge:
            H = G.copy()
            H.add_edges_from(list(Z_.edges))
            B = nx.to_numpy_array(H, nodelist=sorted(H.nodes))
            res += B
            i += 1
        res = res / i
        res[res > alpha] = 1
        res[res <= alpha] = 0
        print("sample Z plus G :  ", np.sum(res))
        return res

    @staticmethod
    def calculate_Accept(x_prob, y_prob):
        return min(1, (1. - y_prob) / (1. - x_prob + 1e-8))

    def get_allProb(self):
        # 得到对应于邻接矩阵的概率矩阵，每个元素表示边生成的概率
        # return 1. - np.exp(-1. * F.dot(F.T))
        return self.F.dot(self.F.T)
