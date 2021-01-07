import os
import re
import sys

import util.draw_graph as Draw
import util.read_data as Read
from model.Strategy import Strategy

sys.path.append(os.path.abspath('..'))


def _read_del_edges_CLMC(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            val = re.findall(r"\d+", line)
            res.append((int(val[0]) - 1, int(val[1]) - 1))
        return res


def _read_cluster_CLMC(path):
    with open(path, 'r') as f:
        res = []
        lines = f.readlines()
        for line in lines:
            res.append(int(line.strip()))
        return res


def _read_dolph_labels(path):
    with open(path, 'r') as f:
        line = f.readline()
        line = line.strip()
        vals = re.findall(r"\d+", line)
        F_argmax = []
        for i in vals:
            F_argmax.append(int(i))
        return F_argmax


def read_mndp_dolph_labels():
    return _read_dolph_labels(sys.path[-1] + '/Experiment_case_study/data/res_dolph_mndp_0.75nmi.txt')


def read_our_dolph_labels():
    return _read_dolph_labels(sys.path[-1] + '/Experiment_case_study/data/res_dolph_our_0.88nmi.txt')


def read_cluster_CLMC_polbook():
    return _read_cluster_CLMC(sys.path[-1] + '/Experiment_case_study/data/clmc_cluster_polbook.txt')


def read_del_edges_CLMC_polbook():
    return _read_del_edges_CLMC(sys.path[-1] + '/Experiment_case_study/data/clmc_del_edges_polbook.txt')


def read_cluster_CLMC_dolph():
    return _read_cluster_CLMC(sys.path[-1] + '/Experiment_case_study/data/clmc_cluster_dolph.txt')


def read_del_edges_CLMC_dolph():
    return _read_del_edges_CLMC(sys.path[-1] + '/Experiment_case_study/data/clmc_del_edges_dolph.txt')


def main_polbook():
    del_edges = read_del_edges_CLMC_polbook()
    data = Strategy.prepare_data(Read.read_polbooks, del_list=del_edges)
    g_real = data['graph_real']
    g_obs = data['observe_graph']
    labels = data['labels']

    # complete network
    Draw.display_polbooks(g_real, labels, fig_title='complete', show=True)

    # 20% edges removed network -> MNDP
    g_obs_painted = Strategy.paint_color(g_obs, g_obs, del_edges, labels)
    Draw.display_polbooks(g_obs_painted, labels, fig_title='MNDP on network with 20% edges removed', show=True)

    # 20% edges removed network -> CLMC
    # add edges: (1, 5), (10, 14), (4, 12), (16, 13), (53, 13), (38, 9), (46, 9), (45, 10), (36, 24), (69, 31)
    clmc_add_edges = [(1, 5), (10, 14), (4, 12), (16, 13), (53, 13), (38, 9), (46, 9), (45, 10), (36, 24), (69, 31), ]
    clmc_g = g_obs.copy()
    clmc_g.add_edges_from(clmc_add_edges)
    clmc_labels = read_cluster_CLMC_polbook()
    clmc_g_painted = Strategy.paint_color(g_obs, clmc_g, del_edges, clmc_labels)
    Draw.display_polbooks(clmc_g_painted, clmc_labels, fig_title='CLMC', show=True)

    # 20% edges removed network -> our method
    # add edges:
    our_add_edges = [(36, 38), (36, 37), (38, 54), ]


def main_dolph(train_mode=False):
    del_edges = read_del_edges_CLMC_dolph()
    data = Strategy.prepare_data(Read.read_dolphins, del_list=del_edges)
    g_real = data['graph_real']
    g_obs = data['observe_graph']
    labels = data['labels']

    # Complete network
    Draw.display_dolphins(g_real, labels, save_path='./res/dolph_true.png', show=True, fig_title='ground true')

    # 20% edges removed
    g_obs_painted = Strategy.paint_color(g_obs, g_obs, del_edges, label=labels)
    Draw.display_dolphins(g_obs_painted, labels, save_path='./res/dolph_0.2_missing.png', fig_title='20% edges missing',
                          show=True)

    # MNDP on 20% edges removed network
    # node with wrong label: 28, 30, 39
    if train_mode:
        mndp_res = Strategy.train_byMNDP_Missing(data)
        mndp_labels = mndp_res['F_argmax']
    else:
        mndp_labels = read_mndp_dolph_labels()
    mndp_g = g_obs.copy()
    print(mndp_labels)
    mndp_g_painted = Strategy.paint_color(g_obs, mndp_g, del_edges, label=mndp_labels)
    Draw.display_dolphins(mndp_g_painted, mndp_labels, fig_title='mndp on 20% network', show=True)

    # CLMC on 20% edges removed network
    # node with wrong label:
    clmc_g = g_obs.copy()
    clmc_labels = read_cluster_CLMC_dolph()
    clmc_g_painted = Strategy.paint_color(g_obs, clmc_g, del_edges, label=clmc_labels)
    Draw.display_dolphins(clmc_g_painted, clmc_labels, fig_title='clmc on 20% network', show=True)


if __name__ == '__main__':
    # main()
    # res = Read.read_polbooks()
    # g = res['graph_real']
    # print(g.edges)
    # labels = res['labels']
    # Draw.display_polbooks(g, labels, show=True)
    main_dolph()
    # print(read_mndp_dolph_labels())
