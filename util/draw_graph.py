import matplotlib.pyplot as plt
import networkx as nx
from numpy.core._multiarray_umath import array
from util import read_data as Read


def display_rate_football():
    # football
    MNDP_M = [0.9254, 0.9008, 0.8883, 0.8517, 0.8516, 0.8297, 0.8080, 0.7857]
    MNDP_EM = [0.9263, 0.9242, 0.9074, 0.9036, 0.9036, 0.9021, 0.8679, 0.8258]
    CLMC = [0.9232, 0.9239, 0.9232, 0.9232, 0.9043, 0.8657, 0.8391, 0.8211]
    GEMSEC = [0.8923, 0.8835, 0.8866, 0.8678, 0.8652, 0.8438, 0.8032, 0.7933]
    # BigClam = [0.3698, 0.3554, 0.3331, 0.3167, 0.2845, 0.2669]
    DANMF = [0.9038, 0.8915, 0.9026, 0.8868, 0.9006, 0.8746, 0.8492, 0.8289]
    Louvain = [0.8903, 0.8850, 0.8795, 0.8582, 0.8513, 0.8415, 0.8288, 0.8213]

    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    # plt.plot(rate, BigClam, marker='o', mec='r', mfc='w')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Football', fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.savefig("football.png")
    plt.show()


def display_rate_karate():
    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']
    # karate
    MNDP_M = [1.0000, 0.8372, 0.8365, 0.8365, 0.7308, 0.6499, 0.5739, 0.5159]
    MNDP_EM = [1.0000, 1.0000, 1.0000, 0.8372, 0.8372, 0.7308, 0.7329, 0.6459]
    CLMC = [1.0000, 1.0000, 0.8365, 0.8333, 0.8333, 0.5743, 0.4175, 0.3476]
    GEMSEC = [1.0000, 1.0000, 0.8372, 0.8365, 0.7329, 0.6766, 0.6499, 0.5801]
    BigClam = [1.0000, 0.8372, 0.8255, 0.8255, 0.7201, 0.6494, 0.5883, 0.5618]
    DANMF = [1.0000, 0.8365, 0.8365, 0.8372, 0.7329, 0.6169, 0.4765, 0.4177]
    Louvain = [0.7071, 0.7071, 0.7071, 0.6873, 0.6175, 0.5804, 0.5214, 0.4923]

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    plt.plot(rate, BigClam, c='#6A5ACD', marker='o', ms=9, label='BigClam')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Zachary’s karate club', fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.savefig("karate.png")
    plt.show()


def display_rate_dolphins():
    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']

    # Dolphins
    MNDP_M = [0.8888, 0.8873, 0.8141, 0.7532, 0.7011, 0.6544, 0.6270, 0.4855]
    MNDP_EM = [0.8991, 0.8936, 0.8865, 0.8870, 0.8870, 0.8870, 0.8083, 0.6147]
    CLMC = [0.8809, 0.8809, 0.8809, 0.8809, 0.8809, 0.8782, 0.7769, 0.5562]
    GEMSEC = [0.8888, 0.8141, 0.8141, 0.7769, 0.7036, 0.6333, 0.6270, 0.5752]
    BigClam = [0.8888, 0.8888, 0.7656, 0.7783, 0.6333, 0.5673, 0.5495, 0.4478]
    DANMF = [0.8141, 0.7543, 0.6040, 0.5660, 0.5449, 0.4809, 0.4530, 0.4064]
    Louvain = [0.8141, 0.7532, 0.6041, 0.5665, 0.5400, 0.5199, 0.4877, 0.4685]

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    plt.plot(rate, BigClam, c='#6A5ACD', marker='o', ms=9, label='BigClam')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, CLMC, c='#A25EA2', marker='o', ms=9, label='CLMC')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Dolphin social network', fontsize=14)
    plt.legend(loc="lower left", ncol=2, fontsize=9)
    plt.savefig("Dolphins.png")
    plt.show()


def display_rate_polblogs():
    rate = ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%']

    # Dolphins
    MNDP_M = [54.73, 53.24, 52.79, 51.32, 51.08, 48.23, 43.71, 42.32]
    MNDP_EM = [54.90, 54.35, 53.78, 52.97, 52.07, 50.10, 46.51, 43.79]
    GEMSEC = [51.17, 50.91, 50.23, 49.30, 49.36, 47.03, 41.46, 40.02]
    DANMF = [52.28, 51.81, 51.31, 43.05, 42.44, 42.17, 41.34, 40.96]
    Louvain = [39.36, 38.91, 36.83, 36.22, 35.45, 35.13, 35.06, 34.79]

    plt.figure(figsize=(6, 4.3))
    plt.plot(rate, GEMSEC, c='#6B6B6B', marker='o', ms=9, label='GEMSEC')
    plt.plot(rate, DANMF, c='#228B22', marker='o', ms=9, label='DANMF')
    plt.plot(rate, Louvain, c='#DD6D22', marker='o', ms=9, label='Louvain')
    plt.plot(rate, MNDP_M, c='#A0522D', marker='<', ms=9, label='MNDP-M')
    plt.plot(rate, MNDP_EM, c='#2B91D5', marker='D', ms=8, label='MNDPEM')
    # fig.suptitle('Categorical Plotting')
    plt.xlabel('Different ratio of missing edges', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    plt.title('Political blogs', fontsize=14)
    plt.legend(loc="lower left", ncol=2, fontsize=9)
    plt.savefig("polblogs.png")
    plt.show()


def draw_karate(g, labels, fig_title):
    def get_color(labels):
        color_list = []
        for i in labels:
            if i == 1:
                color_list.append('#CD4F39')
            else:
                color_list.append('#96CDCD')
        return color_list

    pos = {0: array([0.04035086, -0.35842333]), 1: array([0.23988347, -0.23390369]), 2: array([-0.0759524, 0.08270998]),
           3: array([0.17775027, -0.36712044]), 4: array([-0.18152543, -0.59496255]),
           5: array([-0.01425569, -0.75517756]),
           6: array([0.10152675, -0.76973825]), 7: array([0.31943218, -0.34824782]), 8: array([0.00369568, 0.05416653]),
           9: array([-0.21782102, 0.23451488]), 10: array([-0.08396947, -0.64399256]),
           11: array([-0.25590924, -0.47912845]),
           12: array([0.16543092, -0.57334954]), 13: array([0.07700945, -0.11461636]),
           14: array([-0.33162734, 0.69163498]),
           15: array([-0.08352861, 0.53054236]), 16: array([0.04001051, -1.00]), 17: array([0.26784617, -0.48463474]),
           18: array([-0.41773533, 0.57434359]), 19: array([0.15999416, -0.04323504]),
           20: array([0.35045052, 0.37016786]),
           21: array([0.36654177, -0.43238517]), 22: array([-0.19627026, 0.75827933]),
           23: array([-0.00512236, 0.59473913]),
           24: array([-0.39940143, 0.14419955]), 25: array([0.06879779, 0.92014959]),
           26: array([0.28224878, 0.50139926]),
           27: array([-0.17682913, 0.32894827]), 28: array([-0.20932184, 0.11379152]),
           29: array([0.08351973, 0.51382893]),
           30: array([0.1536715, 0.14563396]), 31: array([-0.16937923, -0.03755861]),
           32: array([-0.10773608, 0.41913559]), 33: array([0.02822436, 0.25828881])}

    plt.figure(figsize=(6, 8))
    color_list_ = get_color(labels)

    edges_colors = []
    for i, j in g.edges():
        edges_colors.append(g.edges[i, j]['color'])
    nx.draw_networkx(g, pos=pos, node_color=color_list_, edge_color=edges_colors, width=1.7)
    plt.title(fig_title, fontsize=14)
    plt.show()


def display_polbooks(g, labels, fig_title):
    def get_color(labels):
        color_list = []
        for i in labels:
            if i == 1:
                color_list.append('#CD4F39')
            elif i == 2:
                color_list.append('#DAA520')
            elif i == 3:
                color_list.append('#96CDCD')
        return color_list

    color_list_ = get_color(labels)
    edges_colors = []
    for i, j in g.edges():
        edges_colors.append(g.edges[i, j]['color'])
    nx.draw_networkx(g, node_color=color_list_, edge_color=edges_colors, width=1.7)
    plt.title(fig_title, fontsize=14)
    plt.show()


def draw_simple_graph(graph, labels):
    plt.subplots(1, 1, figsize=(15, 13))
    nx.draw_spring(graph, with_labels=True, node_color=labels)
    plt.show()


def _draw_different_rate():
    display_rate_karate()
    display_rate_dolphins()
    display_rate_football()
    display_rate_polblogs()


def display_intro_case(mode=1):
    """

    :param mode:  display mode
    :return:
    """
    from util import base_method as tools
    from model.Strategy import Strategy

    stg = Strategy()

    ALL_COLORs = {
        'exist': '#5E5E5E',
        'removed': '#D4D4D4',
        'predict': '#CD00CD'}
    # true graph
    res = Read.read_karate_club()
    g_ = res['graph_real']
    true_labels = res['labels']
    for i, j in g_.edges():
        g_.edges[i, j]['color'] = ALL_COLORs['exist']
    draw_karate(g_, true_labels, fig_title='Louvain on complete karate network')

    # remove 10% edges
    del_edges = [(0, 6), (0, 8), (1, 7), (1, 17), (1, 30), (2, 3), (2, 32), (29, 33)]
    if mode != 1:
        # deleting edges
        g_.remove_edges_from(del_edges)
    else:
        # for displaying removed edges, so not really delete
        for i, j in del_edges:
            g_.edges[i, j]['color'] = ALL_COLORs['removed']
    draw_karate(g_, true_labels, fig_title='Karate network with 10% missing edges')

    # Louvain
    if mode != 1:
        while True:
            louvain_res = stg.train_byLouvain(data=res)
            louvain_labels = louvain_res[-1]
            if 0.8 < louvain_res[0] < 1:
                break
    else:
        louvain_labels = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2]
    g_lou = g_.copy()
    draw_karate(g_lou, louvain_labels, fig_title='Louvain on incomplete karate network')

    # CLMC on 10% edges removed network
    clmc_labels = [1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    g_clmc = g_.copy()
    if mode == 1:
        # link prediction
        # predicted true edges: (1, 7), (29, 33), (1, 17)
        # predicted false edges: (10, 16)
        g_clmc.edges[1, 7]['color'] = ALL_COLORs['predict']
        g_clmc.edges[29, 33]['color'] = ALL_COLORs['predict']
        g_clmc.edges[1, 17]['color'] = ALL_COLORs['predict']
        g_clmc.add_edge(10, 16)
        g_clmc.edges[10, 16]['color'] = ALL_COLORs['predict']
    draw_karate(g_clmc, clmc_labels, fig_title='CLMC on incomplete karate network')

    # MNDPEM
    if mode != 1:
        Z_Edge_ori, Z_noEdge_ori = tools.get_Z_init(g_)
        N = g_.number_of_nodes()
        C = 2
        res['observe_graph'] = g_
        res['Z_noEdge_ori'] = Z_noEdge_ori
        res['Z_Edge_ori'] = Z_Edge_ori
        res['num_del_edges'] = 16
        res['N'] = N
        res['C'] = C

        res_ = stg.train_byMNDPEM(data=res)
        g_mndpem = res_['graph_res']
        mndpem_labels = res_['F_argmax']
        print(f'different [ MNDPEM - CLMC ] : {set(g_mndpem.edges) - set(g_clmc.edges)}')
    else:
        tmp_add_edges = [(0, 33), (2, 3), (13, 12)]
        g_mndpem = g_.copy()
        g_mndpem.add_edges_from(tmp_add_edges)
        for i, j in tmp_add_edges:
            g_mndpem.edges[i, j]['color'] = ALL_COLORs['predict']
        mndpem_labels = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                         2, 2]
    draw_karate(g_mndpem, mndpem_labels, fig_title='Our method on incomplete karate network')

    # MNDP
    # while True:
    #     mndp_res = stg.train_byMNDP_Missing(data=res)
    #     mndp_labels = mndp_res[-1]
    #     if mndp_res[0] < 1:
    #         break
    # g_mndp = g_.copy()


def display_case_study():
    from util import base_method as tools
    from model.Strategy import Strategy

    stg = Strategy()
    ALL_COLORs = {
        'exist': '#5E5E5E',
        'removed': '#D4D4D4',
        'predict': '#CD00CD'}
    data = stg.prepare_data(Read.read_polbooks)
    g_ori = data['graph_real']
    observe_graph = data['observe_graph']
    del_edges = data['del_edges']
    res = stg.train_byMNDPEM(data)
    F_argmax = res['F_argmax']
    g_res = res['graph_res']

    for i, j in observe_graph.edges:
        observe_graph.edges[i, j]['color'] = ALL_COLORs['exist']

    observe_graph.add_edges_from(del_edges)
    for i, j in del_edges:
        observe_graph.edges[i, j]['color'] = ALL_COLORs['removed']

    predicted_edges = list(set(g_res.edges) - set(observe_graph.edges))
    observe_graph.add_edges_from(predicted_edges)
    for i, j in predicted_edges:
        observe_graph.edges[i, j]['color'] = ALL_COLORs['predict']

    display_polbooks(observe_graph, F_argmax, fig_title='Our model on Polbook')


def _analyze_tmpTxt():
    import re

    res = []
    with open('tmp.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            s = line.strip().split(' ')[0]
            s = re.findall(r"\d+", s)
            res.append((int(s[1]) - 1, int(s[0]) - 1))
    print(res)


if __name__ == '__main__':
    # display_rate_polblogs()
    # display_intro_case()
    display_case_study()