from model.Strategy import Strategy
import util.draw_graph as Draw
import Experiment_case_study.case_study as case_study_tools
import util.read_data as Read
import matplotlib.pyplot as plt
import networkx as nx
import model.conf as CONF
import conf.draw_graph_dolph as conf_draw


def draw_(g, del_edges, false_node: list, predict_edges_true: list, predict_edges_false: list, title: str):
    nx.draw_networkx(g, pos=conf_draw.POS, edge_color=CONF.LINK_COLORs['exist'], node_size=200)
    nx.draw_networkx_nodes(g, pos=conf_draw.POS, nodelist=conf_draw.NODE_SHAPE['o'],
                           node_shape='o',
                           node_color=CONF.NODE_LABELs[1],
                           node_size=230)
    nx.draw_networkx_nodes(g, pos=conf_draw.POS, nodelist=conf_draw.NODE_SHAPE['s'],
                           node_shape='s',
                           node_color=CONF.NODE_LABELs[1],
                           node_size=230)
    nx.draw_networkx_nodes(g, pos=conf_draw.POS,
                           nodelist=false_node, node_shape='o',
                           node_color=CONF.NODE_LABELs[2],
                           node_size=230)
    nx.draw_networkx_edges(g, pos=conf_draw.POS, edgelist=del_edges, style='dashed')
    nx.draw_networkx_edges(g, pos=conf_draw.POS, edgelist=predict_edges_true,
                           edge_color=CONF.LINK_COLORs['predict_true'])
    nx.draw_networkx_edges(g, pos=conf_draw.POS, edgelist=predict_edges_false,
                           edge_color=CONF.LINK_COLORs['predict_wrong'])
    plt.title(title)
    plt.show()


def change_labels(label):
    res = []
    for i in label:
        if i == 1:
            res.append(2)
        else:
            res.append(1)
    return res


def main_dolph(train_mode=False):
    del_edges = case_study_tools.read_del_edges_CLMC_dolph()
    data = Strategy.prepare_data(Read.read_dolphins, del_list=del_edges)
    g_real = data['graph_real']
    g_obs = data['observe_graph']
    labels = data['labels']
    labels = change_labels(labels)

    # Complete network
    # Draw.display_dolphins(g_real, labels,
    #                       save_path='./res/dolph_true.png',
    #                       show=True,
    #                       fig_title='ground true')

    # 20% edges removed
    # g_obs_painted = Strategy.paint_color(g_obs, g_obs, del_edges, label=labels)
    # Draw.display_dolphins(g_obs_painted, labels,
    #                       save_path='./res/dolph_0.2_missing.png',
    #                       fig_title='20% edges missing',
    #                       show=True)

    # MNDP on 20% edges removed network
    # node with wrong label: 28, 30, 39
    if train_mode:
        mndp_res = Strategy.train_byMNDP_Missing(data)
        mndp_labels = mndp_res['F_argmax']
    else:
        # mndp_labels = case_study_tools.read_mndp_dolph_labels()
        mndp_labels = [1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2,
                       2, 1, 1, 1, 1,
                       1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1]
    mndp_g = g_obs.copy()

    mndp_g_painted = Strategy.paint_color(g_obs, mndp_g, del_edges, label=mndp_labels)
    Draw.display_dolphins(mndp_g_painted, mndp_labels,
                          fig_title='MNDP on network with 20% missing edges',
                          show=True,
                          save_path='./res/mndp_dolph_0.2.png')

    # CLMC on 20% edges removed network
    # node with wrong label: 39, 30
    # add edges: (40, 0), (30, 19), (7, 27), (56, 22), (17, 27), (6, 51), (43, 38), (11, 33), (51, 23), (43, 53), (43, 20), (36, 37), (45, 43), (50, 42), (1, 54), (13, 57), (13, 27)
    clmc_g = g_obs.copy()
    clmc_add_edges = [(40, 0), (30, 19), (7, 27), (56, 22), (17, 27), (6, 5), (6, 56), (43, 38), (11, 33), (51, 23),
                      (43, 53),
                      (43, 20), (36, 37), (45, 43), (50, 42), (1, 54), (13, 57), (13, 27)]
    clmc_g.add_edges_from(clmc_add_edges)
    clmc_labels = case_study_tools.read_cluster_CLMC_dolph()
    # print(f'clmc edges false: {sorted(set(clmc_g.edges) - set(g_real.edges))}')
    # print(f'clmc edges true: {sorted(set(clmc_g.edges) - set(g_obs.edges) - (set(clmc_g.edges) - set(g_real.edges)))}')
    clmc_g_painted = Strategy.paint_color(g_obs, clmc_g, del_edges, label=clmc_labels)
    print(f'30: {clmc_labels[29]}, 1: {clmc_labels[0]}')
    Draw.display_dolphins(clmc_g_painted, clmc_labels,
                          fig_title='CLMC on network with 20% missing edges',
                          show=True,
                          save_path='./res/clmc_dolph_0.2.png')

    # our model on 20% edges removed network
    # node with wrong label: 30
    # add edges: (1, 39), (36, 28), (28, 39), (30, 19), (1, 54), (7, 27), (54, 25), (26, 17), (26, 57), (1, 56), (35, 14), (23, 10), (18, 2), (43, 38),  (2, 59), (29, 59), (8, 36)
    if train_mode:
        our_res = Strategy.train_byMNDPEM(data, num_EM_iter=20, alpha=0.5)
        our_g = our_res['graph_res']
        our_labels = our_res['F_argmax']
        print(f'our labels: {our_labels}')
    else:
        our_add_edges = [(36, 28), (28, 39), (30, 19), (1, 54), (7, 27), (54, 25), (26, 17), (26, 57), (1, 56),
                         (35, 14), (23, 10), (18, 2), (43, 38), (2, 59), (29, 59), (8, 36)]
        our_g = g_obs.copy()
        our_g.add_edges_from(our_add_edges)
        our_labels = case_study_tools.read_our_dolph_labels()
        # our_labels = [1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2,
        #               1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
        #               2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1]
        # print(f'our edges false: {sorted(set(our_g.edges) - set(g_real.edges))}')
        # print(f'our edges true: {sorted(set(our_g.edges) - set(g_obs.edges) - (set(our_g.edges) - set(g_real.edges)))}')
    our_g_painted = Strategy.paint_color(g_obs, our_g, del_edges, label=our_labels)
    Draw.display_dolphins(our_g_painted, our_labels,
                          fig_title='our model on network with 20% missing edges',
                          show=True,
                          save_path='./res/our_model_dolph_0.2.png')


def main_display_result():
    del_edges = case_study_tools.read_del_edges_CLMC_dolph()
    data = Strategy.prepare_data(Read.read_dolphins, del_list=del_edges)
    g_real = data['graph_real']
    g_obs = data['observe_graph']
    labels = data['labels']
    labels = change_labels(labels)

    # MNDP
    plt.figure(1, figsize=(8, 10))
    draw_(g_obs, del_edges,
          false_node=conf_draw.MNDP_PREDICTED_LABEL['false'],
          predict_edges_false=conf_draw.MNDP_PREDICTED_EDGES['false'],
          predict_edges_true=conf_draw.MNDP_PREDICTED_EDGES['true'],
          title='MNDP on network with 20% edges removed')

    # CLMC
    plt.figure(2, figsize=(8, 10))
    draw_(g_obs, del_edges,
          false_node=conf_draw.CLMC_PREDICTED_LABEL['false'],
          predict_edges_false=conf_draw.CLMC_PREDICTED_EDGES['false'],
          predict_edges_true=conf_draw.CLMC_PREDICTED_EDGES['true'],
          title='CLMC on network with 20% edges removed')

    # Our model
    plt.figure(3, figsize=(8, 10))
    draw_(g_obs, del_edges,
          false_node=conf_draw.OUR_PREDICTED_LABEL['false'],
          predict_edges_false=conf_draw.OUR_PREDICTED_EDGES['false'],
          predict_edges_true=conf_draw.OUR_PREDICTED_EDGES['true'],
          title='Our model on network with 20% edges removed')


if __name__ == '__main__':
    # main_dolph(train_mode=False)
    main_display_result()
    # del_edges = case_study_tools.read_del_edges_CLMC_dolph()
    # print(del_edges)
