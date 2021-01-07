from model.Strategy import Strategy
import util.read_data as read
import util.draw_graph as draw
import networkx as nx
import conf.draw_graph_karate as conf_draw_karate
import model.conf as CONF
import matplotlib.pyplot as plt


def main_train(i: int):
    del_edges = [(0, 6), (0, 8), (1, 7), (1, 17), (1, 30), (2, 3), (2, 32), (29, 33)]
    data = Strategy.prepare_data(read.read_karate_club, missing_rate=0.1, del_list=del_edges)
    res = Strategy.train_byMNDPEM(data=data, num_EM_iter=13, alpha=0.12)
    observe_g = data['observe_graph']
    our_g = res['graph_res']
    our_labels = res['F_argmax']
    our_g_painted = Strategy.paint_color(
        g_observe_=observe_g,
        g_predicted=our_g,
        test_edges=del_edges,
        label=our_labels,
    )
    draw.draw_karate(g=our_g_painted, labels=our_labels,
                     fig_title='our model on network with 8 edges removed',
                     save_path=f'./res/our_{i}.png',
                     show=False)


def main_display_result():
    data = read.read_karate_club()
    g_ori = data['graph_real']
    true_labels = data['labels']

    del_edges = [(0, 6), (0, 8), (1, 7), (1, 17), (1, 30), (2, 3), (2, 32), (29, 33)]
    g_obs = g_ori.copy()
    g_obs.remove_edges_from(del_edges)

    # CLMC on 10% edges removed network
    # link prediction
    # predicted true edges: (1, 7), (29, 33), (1, 17)
    # predicted false edges: (10, 16)
    # node with wrong label: 2
    clmc_labels = [
        1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1,
        2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]
    clmc_predicted_true_edges = conf_draw_karate.CLMC_PREDICTED_EDGES['true']
    clmc_predicted_false_edges = conf_draw_karate.CLMC_PREDICTED_EDGES['false']
    clmc_g_res = g_obs.copy()
    clmc_g_res.add_edges_from(clmc_predicted_false_edges)
    plt.figure(1, figsize=(8, 10))
    nx.draw_networkx(clmc_g_res, pos=conf_draw_karate.POS, edge_color=CONF.LINK_COLORs['exist'])
    nx.draw_networkx_nodes(clmc_g_res, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['o'],
                           node_shape='o',
                           node_color=CONF.NODE_LABELs[1])
    nx.draw_networkx_nodes(clmc_g_res, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['s'],
                           node_shape='s',
                           node_color=CONF.NODE_LABELs[1])
    nx.draw_networkx_nodes(clmc_g_res, pos=conf_draw_karate.POS,
                           nodelist=conf_draw_karate.CLMC_PREDICTED_LABEL['false'], node_shape='o',
                           node_color=CONF.NODE_LABELs[2])
    nx.draw_networkx_edges(clmc_g_res, pos=conf_draw_karate.POS, edgelist=del_edges, style='dashed')
    nx.draw_networkx_edges(clmc_g_res, pos=conf_draw_karate.POS, edgelist=clmc_predicted_true_edges,
                           edge_color=CONF.LINK_COLORs['predict_true'])
    nx.draw_networkx_edges(clmc_g_res, pos=conf_draw_karate.POS, edgelist=clmc_predicted_false_edges,
                           edge_color=CONF.LINK_COLORs['predict_wrong'])
    plt.title('CLMC on network with 10% edges removed')
    plt.show()

    # MNDPEM
    # predicted true edges: (0, 33), (13, 12), (5, 3)
    # predicted false edges: (2, 3)
    # node with wrong label:
    our_predicted_true_edges = conf_draw_karate.OUR_PREDICTED_EDGES['true']
    our_predicted_false_edges = conf_draw_karate.OUR_PREDICTED_EDGES['false']
    em_g = g_obs.copy()
    em_g.add_edges_from(our_predicted_false_edges + our_predicted_true_edges)
    em_labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1,
        1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]

    plt.figure(2, figsize=(8, 10))
    nx.draw_networkx(em_g, pos=conf_draw_karate.POS, edge_color=CONF.LINK_COLORs['exist'])
    nx.draw_networkx_nodes(clmc_g_res, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['o'],
                           node_shape='o',
                           node_color=CONF.NODE_LABELs[1])
    nx.draw_networkx_nodes(em_g, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['s'],
                           node_shape='s',
                           node_color=CONF.NODE_LABELs[1])
    nx.draw_networkx_edges(em_g, pos=conf_draw_karate.POS, edgelist=del_edges, style='dashed')
    nx.draw_networkx_edges(em_g, pos=conf_draw_karate.POS, edgelist=our_predicted_true_edges,
                           edge_color=CONF.LINK_COLORs['predict_true'])
    nx.draw_networkx_edges(em_g, pos=conf_draw_karate.POS, edgelist=our_predicted_false_edges,
                           edge_color=CONF.LINK_COLORs['predict_wrong'])
    plt.title('Our model on network with 10% edges removed')
    plt.show()


if __name__ == '__main__':
    # for i in range(20):
    #     main_train(i)
    data = read.read_karate_club()
    labels = data['labels']
    g = data['graph_real']

    # nx.draw_networkx_nodes(g, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['s'])
    # nx.draw_networkx(g, pos=conf_draw_karate.POS)
    # nx.draw_networkx_nodes(g, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['o'], node_shape='o')
    # nx.draw_networkx_nodes(g, pos=conf_draw_karate.POS, nodelist=conf_draw_karate.NODE_SHAPE['s'], node_shape='s')
    #
    # pylab.show()
    main_display_result()
