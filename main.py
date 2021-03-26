import graph_tool.all as gt
from util import base_method
from util import read_data
import numpy as np

g = gt.collection.data["karate"].copy()
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


gt.mcmc_equilibrate(state, force_niter=5000, mcmc_args=dict(niter=7), callback=collect_marginals)
eprob = u.ep.eprob

F_arg = state.get_block_state().get_bs()[0]

cc = "2 1 2 2 2 1 1 1 2 1 2 2 2 1 2 2 2 1 2 1 2 2 1 2 2 1 1 1 2 2 2 1 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2 2 2 2 2 1 2 1 1 2 2 1 2"
res = read_data.read_karate_club()
label_ = res['labels']
print(label_)

base_method.display_result(F_arg, label_)
