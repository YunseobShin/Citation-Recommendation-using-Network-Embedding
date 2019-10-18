import networkx as nx
import numpy as np
from time_check import tic
from time_check import toc
def take_third(e):
    return e[2]

dataset = int(input('Dataset - [1]:cora [2]:Hep-Ph [3]:Hep-Th [-1]:EXIT -> '))
if dataset == 1:
    dsname = 'cora'
    graph_path = 'citation_nets/cora/cora_converted_txt'
    # graph_path = 'citation_nets/cora/cora_sample_txt'
elif dataset == 2:
    dsname = 'hepph'
    graph_path = 'citation_nets/hep-ph/HepPh_converted_txt'
    # graph_path = 'citation_nets/hep-ph/HepPh_sample_txt'
elif dataset == 3:
    dsname = 'hepth'
    graph_path = 'citation_nets/hep-th/HepTh_converted_txt'
    # graph_path = 'citation_nets/hep-th/HepTh_sample_txt'

graph = nx.read_edgelist(graph_path, create_using=nx.Graph())
graph = nx.convert_node_labels_to_integers(graph)

tic()
aa = list(nx.adamic_adar_index(graph))
aa_pred = sorted(aa, reverse=True, key=take_third)
np.save('lp/adamic_sample_'+dsname, aa_pred)
toc()

jc = list(nx.jaccard_coefficient(graph))
jc_pred = sorted(jc, reverse=True, key=take_third)
np.save('lp/jaccard_sample_'+dsname, jc_pred)

pa = list(nx.preferential_attachment(graph))
pa_pred = sorted(pa, reverse=True, key=take_third)
np.save('lp/preferential_sample_'+dsname, pa_pred)
