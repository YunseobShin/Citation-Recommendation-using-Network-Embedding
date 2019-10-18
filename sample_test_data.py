import random
import networkx as nx
import numpy as np

dataset = int(input('Dataset - [1]:cora [2]:Hep-Ph [3]:Hep-Th [-1]:EXIT -> '))
if dataset == 1:
    dsname = 'cora'
    graph_path = 'citation_nets/cora/cora_converted_txt'
    adj_path = 'adj/cora_adj.npy'
elif dataset == 2:
    dsname = 'hepph'
    graph_path = 'citation_nets/hep-ph/HepPh_converted_txt'
    adj_path = 'adj/hepph_adj.npy'
elif dataset == 3:
    dsname = 'hepth'
    graph_path = 'citation_nets/hep-th/HepTh_converted_txt'
    adj_path = 'adj/hepth_adj.npy'

origin_vs_split = {}

def preproces_graph(g):
    gstr = str(g.edges()).replace('), (', '\n')
    gstr = gstr.replace('(', '')
    gstr = gstr.replace(')', '')
    gstr = gstr.replace(']', '')
    gstr = gstr.replace('[', '')
    gstr = gstr.replace(',', '')
    return gstr

graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph())
graph = nx.convert_node_labels_to_integers(graph)
adj = np.load(adj_path)

test_size = int(graph.number_of_nodes() / 4)
training_size = graph.number_of_nodes() - test_size
training_set = graph.nodes()[:training_size]
testing_set =  graph.nodes()[training_size:]

g_train = graph.subgraph(training_set)
print('#nodes_train:', len(training_set))
print('#nodes_test:', len(testing_set))
train_fn ='split/' + dsname + '_train'
test_set_fn ='split/' + dsname + '_test_nodes'

f = open(train_fn, 'w')
f.write(preproces_graph(g_train))
np.save(test_set_fn, testing_set)

f.close()



















#
