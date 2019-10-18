import matplotlib
matplotlib.use('Agg')
import networkx as nx
from matplotlib import pyplot as plt

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

graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph())
graph = nx.convert_node_labels_to_integers(graph)

n, k = graph.order(), graph.size()

avg_deg = float(k)/n

print('Nodes:', n)
print('Edges:', k)
print('Avg. Degree:', avg_deg)

in_degrees = graph.in_degree() # dictionary node:degree
in_values = sorted(set(in_degrees.values()))
in_hist = [list(in_degrees.values()).count(x) for x in in_values]

plt.figure()
if dataset == 1:
    fmt='ro'
elif dataset == 2:
    fmt = 'bo'
elif dataset == 3:
    fmt = 'yo'
plt.loglog(in_values,in_hist, fmt) # in-degree
plt.legend(['In-degree'])
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title(dsname + ' citation network')
plt.savefig(dsname+'_degree_distribution.eps')
plt.close()
