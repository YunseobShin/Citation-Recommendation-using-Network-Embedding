import networkx as nx
import numpy as np

def take_second(e):
    return e[1]

dataset = int(input('Dataset - [1]:cora [2]:Hep-Ph [3]:Hep-Th [-1]:EXIT -> '))
if dataset == 1:
    graph_path = 'citation_nets/cora/cora_converted_txt'
    output_file = 'citation_nets/cora/cora_sample_txt'
elif dataset == 2:
    graph_path = 'citation_nets/hep-ph/HepPh_converted_txt'
    output_file = 'citation_nets/hep-ph/HepPh_sample_txt'
elif dataset == 3:
    graph_path = 'citation_nets/hep-th/HepTh_converted_txt'
    output_file = 'citation_nets/hep-th/HepTh_sample_txt'

sample_size = int(input('sample size: '))

g = nx.read_edgelist(graph_path, create_using=nx.DiGraph())
g = nx.convert_node_labels_to_integers(g)
PR = nx.pagerank(g, alpha=0.9)
pr_sorted = sorted([[k,v] for k,v in PR.items()], reverse=True, key=take_second)
pr_sorted = [int(x) for x in np.array(pr_sorted)[:,0]]
sample = pr_sorted[:sample_size]
gs = g.subgraph(sample)
print('#nodes:', len(sample))
print('#edges:', gs.number_of_edges())
gstr = str(gs.edges()).replace('), (', '\n')
gstr = gstr.replace('(', '')
gstr = gstr.replace(')', '')
gstr = gstr.replace(']', '')
gstr = gstr.replace('[', '')
gstr = gstr.replace(',', '')
with open(output_file, 'w') as f:
    f.write(gstr)
