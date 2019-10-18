import pickle as pkl
import networkx as nx
while True:
    dataset = int(input('Dataset - [1]:cora [2]:citeseer [3]:patent [4]:Hep-Ph [5]:Hep-Th [-1]:EXIT -> '))
    if dataset == 1:
        data_path = 'data/citation_nets/cora/out.subelj_cora_cora'
        output_name = 'data/citation_nets/cora/cora_converted'
        break
    elif dataset ==2:
        data_path = 'data/citation_nets/citeseer/out.citeseer'
        output_name = 'data/citation_nets/citeseer/citeseer_converted'
        break
    elif dataset == 3:
        data_path = 'data/citation_nets/patentcite/out.patentcite'
        output_name = 'data/citation_nets/patentcite/patentcite_converted'
        break
    elif dataset == 4:
        data_path = 'data/citation_nets/hep-ph/cit-HepPh.txt'
        output_name = 'data/citation_nets/hep-ph/HepPh_converted'
        break
    elif dataset == 5:
        data_path = 'data/citation_nets/hep-th/out.cit-HepTh'
        output_name = 'data/citation_nets/hep-th/HepTh_converted'
        break
    elif dataset == -1:
        exit()
    else:
        print('invalid dataset')

g = nx.read_edgelist(data_path, create_using=nx.DiGraph())
g = nx.convert_node_labels_to_integers(g)
gstr = str(g.edges()).replace('), (', '\n')
gstr = gstr.replace('(', '')
gstr = gstr.replace(')', '')
gstr = gstr.replace(']', '')
gstr = gstr.replace('[', '')
gstr = gstr.replace(',', '')

fn = output_name + '_txt'
with open(fn, 'w') as f:
    f.write(gstr)
pkl.dump(g.edges(), open(output_name, 'wb'))
