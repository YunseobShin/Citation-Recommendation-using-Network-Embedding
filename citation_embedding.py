# coding: utf-8
import math
import networkx as nx
import pickle as pkl

from itertools import product
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from tqdm import tqdm
from core import SDNE
import numpy as np
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config=config)

while True:
    dataset = int(input('Dataset - [1]:cora [2]:citeseer [3]:patent [4]:Hep-Ph [5]:Hep-Th [-1]:EXIT -> '))
    is_sample = int(input('Use sample? [1]:yes [2]:no -> '))
    if dataset == 1:
        if is_sample == 1:
            dsname = 'cora_sample' # 1,000 nodes
            data_path = 'citation_nets/cora/cora_sample_txt'
            batch_size = 256
            encoding_layer_dims=[500]
        else:
            dsname = 'cora' # 23,166 nodes
            data_path = 'data/citation_nets/cora/out.subelj_cora_cora'
            batch_size = 16
            encoding_layer_dims=[1000]
        break
    elif dataset ==2:
        dsname = 'citeseer' # 384,413 nodes
        data_path = 'data/citation_nets/citeseer/out.citeseer'
        batch_size = 2
        encoding_layer_dims=[5000]
        break
    elif dataset == 3:
        dsname = 'patent' # 3,774,768 nodes
        data_path = 'data/citation_nets/patentcite/out.patentcite'
        batch_size = 1
        encoding_layer_dims=[23166, 5000]
        break
    elif dataset == 4:
        if is_sample == 1:
            dsname = 'hepph_sample'
            data_path = 'citation_nets/hep-ph/HepPh_sample_txt'
            batch_size = 256
            encoding_layer_dims=[500]
        else:
            dsname = 'hepph' # 34,546 nodes
            data_path = 'data/citation_nets/hep-ph/HepPh_converted_txt'
            batch_size = 16
            encoding_layer_dims=[1000]
        break
    elif dataset == 5:
        if is_sample == 1:
            dsname = 'hepth_sample'
            data_path = 'citation_nets/hep-th/HepTh_sample_txt'
            batch_size = 256
            encoding_layer_dims=[500]
        else:
            dsname = 'hepth'
            data_path = 'data/citation_nets/hep-th/HepTh_converted_txt'
            batch_size = 16
            encoding_layer_dims=[1000]
        break
    elif dataset == -1:
        exit()
    else:
        print('invalid dataset')
if is_sample == 1:
    epochs = 50
else:
    epochs = 5
g = nx.read_edgelist(data_path, create_using=nx.DiGraph())
g = nx.convert_node_labels_to_integers(g)

parameter_grid = {'alpha': [2],
                  'l2_param': [1e-3],
                  'pretrain_epochs': [1],
                  'epochs': [epochs]}

parameter_values = list(product(*parameter_grid.values()))
parameter_keys = list(parameter_grid.keys())
parameter_dicts = [dict(list(zip(parameter_keys, values))) for values in parameter_values]

def one_run(params, batch_size, dsname, encoding_layer_dims):
    alpha = params['alpha']
    l2_param = params['l2_param']
    pretrain_epochs = params['pretrain_epochs']
    epochs = params['epochs']
    beta = 2
    model = SDNE(g, encode_dim=100, encoding_layer_dims=encoding_layer_dims,
                 beta=beta,
                 alpha=alpha,
                 l2_param=l2_param)


    print('######Pre-training...######')
    model.pretrain(epochs=pretrain_epochs, batch_size=batch_size)
    print('done')
    n_batches = math.ceil(g.number_of_edges() / batch_size)

    print('######training...######')
    model.fit(epochs=epochs, log=True, batch_size=batch_size,
              steps_per_epoch=n_batches)


    print('done')
    embedding_path = 'embeddings/' + dsname + '/' + dsname + '_sdne_a{}b{}e{}.pkl'.format(
        alpha, beta, epochs
    )
    embeddings = model.get_node_embedding()
    pkl.dump(embeddings, open(embedding_path, 'wb'))

for params in tqdm(parameter_dicts):
    one_run(params, batch_size, dsname, encoding_layer_dims)
