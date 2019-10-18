import networkx as nx
import numpy as np
import pickle, random
from recommender import Recommender
from time_check import tic
from time_check import toc
from tqdm import tqdm
import scipy as sp

def get_embedding_path(dataset, method, is_sample):
    embedding_path = 'embeddings/dicts/'
    if dataset == 1:
        if is_sample == 1:
            if method == 'n':
                embedding_path += 'cora_n2v_sample.npy'
            elif method == 's':
                embedding_path += 'cora_sample_sdne_a2b2e50.pkl'
            else:
                return False
        else:
            if method == 'n':
                embedding_path += 'cora_n2v_train.npy'
            elif method == 's':
                embedding_path += 'cora_sdne_a2b10e5.pkl'
            else:
                return False
    elif dataset == 2:
        if is_sample == 1:
            if method == 'n':
                embedding_path += 'hepph_n2v_sample.npy'
            elif method == 's':
                embedding_path += 'hepph_sample_sdne_a2b2e50.pkl'
            else:
                return False
        else:
            if method == 'n':
                embedding_path += 'hepph_n2v_train.npy'
            elif method == 's':
                embedding_path += 'hepph_sdne_a2b4e5.pkl'
            else:
                return False
    elif dataset == 3:
        if is_sample == 1:
            if method == 'n':
                embedding_path += 'hepth_n2v_sample.npy'
            elif method == 's':
                embedding_path += 'hepth_sample_sdne_a2b2e50.pkl'
            else:
                return False
        else:
            if method == 'n':
                embedding_path += 'hepth_n2v_train.npy'
            elif method == 's':
                embedding_path += 'hepth_sdne_a2b2e5.pkl'
            else:
                return False
    else:
        return False
    return embedding_path

def get_embedding(embedding_path, method):
    if method == 'n':
        embeddings = np.load(embedding_path)
    elif method == 's':
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
    return embeddings

def sample_inputs(number_of_inputs, number_of_training, adj, n_givens, testing_set, dataset):
    res = []
    adj = adj[:,:number_of_training]
    while len(res) < number_of_inputs:
        sample_node = random.sample(list(testing_set), 1)[0]
        if sample_node in res:
            continue
        # cora is too sparse
        if dataset == 1:
            if sum(adj[sample_node]) == 0:
                continue
        else:
            if sum(adj[sample_node]) < n_givens:
                continue
        res.append(sample_node)
    return res

def save_results(dataset, method, n_inputs, n_givens, fname, ndcg):
    if dataset == 1:
        ds = 'cora_'
    elif dataset == 2:
        ds = 'hepph_'
    elif dataset == 3:
        ds = 'hepth_'
    if method == 'n':
        mt = 'node2vec_'
    elif method == 's':
        mt = 'SDNE_'
    res = {}
    res['dataset'] = ds
    res['method'] = mt
    res['n_inputs'] = n_inputs
    res['n_givens'] = n_givens
    res['ndcg'] = ndcg
    pickle.dump(res, open(fname, 'wb'))

if __name__ == '__main__':
    dataset = int(input('Dataset - [1]:cora [2]:Hep-Ph [3]:Hep-Th [-1]:EXIT -> '))
    is_sample = int(input('Use sample? [1]:yes [2]:no -> '))
    if dataset == 1:
        if is_sample == 1:
            dsname = 'cora_sample'
            graph_path = 'citation_nets/cora/cora_sample_txt'
            adj_path = 'adj/cora_sample.npy'
        else:
            dsname = 'cora'
            graph_path = 'split/cora_train'
            original = 'citation_nets/cora/cora_converted_txt'
            adj_path = 'split/cora_train_adj.npy'
            origin_adj_path = 'adj/cora_adj.npy'
            test_nodes = 'split/cora_test_nodes.npy'
    elif dataset == 2:
        if is_sample == 1:
            dsname = 'hepph_sample'
            graph_path = 'citation_nets/hep-ph/HepPh_sample_txt'
            adj_path = 'adj/hepph_sample.npy'
        else:
            dsname = 'hepph'
            graph_path = 'split/hepph_train'
            original = 'citation_nets/hep-ph/HepPh_converted_txt'
            adj_path = 'split/hepph_train_adj.npy'
            test_nodes = 'split/hepph_test_nodes.npy'
            origin_adj_path = 'adj/hepph_adj.npy'
    elif dataset == 3:
        if is_sample == 1:
            dsname = 'hepth_sample'
            graph_path = 'citation_nets/hep-th/HepTh_sample_txt'
            adj_path = 'adj/hepth_sample.npy'
        else:
            dsname = 'hepth'
            graph_path = 'split/hepth_train'
            original = 'citation_nets/hep-th/HepTh_converted_txt'
            adj_path = 'split/hepth_train_adj.npy'
            test_nodes = 'split/hepth_test_nodes.npy'
            origin_adj_path = 'adj/hepth_adj.npy'

    method = input('Embedding method - [n]:node2vec [s]:SDNE -> ')
    number_of_inputs = int(input('number of inputs: '))
    number_of_given = int(input('given neighbors per each node: '))
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph())
    graph = nx.convert_node_labels_to_integers(graph)
    train_adj = np.load(adj_path)
    origin_adj = np.load(origin_adj_path)
    print('#nodes:', graph.number_of_nodes())
    print('#edges:', graph.number_of_edges())
    testing_set = np.load(test_nodes)
    embedding_path = get_embedding_path(dataset, method, is_sample)
    if not embedding_path:
        print('invalid input')
        exit()
    embeddings = get_embedding(embedding_path, method)

    inputs = sample_inputs(number_of_inputs, len(embeddings), origin_adj, number_of_given, testing_set, dataset)

    params = 'd' + str(dataset) + 'm<' + str(method) + '>#inputs' + str(number_of_inputs) + '#given' + str(number_of_given)
    # inputs = [1,2,4]
    ks = [1, 5, 10, 20, 30, 50]
    rec = Recommender(embeddings = embeddings, inputs = inputs,
                      graph = graph, number_of_given = number_of_given,
                      params = params, adj_mat = train_adj, origin_adj = origin_adj,
                      testing_set = testing_set, max_recs=ks[-1])
    print('recommending...')
    tic()
    recommendations = rec.recommend_by_embedding(method = 2)
    # recommendations = rec.pred_by_neighbors()
    # recommendations = rec.recommend_by_pagerank()
    # print(recommendations)
    # recommendations = rec.true_neighbors
    
    for k in ks:
        precision = rec.precision_at_k(recommendations, k=k)
        print('Precision at ', k, ':', precision)
    for k in ks:
        recall = rec.recall_at_k(recommendations, k=k)
        print('Recall at ', k, ':', recall)

    print('calcuating NDCG...')
    avg_ccp, p_max = rec.get_avg_ccp()
    rel = rec.get_rel_rank(avg_ccp, p_max)
    ndcg_fn = 'results/ndcg_' + params+'.txt'
    f = open(ndcg_fn, 'w')
    for k in ks:
        ndcg_at_k = rec.ndcg_at_k(recommendations, k, avg_ccp, p_max, rel)
        print('NDCG at {}: {}'.format(k, ndcg_at_k))
        f.write(str(k)+'    '+str(ndcg_at_k))
    f.close()

    fname = 'results/' + params + '.pkl'
    save_results(dataset, method, number_of_inputs, number_of_given, fname, ndcg_at_k)












    toc()
