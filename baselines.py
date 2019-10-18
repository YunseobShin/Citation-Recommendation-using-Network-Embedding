import networkx as nx
import numpy as np
import pickle, random
from time_check import tic
from time_check import toc
from tqdm import tqdm
from link_pred_baselines import Link_prediction
import scipy as sp

def sample_inputs(number_of_inputs, size, adj, dataset):
    res = []
    while len(res) < number_of_inputs:
        sample_node = random.sample(list(range(size)), 1)[0]
        if sample_node in res:
            continue
        # cora is too sparse
        if dataset == 1:
            if sum(adj[sample_node]) == 0:
                continue
        else:
            if sum(adj[sample_node]) < 5:
                continue
        res.append(sample_node)
    return res

def save_results(dataset, n_inputs, fname, ndcg):
    if dataset == 1:
        ds = 'cora_'
    elif dataset == 2:
        ds = 'hepph_'
    elif dataset == 3:
        ds = 'hepth_'
    res = {}
    res['dataset'] = ds
    res['n_inputs'] = n_inputs
    res['ndcg'] = ndcg
    pickle.dump(res, open(fname, 'wb'))

if __name__ == '__main__':
    dataset = int(input('Dataset - [1]:cora [2]:Hep-Ph [3]:Hep-Th [-1]:EXIT -> '))
    number_of_inputs = int(input('number of inputs: '))
    number_of_given = int(input('given neighbors per each node: '))
    if dataset == 1:
        dsname = 'cora'
        aa = 'lp/adamic_cora.npy'
        jc = 'lp/jaccard_cora.npy'
        pa = 'lp/preferential_cora.npy'
        graph_path = 'citation_nets/cora/cora_sample_txt'
        adj_path = 'adj/cora_sample.npy'

    elif dataset == 2:
        dsname = 'hepph'
        is_sample = int(input('Use sample? [1]:yes [2]:no -> '))
        if is_sample == 1:
            graph_path = 'citation_nets/hep-ph/HepPh_sample_txt'
            adj_path = 'adj/hepph_adj.npy'
            aa = 'lp/adamic_sample_hepph.npy'
            jc = 'lp/jaccard_sample_hepph.npy'
            pa = 'lp/preferential_sample_hepph.npy'
        else:
            graph_path = 'citation_nets/hep-ph/HepPh_converted_txt'
            aa = 'lp/adamic_hepph.npy'
            jc = 'lp/jaccard_hepph.npy'
            pa = 'lp/preferential_hepph.npy'
            adj_path = 'adj/hepph_adj.npy'

    elif dataset == 3:
        dsname = 'hepth'
        aa = 'lp/adamic_hepth.npy'
        jc = 'lp/jaccard_hepth.npy'
        pa = 'lp/preferential_hepth.npy'
        graph_path = 'citation_nets/hep-th/HepTh_sample_txt'
        adj_path = 'adj/hepth_sample.npy'

    graph = nx.read_edgelist(graph_path, create_using=nx.Graph())
    graph = nx.convert_node_labels_to_integers(graph)
    adj_mat = nx.adjacency_matrix(graph).toarray()
    print('#nodes:', graph.number_of_nodes())
    print('#edges:', graph.number_of_edges())
    adamic = np.load(aa)
    jaccard = np.load(jc)
    preferential = np.load(pa)
    inputs = sample_inputs(number_of_inputs, graph.number_of_nodes(), adj_mat, dataset)
    print('INPUTS:', inputs)
    k=30
    # inputs = [1,2,4]
    ks = [5, 10, 20, 30, 50]
    rec = Link_prediction(inputs = inputs, graph = graph, adj_mat=adj_mat, number_of_given= number_of_given,
                          adamic = adamic, jaccard = jaccard, preferential = preferential,
                          max_recs=ks[-1])
    print('recommending...')
    tic()
    # recommendations = rec.recommend_by_embedding()
    print('===adamic adar==================================================================')
    recommendations = rec.pred_by_adamic_adar()
    for k in ks:
        precision = rec.precision_at_k(recommendations, k=k)
        print('Precision at ', k, ':', precision)
    for k in ks:
        recall = rec.recall_at_k(recommendations, k=k)
        print('Recall at ', k, ':', recall)

    print('calcuating NDCG...')
    avg_ccp, p_max = rec.get_avg_ccp()
    rel = rec.get_rel_rank(avg_ccp, p_max)

    ndcg_fn = 'results/ndcg_adamic_adar' +dsname+ '.txt'
    f = open(ndcg_fn, 'w')
    for k in ks:
        ndcg_at_k = rec.ndcg_at_k(recommendations, k, avg_ccp, p_max, rel)
        print('NDCG at {}: {}'.format(k, ndcg_at_k))
        f.write(str(k) + str(ndcg_at_k))
    f.close()
    fname = 'results/baselines/adamic_adar.pkl'
    save_results(dataset, number_of_inputs, fname, ndcg_at_k)

    print('===jaccard coefficient==========================================================')
    recommendations = rec.pred_by_jaccard_coefficient()
    for k in ks:
        precision = rec.precision_at_k(recommendations, k=k)
        print('Precision at ', k, ':', precision)
    for k in ks:
        recall = rec.recall_at_k(recommendations, k=k)
        print('Recall at ', k, ':', recall)
    print('calcuating NDCG...')

    ndcg_fn = 'results/ndcg_jaccard_coefficient' +dsname+ '.txt'
    f = open(ndcg_fn, 'w')
    for k in ks:
        ndcg_at_k = rec.ndcg_at_k(recommendations, k, avg_ccp, p_max, rel)
        print('NDCG at {}: {}'.format(k, ndcg_at_k))
        f.write(str(k) + str(ndcg_at_k))
    f.close()
    fname = 'results/baselines/jaccard_coefficient.pkl'
    save_results(dataset, number_of_inputs, fname, ndcg_at_k)

    print('===preferential attachment======================================================')
    recommendations = rec.pred_by_preferential_attachment()
    for k in ks:
        precision = rec.precision_at_k(recommendations, k=k)
        print('Precision at ', k, ':', precision)
    for k in ks:
        recall = rec.recall_at_k(recommendations, k=k)
        print('Recall at ', k, ':', recall)
    print('calcuating NDCG...')

    ndcg_fn = 'results/ndcg_preferential_attachment' +dsname+ '.txt'
    f = open(ndcg_fn, 'w')
    for k in ks:
        ndcg_at_k = rec.ndcg_at_k(recommendations, k, avg_ccp, p_max, rel)
        print('NDCG at {}: {}'.format(k, ndcg_at_k))
        f.write(str(k) + str(ndcg_at_k))
    f.close()
    fname = 'results/baselines/preferential_attachment.pkl'
    save_results(dataset, number_of_inputs, fname, ndcg_at_k)
    # print('===second neighbors===========================================================')
    # recommendations = rec.pred_by_neighbors()
    # print('calcuating NDCG...')
    # ndcg_fn = 'results/ndcg_second_neighbors' +dsname+ '.txt'
    # f = open(ndcg_fn, 'w')
    # for k in ks:
    #     ndcg_at_k = rec.ndcg_at_k(recommendations = recommendations, k=k)
    #     print('NDCG at {}: {}'.format(k, ndcg_at_k))
    #     f.write(str(k) + str(ndcg_at_k))
    # f.close()
    # fname = 'results/baselines/second_neighbors.pkl'
    # save_results(dataset, number_of_inputs, fname, ndcg_at_k)







    toc()
