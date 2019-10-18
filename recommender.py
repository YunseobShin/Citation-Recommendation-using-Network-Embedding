import math, random
import networkx as nx
import numpy as np
from scipy import spatial
import sys
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from time_check import tic
from time_check import toc
sys.setrecursionlimit(10000) # 10000 is an example, try with different values

def get_answer(inputs, adj_mat, number_of_given, number_of_training):
    adj_mat = adj_mat[:,:number_of_training]
    answer_dic = {}
    given = {}
    to_pred = {}
    for inp in inputs:
        # print(inp)
        if number_of_given > sum((adj_mat[inp])):
            number_of_given = sum((adj_mat[inp]))
        neighbors = np.where(adj_mat[inp])[0]
        print(neighbors)
        answer_dic[inp] = neighbors
        given[inp] = neighbors[0:number_of_given]
        to_pred[inp] = neighbors[number_of_given:]
        print('#neighbors:', len(answer_dic[inp]))
        print('#given:', len(given[inp]))
        print('#true:', len(to_pred[inp]))
    return answer_dic, given, to_pred

def nn_from_centroid(embeddings, inputs, given_neighbors, tree, max_recs):
    recommendations = {}
    for inp in tqdm(inputs):
        # print(embeddings[given_neighbors[inp]])
        centroid = np.mean(embeddings[given_neighbors[inp]], axis=0)
        recommendation = tree.query(centroid, max_recs)[1]
        recommendations[inp] = recommendation
    # print(recommendations)
    return recommendations

def nn_from_each_one(embeddings, inputs, given_neighbors, tree, max_recs):
    recommendations = {}
    for inp in tqdm(inputs):
        recommendation = []
        if len(given_neighbors[inp]) > 0:
            n = math.ceil(max_recs/len(given_neighbors[inp]))
            rec = {}
            for neighbor in given_neighbors[inp]:
                input_vec = embeddings[neighbor]
                rec[neighbor] = tree.query(input_vec, n)[1]
                rec[neighbor] = rec[neighbor][np.where(rec[neighbor] < len(embeddings))]
                # print(rec[neighbor])

            for i in range(n):
                for neighbor in rec:
                    # print('i:',i,'element:',rec[neighbor][i])
                    recommendation.append(rec[neighbor][i])
            recommendations[inp] = recommendation
        else:
            recommendations[inp] = []
    return recommendations

def avg_dis_from_centroid(centroid, preds, embeddings):
    dis = 0
    for pred in preds:
        dis += np.linalg.norm(centroid-embeddings[pred])
    return dis / len(preds)

def get_centroid(embeddings, given, inp):
    return np.mean(embeddings[given[inp]], axis=0)

def take_second(e):
    return e[1]

def get_co_citing_prob(i, j, adj):
    p_cited_i = adj[i,:]
    p_cited_j = adj[j,:]
    s = p_cited_j + p_cited_i
    numer = len(np.where(s==2)[0])
    deno = len(np.where(s>=1)[0])
    # print(i, j, '/', numer, deno)
    if deno > 0:
        return float(numer)/float(deno)
    else:
        return 0

def get_rel_rank(cp, p_max):
    if cp <= 1 and cp > 3/4*p_max:
        return 4
    elif cp <= 3/4*p_max and cp > 1/2*p_max:
        return 3
    elif cp <= 1/2*p_max and cp > 1/4*p_max:
        return 2
    elif cp <= 1/4*p_max and cp > 0:
        return 1
    else:
        return 0

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

class Recommender():
    def __init__(self, embeddings, inputs,
                 graph, number_of_given, params,
                 adj_mat, origin_adj, testing_set, max_recs=50):
        self.number_of_inputs = len(inputs)
        self.embeddings = embeddings
        self.inputs = inputs
        print('IMPUT:', inputs)
        self.number_of_given = number_of_given
        self.tree = spatial.KDTree(self.embeddings)
        self.adj_mat = adj_mat
        self.origin_adj = origin_adj
        self.testing_set = testing_set
        self.pr = nx.pagerank(graph, alpha=0.9)
        self.pr_sorted = sorted([[k,v] for k,v in self.pr.items()], reverse=True, key=take_second)
        self.pr_sorted = [int(x) for x in np.array(self.pr_sorted)[:,0]]
        self.answer_dic, self.given_neighbors, self.true_neighbors = get_answer(inputs, self.origin_adj, number_of_given, len(embeddings))
        # print(self.true_neighbors)
        self.max_recs = max_recs
        self.params = params
        print(params)
        avd = []
        for inp in self.inputs:
            centroid = get_centroid(self.embeddings, self.given_neighbors, inp)
            dis = 0
            if len(self.given_neighbors[inp]) == 0:
                continue
            for t in self.given_neighbors[inp]:
                dis += np.linalg.norm(centroid - self.embeddings[t])
            avg_dis = dis/len(self.given_neighbors[inp])
            avd.append(avg_dis)
        self.ad = np.mean(avd)
        print('ave distance from centorid:', self.ad)

    def recommend_by_pagerank(self):
        recommendations = {}
        """
        recommend: doughnut shape,
                   similar distance from centorid with given neighbors
        priority:
            high PageRank & doughnut space > doughnut space > recommend
        high PageRank: top 10%
        """
        aug = 10
        radius = self.ad
        gamma = radius / 3
        n_nodes = len(self.embeddings)
        top_pgs = self.pr_sorted[:math.ceil(n_nodes*1/10)]
        for inp in tqdm(self.inputs):
            centroid = get_centroid(self.embeddings, self.given_neighbors, inp)
            # print(centroid)
            recs = self.tree.query(centroid, self.max_recs * aug)[1]
            recs = recs[np.where(recs < len(self.embeddings))]
            recs = [int(x) for x in recs]
            # print(recs)

            # Priority 1
            recommendations[inp] = []
            for rec in recs:
                dis = np.linalg.norm(centroid - self.embeddings[rec])
                if dis >= radius - gamma and dis <= radius + gamma and rec in top_pgs:
                    recommendations[inp].append(rec)
                    if len(recommendations[inp]) == self.max_recs:
                            break
            # print('rec by pr:', len(recommendations[inp]))
            # Priority 2
            tmp = len(recommendations[inp])
            for rec in recs:
                dis = np.linalg.norm(centroid - self.embeddings[rec])
                if dis >= radius - gamma and dis <= radius + gamma and rec not in top_pgs:
                    recommendations[inp].append(rec)
                    if len(recommendations[inp]) == self.max_recs:
                        break
            # print('rec by distance recs:', len(recommendations[inp]) - tmp)
            # Priority 3
            if len(recommendations[inp]) < self.max_recs:
                for rec in recs:
                    if rec not in recommendations[inp]:
                        recommendations[inp].append(rec)
                    if len(recommendations[inp]) == self.max_recs:
                            break
        return recommendations

    def recommend_by_embedding(self, method=2):
        if method == 1:
            return nn_from_centroid(self.embeddings, self.inputs,
                                               self.given_neighbors,
                                               self.tree, self.max_recs)
        else:
            return nn_from_each_one(self.embeddings, self.inputs,
                                               self.given_neighbors,
                                               self.tree, self.max_recs)
    # baseline
    def pred_by_neighbors(self):
        recommendations = {}
        for inp in tqdm(self.inputs):
            second_neighbors = []
            for neighbor in self.given_neighbors[inp]:
                second_neighbors += list(np.where(self.adj_mat[neighbor, :])[0])
                # second_neighbors += list(np.where(self.adj_mat[:, neighbor])[0])
            random.shuffle(second_neighbors)
            recommendations[inp] = second_neighbors[:self.max_recs]
        return recommendations

    def get_avg_ccp(self):
        avg_ccp = {}
        p_max = {}
        for inp in tqdm(self.inputs):
            ccps = []
            for j in range(len(self.embeddings)):
                ccp = []
                for neis in self.answer_dic[inp]:
                    ccp.append(get_co_citing_prob(neis, j, self.origin_adj))
                ccps.append(np.mean(ccp))

            avg_ccp[inp] = ccps
            p_max[inp] = np.max(ccps)
        return avg_ccp, p_max

    def get_rel_rank(self, avg_ccp, p_max):
        rels={}
        for inp in self.inputs:
            rels[inp] = []
            # avg_ccp: 1xd' (d': number of nodes in training set)
            for cp in avg_ccp[inp]:
                if cp <= 1 and cp > 3/4*p_max[inp]:
                    rels[inp].append(4)
                elif cp <= 3/4*p_max[inp] and cp > 1/2*p_max[inp]:
                    rels[inp].append(3)
                elif cp <= 1/2*p_max[inp] and cp > 1/4*p_max[inp]:
                    rels[inp].append(2)
                elif cp <= 1/4*p_max[inp] and cp > 0:
                    rels[inp].append(1)
                else:
                    rels[inp].append(0)
        return rels

    def ndcg_at_k(self, recommendations, k, cps, p_max, rels, method=0):
        ndcgs = []
        for inp in self.inputs:
            r = []
            recs = [int(x) for x in recommendations[inp]][:k]
            rel = np.array(rels[inp])
            rel_recs = rel[recs]
            rel_recs = sorted(rel_recs, reverse=True)
            dcg_max = dcg_at_k(sorted(rels[inp], reverse=True), k, method)

            if not dcg_max:
                dcg_max = 10000
            ndcgs.append(dcg_at_k(rel_recs, k, method) / dcg_max)
        return np.mean(ndcgs)

    def precision_at_k(self, recommendations, k):
        ps = []
        for inp in self.inputs:
            inp = int(inp)
            if len(self.true_neighbors[inp])==0:
                continue
            recs = set([int(x) for x in recommendations[inp]][:k])
            true_y = set(list(self.true_neighbors[inp]))
            a = len(recs & true_y)
            ps.append(a / k)
        return np.mean(ps)

    def recall_at_k(self, recommendations, k):
        rc = []
        for inp in self.inputs:
            inp = int(inp)
            if len(self.true_neighbors[inp])==0:
               continue
            recs = set([int(x) for x in recommendations[inp]][:k])
            true_y = set(list(self.true_neighbors[inp]))
            a = len(recs & true_y)
            if len(true_y) == 0:
                continue
            rc.append(a / len(true_y))
        return np.mean(rc)










#
