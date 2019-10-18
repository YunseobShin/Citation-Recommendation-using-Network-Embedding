# coding: utf-8
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import time
import sys
from time_check import tic
from time_check import toc

def train(input_file, output_file):
    if len(sys.argv) < 2:
        print('insufficient arguments')
        exit()
    else:
        print('input file:', sys.argv[1], '\noutput file:', sys.argv[2])
    # Create a graph
    print('reading edges...')
    graph = nx.read_edgelist(input_file, create_using=nx.DiGraph())
    graph = nx.convert_node_labels_to_integers(graph)

    print('number of nodes: ', nx.number_of_nodes(graph))
    # H=nx.DiGraph(G)   # create a DiGraph using the connections from G
    # H.edges()
    # edgelist=[(0,1),(1,2),(2,3)]
    # H=nx.Graph(edgelist)

    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=100, workers=12)
    # Embed
    model = node2vec.fit(window=3, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings
    # Save embeddings for later use
    model.wv.save(output_file)
    # Save model for later use
    # model.save('ex_model')

if __name__ == '__main__':
    tic()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    train(input_file, output_file)
    toc()
