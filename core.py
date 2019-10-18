import math
import networkx as nx
import numpy as np
from functools import reduce

import keras
from keras import Model, backend as K, regularizers
from keras.layers import Dense, Embedding, Input, Reshape, Subtract, Lambda
from keras.utils.training_utils import multi_gpu_model
from keras.activations import elu

def build_reconstruction_loss(beta):
    """
    return the loss function for 2nd order proximity
    beta: the definition below Equation 3
    """
    assert beta > 1

    def reconstruction_loss(true_y, pred_y):
        diff = K.square(true_y - pred_y)

        # borrowed from https://github.com/suanrong/SDNE/blob/master/model/sdne.py#L93
        weight = true_y * (beta - 1) + 1

        weighted_diff = diff * weight
        return K.mean(K.sum(weighted_diff, axis=1))  # mean sqaure error
    return reconstruction_loss

def edge_wise_loss(true_y, embedding_diff):
    """
    1st order proximity
    """
    # true_y supposed to be None
    # we don't use it
    return K.mean(K.sum(K.square(embedding_diff), axis=1))  # mean sqaure error

class SDNE():
    def __init__(self,
                 graph,
                 encode_dim,
                 weight='weight',
                 encoding_layer_dims=[],
                 beta=2, alpha=2,
                 l2_param=0.01):
        """graph: nx.Graph
        encode_dim: int, length of inner most dim
        beta: beta parameter under Equation 3
        alpha: weight of loss function on self.edges
        """
        self.encode_dim = encode_dim

        ###################
        # GRAPH STUFF
        ###################

        self.graph = graph
        self.N = graph.number_of_nodes()
        self.adj_mat = nx.adjacency_matrix(self.graph).toarray()
        self.edges = np.array(list(self.graph.edges_iter()))

        # weights
        # default to 1
        weights = [graph[u][v].get(weight, 1.0)
                   for u, v in self.graph.edges_iter()]
        self.weights = np.array(weights, dtype=np.float32)[:, None]

        if len(self.weights) == self.weights.sum():
            print('the graph is unweighted')

        ####################
        # INPUT
        ####################

        # one end of an edge
        input_a = Input(shape=(1,), name='input-a', dtype='int32')
        # the other end of an edge
        input_b = Input(shape=(1,), name='input-b', dtype='int32')
        edge_weight = Input(shape=(1,), name='edge_weight', dtype='float32')

        ####################
        # network architecture
        ####################
        encoding_layers = []
        decoding_layers = []

        embedding_layer = Embedding(output_dim=self.N, input_dim=self.N,
                                    trainable=False, input_length=1, name='nbr-table')
        # if you don't do this, the next step won't work
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.adj_mat])

        encoding_layers.append(embedding_layer)
        encoding_layers.append(Reshape((self.N,)))

        # encoding
        # encoding_layer_dims = [encode_dim]
        encoding_layer_dims.append(encode_dim)
        print('@@@@@ELD:', encoding_layer_dims)
        for i, dim in enumerate(encoding_layer_dims):
            if i == len(encoding_layer_dims) - 1:
                activation = 'sigmoid'
            else:
                activation = elu
            layer = Dense(dim, activation=activation,
                          kernel_regularizer=regularizers.l2(l2_param),
                          name='encoding-layer-{}'.format(i))
            encoding_layers.append(layer)

        # decoding
        decoding_layer_dims = encoding_layer_dims[::-1][1:] + [self.N]

        for i, dim in enumerate(decoding_layer_dims):
            print('&&&&&', i, dim)
            if i == len(decoding_layer_dims) - 1:
                activation = 'sigmoid'
            else:
                activation = elu
            layer = Dense(
                dim, activation=activation,
                kernel_regularizer=regularizers.l2(l2_param),
                name='decoding-layer-{}'.format(i))
            decoding_layers.append(layer)

        all_layers = encoding_layers + decoding_layers

        ####################
        # VARIABLES
        ####################
        # ex) reduce(lambda x, y: x+y, [1,2,3,4,5]) -> ((((1+2)+3)+4)+5)

        encoded_a = reduce(lambda arg, f: f(arg), encoding_layers, input_a)
        encoded_b = reduce(lambda arg, f: f(arg), encoding_layers, input_b)

        decoded_a = reduce(lambda arg, f: f(arg), all_layers, input_a)
        decoded_b = reduce(lambda arg, f: f(arg), all_layers, input_b)

        embedding_diff = Subtract()([encoded_a, encoded_b])

        # add weight to diff
        embedding_diff = Lambda(lambda x: x * edge_weight)(embedding_diff)

        ####################
        # MODEL
        ####################
        self.model = Model([input_a, input_b, edge_weight],
                           [decoded_a, decoded_b, embedding_diff])
        self.model.summary()
        reconstruction_loss = build_reconstruction_loss(beta)
        # self.parallel_model = multi_gpu_model(self.model, gpus=4)
        # self.parallel_model.compile(optimizer='adadelta',
        #                    loss=[reconstruction_loss, reconstruction_loss, edge_wise_loss],
        #                    loss_weights=[1, 1, alpha])
        self.model.compile(optimizer='adadelta',
                           loss=[reconstruction_loss, reconstruction_loss, edge_wise_loss],
                           loss_weights=[1, 1, alpha])

        self.encoder = Model(input_a, encoded_a)

        # for pre-training
        self.decoder = Model(input_a, decoded_a)
        # self.paralle_decoder = multi_gpu_model(self.decoer, gpus=4)
        # self.paralle_decoder.compile(optimizer='adadelta',
        #                      loss=reconstruction_loss)
        self.decoder.compile(optimizer='adadelta',
                             loss=reconstruction_loss)

    def pretrain(self, **kwargs):
        """pre-train the autoencoder without edges"""
        nodes = np.arange(self.graph.number_of_nodes())
        node_neighbors = self.adj_mat[nodes]
        # self.paralle_decoder.fit(nodes[:, None],
        #                  node_neighbors,
        #                  shuffle=True,
        #                  **kwargs)
        self.decoder.fit(nodes[:, None],
                         node_neighbors,
                         shuffle=True,
                         **kwargs)

    def train_data_generator(self, batch_size=32):
        # this can become quadratic if using dense
        m = self.graph.number_of_edges()
        while True:
            for i in range(math.ceil(m / batch_size)):
                sel = slice(i*batch_size, (i+1)*batch_size)
                nodes_a = self.edges[sel, 0][:, None]
                nodes_b = self.edges[sel, 1][:, None]
                weights = self.weights[sel]

                neighbors_a = self.adj_mat[nodes_a.flatten()]
                neighbors_b = self.adj_mat[nodes_b.flatten()]

                # requires to have the same shape as embedding_diff
                dummy_output = np.zeros((nodes_a.shape[0], self.encode_dim))

                yield ([nodes_a, nodes_b, weights],
                       [neighbors_a, neighbors_b, dummy_output])

    def fit(self, log=False, **kwargs):
        """kwargs: keyword arguments passed to `model.fit`"""
        if log:
            callbacks = [keras.callbacks.TensorBoard(
                log_dir='./log', histogram_freq=0,
                write_graph=True, write_images=False)]
        else:
            callbacks = []

        callbacks += kwargs.get('callbacks', [])
        if 'callbacks' in kwargs:
            del kwargs['callbacks']

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            del kwargs['batch_size']
            gen = self.train_data_generator(batch_size=batch_size)
        else:
            gen = self.train_data_generator()

        # self.parallel_model.fit_generator(
        #     gen,
        #     shuffle=True,
        #     callbacks=callbacks,
        #     pickle_safe=True,
        #     **kwargs)
        self.model.fit_generator(
            gen,
            shuffle=True,
            callbacks=callbacks,
            pickle_safe=True,
            **kwargs)

    def get_node_embedding(self):
        """return the node embeddings as 2D array, #nodes x dimension"""
        nodes = np.array(self.graph.nodes())[:, None]
        return self.encoder.predict(nodes)

    def save(self, path):
        self.model.save(path)
