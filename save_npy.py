import numpy as np
import time
from time_check import tic
from time_check import toc
import sys
from gensim.models import KeyedVectors as KV

def by_node2vec(wv):
    indices = wv.index2word
    dimension = len(wv['0'])
    embeddings = np.zeros((len(indices), dimension))
    for i in range(len(indices)):
        embeddings[int(indices[i])] = wv[indices[i]]
        # key = indices[i]
        # val = wv[key]

    output_file = input_file + '.npy'
    print(len(embeddings))
    np.save(output_file, embeddings)

    # Load
    # read_dictionary = np.load('my_file.npy').item()
    # print(read_dictionary['hello']) # displays "world"

# def by_sdne():


tic()
input_file = sys.argv[1]
wv = KV.load(input_file, mmap='r')
by_node2vec(wv)
