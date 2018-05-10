"""
Output pickle files based on our protein-protein interaction dataset

Create files that replicate the standardized train-test splits as follows

There are three different types of sets here
    Training (labeled) / Test (labeled) / Other (unlabeled)


x, the feature vectors of the labeled training instances,
    [examples, features]
y, the one-hot labels of the labeled training instances,
    [examples, hot-labels]
allx, the feature vectors of both labeled and unlabeled training instances (a superset of x),
    [all example, features].
graph, a dict in the format {index: [index_of_neighbor_nodes]}.

tx, the feature vectors of the test instances,
    [examples, features]
ty, the one-hot labels of the test instances,
    [examples, hot-labels]
test.index, the indices of test instances in graph, for the inductive setting,
    separate lines of test examples
ally, the labels for instances in allx.
    [all examples, hot-labels]
"""
from __future__ import division
import csv
import os
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import tensorflow as tf 

from gcn.train import perform_train
from gcn.utils import format_data, sample_mask, inverse_sample_mask, get_negatives

def compute_gcn_scores(ppi_adj, train_pos, val_pos, params):
    # Adjacency: Get sparse representation of ppi_adj
    n = ppi_adj.shape[0]
    ppi_adj = scipy.sparse.coo_matrix(ppi_adj)

    # X: Use identity for input features 
    X = np.identity(n)

    # Y: Build 
    Y = np.zeros((n, 2))
    Y[train_pos, 1] = 1
    Y[val_pos, 1] = 1
    train_neg = get_negatives(Y, len(train_pos))
    Y[train_neg, 0] = 1

    # Create index arrays
    train_nodes = np.concatenate((train_pos, train_neg))
    val_nodes = val_pos

    # Initialize tensorflow session 
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    scores = None
    
    try: 
        data = format_data(X, Y, ppi_adj, train_nodes, val_nodes)
        epoch_outputs = perform_train(*data, params = params, sess = sess, verbose=True)
        scores = epoch_outputs[-1][:,1]
    except Exception as e:
        print "Exception on GCN Execution:", str(e)

    sess.close()
    tf.reset_default_graph()

    return scores