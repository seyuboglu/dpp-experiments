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

from gcn.train import perform_train, LAYER_SPEC
from gcn.utils import format_data, sample_mask, inverse_sample_mask

def compute_gcn_scores(ppi_adj, training_nodes, params):
    ppi_adj_sparse = scipy.sparse.coo_matrix(ppi_adj)

def fetch_disease_data(current_disease):
    # Find the proteins that are known to be within this disease
    all_nodes = set(protein_to_node.values())
    in_disease_nodes = set([protein_to_node[protein] for protein in current_disease.proteins if protein in protein_to_node])
    out_disease_nodes = all_nodes - in_disease_nodes 
    in_disease_nodes, out_disease_nodes = list(in_disease_nodes), list(out_disease_nodes)

    # Determine the proportion of negative examples that we want to include
    # within our disease data set - we do this to keep the proportion relative
    # to the disease size since this can vary wildly
    shuffle(out_disease_nodes)
    quantity_negative = int(FRACTION_NEGATIVE*len(in_disease_nodes) / (1-FRACTION_NEGATIVE))
    out_disease_nodes = out_disease_nodes[:quantity_negative]

    n = len(graph)
    k = 2 # binary classification of in-disease

    # Now build up the desired prediction classes
    # [examples, classes]
    Y = np.zeros((n, k))

    # Set the values depending on in_disease vs. out_of_disease
    # Leave the "unknown quantities blank"
    Y[out_disease_nodes, 0] = 1
    Y[in_disease_nodes, 1] = 1

    # Full protein ids under consideration
    # Re-map the protein to node mappings with this
    # We use the labeled ids as our training/test set
    labeled_ids = list(set(in_disease_nodes) | set(out_disease_nodes))
    #shuffle(labeled_ids)

    training_quantity = int(PROPORTION_TRAIN*len(labeled_ids))
    test_quantity = int(PROPORTION_TEST*len(labeled_ids))

    # Select the nodes that we'll use for our training+test sets
    training_ids = labeled_ids[:training_quantity]
    test_ids = labeled_ids[training_quantity:training_quantity+test_quantity]
    return X, XS, Y, training_ids, test_ids


def train_on_disease(disease):
    X, XS, Y, training_ids, test_ids = fetch_disease_data(disease)
    metrics = initialize_metrics()
    fold_number = 1 
    #Initialize tensorflow session 
    sess = tf.Session()
    try: 
        data = format_data(X, XS, Y, graph, training_ids, validation_ids, test_ids)
        epoch_outputs, epoch_activations = perform_train(*data, sess = sess, verbose=False)
    except Exception as e: 
        print "Exception on Disease Pathway: ", disease.name
        print "Exception Message: ", str(e)
        print "Returning Empy Scores" 
        
    sess.close()
    tf.reset_default_graph()

    return scores