"""
Logistic regression method for disease protein prediction
"""
from random import shuffle

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import get_negatives


def build_embedding_feature_matrix(protein_to_node, embedding_filename): 
    """ Builds a numpy matrix for a node embedding encoded in the embeddingf ile
    passed in. Row indices are given by protein_to_node dictionary passed in. 
    Args:
        protein_to_node (dictionary)
        embedding_filename (string)
    """
    with open(embedding_filename) as embedding_file:
        # get  
        line = embedding_file.readline()
        n_nodes, n_dim = map(int, line.split(" "))
        feature_matrix = np.empty((len(protein_to_node), n_dim))

        for index in range(n_nodes):
            line = embedding_file.readline()
            line_elements = line.split(" ")
            protein = int(line_elements[0])

            if protein not in protein_to_node: 
                continue  
                
            node_embedding = map(float, line_elements[1:])
            feature_matrix[protein_to_node[protein],:] = node_embedding
    return feature_matrix

def compute_lr_scores(features, train_pos, params):
    """ Runs logistic regression to compute scores for the nodes. 
    Args:
        features (list) list of feature matrices
        train_pos (string) 
        params (dictionary)
    """
    # X: concatenate all feature matrices passed in (allows combining multiple)
    X = np.concatenate(features)
    N, D = X.shape

    # Y: Build 
    Y = np.zeros(N)
    Y[train_pos] = 1

    # Get sample of negative examples
    train_neg = get_negatives(Y, len(train_pos))
    train_nodes = np.concatenate((train_pos, train_neg))
    Y_train = Y[train_nodes]
    X_train = X[train_nodes, :]

    # Model 
    model = LogisticRegression(C = 1.0 / params.reg_L2)

    # Run training 
    model.fit(X_train, Y_train)
    scores = model.predict_proba(X)[:,1]
    return scores