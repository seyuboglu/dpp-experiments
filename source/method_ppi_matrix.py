"""
Provides methods for generating matrices that describe pairwise relationships 
between proteins in the protein-protein interaction network. 
"""
import os
import logging

from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

from disease import Disease, load_diseases, load_network
from output import ExperimentResults
from util import set_logger, get_negatives

def softmax(x):
    """softmax for a vector x. Numerically stable implementation
    Args:
        x (np.array)
    """
    exp_x = np.exp(x - np.max(x, axis = 0))
    return exp_x / np.sum(exp_x, axis = 0)

def compute_matrix_scores(ppi_matrix, training_ids, params):
    """ Compute scores across all proteins using ppi_matrix and training_ids. 
    Score for some protein k is given by some weighted sum of the ppi_matrix
    entries ppi_matrix[k, training_ids[i]] for all i. Weighting technique is
    specified in the params parameter. Default weighting is uniform. 
    Args: 
        ppi_matrix: (np.array) compute 
        training_ids: (np.array)
    Returns: 
        scores: (np.array)
    """
    # building weighting vector  
    weights = None
    if  not hasattr(params, 'weighting') or params.weighting == "uniform":
        weights = np.ones(len(training_ids))

    elif params.weighting == "sup":
        # compute supervised weights
        weights = np.sum(ppi_matrix[training_ids, :][:, training_ids], axis=1)
        
        # normalize 
        weights -= np.min(weights)
        weights /= np.sum(weights)
        weights += 1.0 / len(weights)
        weights = weights ** (-1)
    
    elif params.weighting == "mle":
        train_pos = training_ids
        X = ppi_matrix[:, train_pos]
        N, D = X.shape

        Y = np.zeros(N)
        Y = np.zeros(N)
        Y[train_pos] = 1

        train_neg = get_negatives(Y, params.neg_examples*len(train_pos))
        train_nodes = np.concatenate((train_pos, train_neg))
        Y_train = Y[train_nodes]
        X_train = X[train_nodes, :]
        model = LogisticRegression(C = 1.0 / params.reg_L2, 
                                   fit_intercept = params.intercept, 
                                   class_weight = 'balanced')
        model.fit(X_train, Y_train)
        return model.predict_proba(X)[:,1]
        #weights = model.coef_.T
        #print(weights)

    elif params.weighting == "pca":
        logging.error("Not Implemented")

    else: 
        logging.error("Weighting scheme not recognized")

    # normalize
    weights /= np.sum(weights)

    # get cns vector
    scores = np.dot(ppi_matrix[:, training_ids], weights) 
    # compute scores 
    return scores 

def build_ppi_comp_matrix(ppi_adj, deg_fn = 'id', row_norm = False, col_norm = False):
    """ Builds a CNS PPI matrix, using parameters specified, and saves as numpy object. 
    Args: 
        deg_fn (string)
        row_norm (bool)
        col_norm (bool)
    Return:
        comp_matrix (np.array)
    """
    name = 'comp'
    # Build vector of node degrees
    deg_vector = np.sum(ppi_adj, axis = 1, keepdims=True)

    # Apply the degree function
    name += '_' + deg_fn
    if deg_fn == 'log':
        # Take the natural log of the degrees. Add one to avoid division by zero
        deg_vector = np.log(deg_vector) + 1
    elif deg_fn == 'sqrt':
        # Take the square root of the degrees
        deg_vector = np.sqrt(deg_vector) 

    # Take the inverse of the degree vector
    inv_deg_vector = np.power(deg_vector, -1)

    # Build the complementarity matrix with sparse 
    comp_matrix = (csr_matrix((inv_deg_vector*ppi_adj).T) * csr_matrix(ppi_adj)).toarray()

    if(row_norm):
        # Normalize by the degree of the query node. (row normalize)
        name += '_rnorm'
        comp_matrix = inv_deg_vector * comp_matrix
    
    if(col_norm):
        # Normalize by the degree of the disease node. (column normalize)
        name += '_cnorm'
        comp_matrix = (comp_matrix.T * inv_deg_vector).T
    
    print(os.path.join('data', 'ppi_matrices', name + ".npy"))
    np.save(os.path.join('data', 'ppi_matrices', name + ".npy"), comp_matrix)
    return comp_matrix 

# Functions for building other ppi matrices
def build_dn_normalized():
    ppi_sqrt_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -(0.5))
    dn_norm = (ppi_adj*ppi_sqrt_inv_deg).T * ppi_sqrt_inv_deg
    np.save("data/ppi_matrices/dn_norm.npy", dn_norm)
    return dn_norm

def build_dn_query_normalized():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    dn_query_norm = ppi_adj*ppi_inv_deg
    np.save("data/ppi_matrices/dn_query_norm.npy", dn_query_norm)
    return dn_query_norm

def build_l3():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    ppi_l3 = np.dot((np.dot((ppi_inv_deg*ppi_adj).T, ppi_adj) * ppi_inv_deg).T, ppi_adj)
    np.save("data/ppi_matrices/ppi_l3.npy", ppi_l3)
    return ppi_l3

def build_l3_query_normalized():
    l3 = np.load("data/ppi_matrices/ppi_l3.npy")
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    l3_qnorm = l3 *ppi_inv_deg
    np.save("data/ppi_matrices/ppi_l3_qnorm.npy", l3_qnorm)
    return l3_qnorm


if __name__ == "__main__":
    print("Build PPI Matrices with PPI Network")
    print("Sabri Eyuboglu  -- Stanford University")
    print("======================================")

    print("Loading PPI Network...")
    _, ppi_network_adj, _ = load_network("data/networks/bio-pathways-network.txt")

    print("Building PPI Matrix...")
    build_ppi_comp_matrix(ppi_network_adj, deg_fn = 'sqrt', row_norm = True, col_norm = False)

