"""
Provides methods for generating matrices that describe pairwise relationships 
between proteins in the protein-protein interaction network. 
"""
import os
import logging

from collections import defaultdict
import numpy as np 
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

<<<<<<< HEAD


=======
>>>>>>> 3f113d621fc0bbe03228e78fa454e41f1d5675c6
#from utils import get_negatives

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
    if  not hasattr(params, 'weighting') or params.weighting == "uniform":
        weights = np.ones(len(training_ids))
        weights /= np.sum(weights)
        scores = np.dot(ppi_matrix[:, training_ids], weights) 

    elif params.weighting == "sup":
        # compute supervised weights
        weights = np.sum(ppi_matrix[training_ids, :][:, training_ids], axis=1)
        
        # normalize 
        weights -= np.min(weights)
        weights /= np.sum(weights)
        weights += 1.0 / len(weights)
        weights = weights ** (-1)

        weights /= np.sum(weights)
        scores = np.dot(ppi_matrix[:, training_ids], weights) 

    elif params.weighting == "mle":
<<<<<<< HEAD
        #train_pos = training_ids
        #X = ppi_matrix[:, train_pos]
        #N, D = X.shape

        #Y = np.zeros(N)
        #Y[train_pos] = 1

        #train_neg = get_negatives(Y, params.neg_examples*len(train_pos))
        #train_nodes = np.concatenate((train_pos, train_neg))
        #Y_train = Y[train_nodes]
        #X_train = X[train_nodes, :]
        #model = LogisticRegression(C = 1.0 / params.reg_L2, 
        #                           fit_intercept = False, 
        #                           class_weight = 'balanced')
        #model.fit(X_train, Y_train)
        #weights = np.array(model.coef_).reshape(-1)
        
        #Apply ReLU to Weights
        #weights += np.ones(len(training_ids))
        #weights /= np.sum(weights)
        #scores = np.dot(ppi_matrix[:, training_ids], weights) 
        pass
=======
        train_pos = training_ids
        X = ppi_matrix[:, train_pos]
        N, D = X.shape

        Y = np.zeros(N)
        Y[train_pos] = 1

        train_neg = get_negatives(Y, params.neg_examples*len(train_pos))
        train_nodes = np.concatenate((train_pos, train_neg))
        Y_train = Y[train_nodes]
        X_train = X[train_nodes, :]
        model = LogisticRegression(C = 1.0 / params.reg_L2, 
                                   fit_intercept = False, 
                                   class_weight = 'balanced')
        model.fit(X_train, Y_train)
        weights = np.array(model.coef_).reshape(-1)
        
        #Apply ReLU to Weights
        weights += np.ones(len(training_ids))
        weights /= np.sum(weights)
        scores = np.dot(ppi_matrix[:, training_ids], weights) 

>>>>>>> 3f113d621fc0bbe03228e78fa454e41f1d5675c6
    elif params.weighting == "pca":
        logging.error("Not Implemented")
    
    elif params.weighting == "max":
        scores = np.max(ppi_matrix[:, training_ids], axis = 1)

    else: 
        logging.error("Weighting scheme not recognized")

    # compute scores 
    return scores 

def build_ppi_comp_matrix(ppi_adj, deg_fn = 'id', row_norm = False, col_norm = False, 
                          self_loops = False, network_name = None):
    """ Builds a CNS PPI matrix, using parameters specified, and saves as numpy object. 
    Args: 
        deg_fn (string)
        row_norm (bool)
        col_norm (bool)
    Return:
        comp_matrix (np.array)
    """
    name = 'comp'
    if self_loops:
        ppi_adj += np.identity(ppi_adj.shape[0])
        name += '_sl'

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
    
    if network_name == None:
        file_path = os.path.join('data', 'ppi_matrices', name + ".npy")
    else: 
        file_path = os.path.join('data', 'ppi_matrices', network_name, name + ".npy")
    print(file_path)
    np.save(file_path, comp_matrix)
    return comp_matrix 

def build_ppi_dn_matrix(ppi_adj, deg_fn = 'id', row_norm = False, col_norm = False, network_name = None):
    """Builds a direct neighbor score matrix with optional normalization.
    """
    name = 'dn'

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

    dn_matrix = ppi_adj

    if(row_norm):
        # Normalize by the degree of the query node. (row normalize)
        name += '_rnorm'
        dn_matrix = inv_deg_vector * dn_matrix
    
    if(col_norm):
        # Normalize by the degree of the disease node. (column normalize)
        name += '_cnorm'
        dn_matrix = (dn_matrix.T * inv_deg_vector).T
    
    if network_name == None:
        file_path = os.path.join('data', 'ppi_matrices', name + ".npy")
    else: 
        file_path = os.path.join('data', 'ppi_matrices', network_name, name + ".npy")
    print(file_path)
<<<<<<< HEAD
    np.save(file_path, dn_matrix)
    return dn_matrix 
=======
    np.save(file_path, ppi_adj)
    return dn_matrix 

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
>>>>>>> 3f113d621fc0bbe03228e78fa454e41f1d5675c6
