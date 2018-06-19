"""
"""
from __future__ import division
import csv
import os
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import spearmanr

from gcn.train import perform_train
from gcn.utils import format_data, sample_mask, inverse_sample_mask, get_negatives
from method import DPPMethod

class GCN(DPPMethod):
    """ GCN method class
    """
    def __init__(self, params, ppi_network_adj):
        super(GCN, self).__init__(params)
        self.ppi_adj = ppi_network_adj
        # build sparse adjacency matrix
        self.ppi_adj_sparse = sp.coo_matrix(ppi_network_adj)

        # build sparse feature matrix 
        features_sparse = np.identity(ppi_network_adj.shape[0])
        features_sparse = features_sparse.astype(np.float32)
        self.features_sparse = sp.coo_matrix(features_sparse).tolil()

    def compute_scores(self, train_pos, val_pos):
        """ Compute the scores predicted by GCN.
        Args: 
            ppi_adj_sparse (sp.coo)
            features_sparse (sp.lil)
            train_pos (np.array)
            val_pos (np.array)
            params (dict)
        """
        # Adjacency: Get sparse representation of ppi_adj
        n = self.ppi_adj_sparse.shape[0]

        # Y: Build 
        Y = np.zeros((n, 2))
        Y[train_pos, 1] = 1
        Y[val_pos, 1] = 1
        train_neg = get_negatives(Y, len(train_pos))
        Y[train_neg, 0] = 1

        # Create index arrays
        train_nodes = np.concatenate((train_pos, train_neg))
        val_nodes = val_pos

        # Run training 
        data = format_data(self.features_sparse, Y, self.ppi_adj_sparse, train_nodes, val_nodes)
        epoch_outputs, saliency_maps = perform_train(*data, train_pos = train_pos, val_pos= val_pos, params = self.params, verbose=self.params.verbose)
        scores = epoch_outputs[-1][:,1]

        if self.params.saliency_map:
            self.saliency_maps = saliency_maps
            self.val_pos = val_pos 
            self.train_pos = train_pos

        return scores

    def analyze_saliency_maps(self, directory, node_to_protein):
        """ Plot saliency maps 
        Args:
            directory  
        """
        assert(hasattr(self, 'saliency_maps'))

        correlations = []
        p_values = []

        deg = np.sum(self.ppi_adj, axis = 0)

        # compute comp vector
        X = (self.ppi_adj[self.train_pos, :].T / deg[self.train_pos]).T
        X = np.sum(X, axis = 0)
        comp = X / deg

        train_neighbors = np.any(self.ppi_adj[self.train_pos, :].astype(bool), axis = 0).astype(int)
        
        for i, node in enumerate(self.val_pos):
            # get common neighbors between train and node
            node_neighbors = self.ppi_adj[node, :].astype(int)
            cn = train_neighbors[node_neighbors.astype(bool)]
            
            cn_comp = comp[node_neighbors.astype(bool)]

            # get saliency map 
            saliency_map = self.saliency_maps[i][node, node_neighbors.astype(bool)]

            # compute pearson rank correlation 
            if np.all(cn == 1) or np.all(cn == 0):
                continue 
            rank_corr, p_value = spearmanr(saliency_map, cn_comp)
            correlations.append(rank_corr)
            p_values.append(p_value)
            print("Node:", node_to_protein[node])
            print("PValue:", p_value)
            print("Rank Correlation:", rank_corr)
        
        return correlations, p_values