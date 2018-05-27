"""
"""
from __future__ import division
import csv
import os
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool

import numpy as np

from gcn.train import perform_train
from gcn.utils import format_data, sample_mask, inverse_sample_mask, get_negatives

def compute_gcn_scores(ppi_adj_sparse, features_sparse, train_pos, val_pos, params):
    """ Compute the scores predicted by GCN.
    Args: 
        ppi_adj_sparse (sp.coo)
        features_sparse (sp.lil)
        train_pos (np.array)
        val_pos (np.array)
        params (dict)
    """
    # Adjacency: Get sparse representation of ppi_adj
    n = ppi_adj_sparse.shape[0]

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
    data = format_data(features_sparse, Y, ppi_adj_sparse, train_nodes, val_nodes)
    epoch_outputs = perform_train(*data, params = params, verbose=True)
    scores = epoch_outputs[-1][:,1]

    return scores