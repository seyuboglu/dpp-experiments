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

from gcn.train import perform_train
from gcn.utils import format_data, sample_mask, inverse_sample_mask, get_negatives

def show_saliency_maps(epoch_saliency_maps, val_pos, train_pos):
    """ Plot saliency maps 
    """
    
    for saliency_maps in epoch_saliency_maps:
        print(type(saliency_maps[0]))
        vec = np.array(saliency_maps[0][val_pos[0], :])
        plt.bar(np.arange(vec.shape[0]), vec)
        plt.show

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
    epoch_outputs, epoch_saliency_maps = perform_train(*data, train_pos = train_pos, val_pos= val_pos, params = params, verbose=params.verbose)
    scores = epoch_outputs[-1][:,1]

    show_saliency_maps(epoch_saliency_maps, val_pos, train_pos)

    return scores