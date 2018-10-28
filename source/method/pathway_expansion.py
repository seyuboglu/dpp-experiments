"""
"""
from collections import defaultdict

import numpy as np 
import networkx as nx

from method import DPPMethod


class PathwayExpansion(DPPMethod):
    def __init__(self, params, ppi_network_adj):
        super(PathwayExpansion, self).__init__(params)
        self.ppi_network_adj = ppi_network_adj
        self.ppi_matrix = np.load(self.params.ppi_matrix)

    def compute_scores(self, train_nodes, val_nodes):
        """
        """
        expansion = np.sum(self.ppi_network_adj[:, train_nodes], axis=1)
        expansion /= len(train_nodes)
        #expansion[train_nodes] = 1

        scores = np.dot(self.ppi_matrix, expansion)
        return scores 


