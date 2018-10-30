"""
"""
from collections import defaultdict

import numpy as np 
import networkx as nx

from method import DPPMethod


class PathwayExpansion(DPPMethod):
    def __init__(self, params, ppi_networkx, ppi_network_adj):
        super(PathwayExpansion, self).__init__(params)
        self.ppi_network_adj = ppi_network_adj
        self.ppi_networkx = ppi_networkx

        if self.params.sub_method == "ppi_matrix":
            self.ppi_matrix = np.load(self.params.ppi_matrix)

    def compute_scores(self, train_nodes, val_nodes):
        """
        """
        expansion = np.sum(self.ppi_network_adj[:, train_nodes], axis=1)
        expansion /= len(train_nodes)
        expansion[train_nodes] = 1
        expansion_sum = np.sum(expansion)

        if self.params.sub_method == "random_walk":
            personalization = {i: (score / expansion_sum) 
                               for i, score in enumerate(expansion)}
            page_rank = nx.pagerank(self.ppi_networkx,
                                    alpha=self.params.alpha,
                                    personalization=personalization)
            scores = np.array([page_rank.get(node, 0) 
                               for node 
                               in self.ppi_networkx.nodes()])

        elif self.params.sub_method == "ppi_matrix": 
            scores = np.dot(self.ppi_matrix, expansion)

        return scores 


