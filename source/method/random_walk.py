"""
Provides methods for running random walks on ppi
"""
from collections import defaultdict

import numpy as np 
import networkx as nx

from method import DPPMethod


def compute_random_walk_scores(ppi_networkx, training_ids, params):
    alpha = params.rw_alpha
    training_ids = set(training_ids)
    training_personalization = {node: (1.0 / len(training_ids) 
                                if node in training_ids else 0) 
                                for node in ppi_networkx.nodes()}
    page_rank = nx.pagerank(ppi_networkx, 
                            alpha=alpha, 
                            personalization=training_personalization)
    scores = np.array([page_rank.get(node, 0) for node in ppi_networkx.nodes()])   
    return scores 


class L2RandomWalk(DPPMethod):
    """ L2RandomWalk method class
    """
    def __init__(self, params):
        super(L2RandomWalk, self).__init__(params)
        
        self.ppi_matrix = np.load(params.ppi_matrix)
        self.alpha = params.rw_alpha
        self.percentile = params.percentile

    def compute_scores(self, train_nodes, val_nodes):
        """
        """
        #ranked = np.argsort(np.sum(self.ppi_matrix[:, train_nodes], axis=1))
        #l2_nodes = ranked[int((1 - self.percentile) * len(ranked)):]

        l2_nodes = np.nonzero(np.sum(self.ppi_matrix[:, train_nodes], axis=1))
        l2_train_nodes = np.union1d(train_nodes, l2_nodes)
        l2_matrix = self.ppi_matrix[l2_train_nodes, :][:, l2_train_nodes]
        l2_networkx = nx.from_numpy_matrix(l2_matrix)
        print("Density: {}".format(nx.density(l2_networkx)))
        train_nodes = set(train_nodes)
        training_personalization = {i: (1.0 / len(train_nodes) 
                                    if node in train_nodes else 0) 
                                    for i, node in enumerate(l2_train_nodes)}

        page_rank = nx.pagerank(l2_networkx, 
                                alpha=self.alpha, 
                                personalization=training_personalization,
                                weight="weight")
        l2_scores = np.array([page_rank.get(node, 0) for node in l2_networkx.nodes()])   

        scores = np.zeros(self.ppi_matrix.shape[0])
        scores[l2_train_nodes] = l2_scores

        return scores 



    
