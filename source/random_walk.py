"""
Provides methods for running random walks on ppi
"""
from collections import defaultdict

import numpy as np 
import networkx as nx

def compute_random_walk_scores(ppi_networkx, training_ids, params):
    alpha = params.rw_alpha
    training_ids = set(training_ids)
    training_personalization = {node: (1.0/len(training_ids) if node in training_ids else 0) for node in ppi_networkx.nodes()}
    page_rank = nx.pagerank(ppi_networkx, alpha=alpha, personalization=training_personalization)
    scores = np.array([page_rank.get(node, 0) for node in ppi_networkx.nodes()])   
    return scores 