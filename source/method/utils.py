"""
Utility functions for disease protein prediction methods. 
"""

import numpy as np

def get_negatives(Y, n_neg):
    """ Generate n_neg indices for negative examples
    excluding examples already positive in Y. 
    """
    n = Y.shape[0]
    n_pos = np.sum(np.sum(Y))
    neg_indices = np.random.choice(range(n), 
                                   size=int(n_neg), 
                                   replace=False, 
                                   p=(1 - Y) / (n-n_pos))                             
    return neg_indices 