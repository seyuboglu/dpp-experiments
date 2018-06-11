""" Defines DPPMethod
"""

class DPPMethod(object):
    """ 
    Base class for all disease protein prediction methods.
    """
    def __init__(self, params):
        self.params = params

    def compute_scores(self, train_nodes, val_nodes): 
        pass

    def __call__(self, train_nodes, val_nodes): 
        return self.compute_scores(train_nodes, val_nodes)