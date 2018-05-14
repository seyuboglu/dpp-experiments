"""
Logistic regression metho for disease protein prediction
"""
from random import shuffle

import numpy as np
from sklearn.linear_model import LogisticRegression

from util import get_negatives

def compute_lr_scores(features, train_pos, params):
    # X: concatenate all feature matrices passed in (allows combining multiple)
    X = np.concatenate(features)
    N, D = X.shape

    # Y: Build 
    Y = np.zeros(N)
    Y[train_pos] = 1

    # Get sample of negative examples
    train_neg = get_negatives(Y, len(train_pos))
    train_nodes = np.concatenate((train_pos, train_neg))
    Y_train = Y[train_nodes]
    X_train = X[train_nodes, :]

    # Model 
    model = LogisticRegression(C = 1.0 / params.reg_L2)

    # Run training 
    model.fit(X_train, Y_train)
    scores = model.predict_proba(X)[:,1]
    return scores