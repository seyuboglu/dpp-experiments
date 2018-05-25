from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import os

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

#CHECK THIS
os.environ['CUDA_VISIBLE_DEVICES'] = "/device:GPU:1"

def perform_train(adj, features, y_train, y_val, train_mask, val_mask, params, verbose = True):
    """
    Perform training process

    Returns the outputs from the last training pass
    """
    # Some preprocessing
    features = preprocess_features(features)
    if params.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif params.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, params.max_degree)
        num_supports = 1 + params.max_degree
        model_func = GCN
    elif params.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(params.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64, name='features_shape'), name='features'),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    with tf.device(params.device):
        model = model_func(placeholders, input_dim=features[2][1], params=params, logging=True)

    #Initialize Session 
    sess = tf.Session(config=tf.ConfigProto(allow_growth = True, allow_soft_placement = True))

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.outputs, model.activations], feed_dict=feed_dict_val) #Investigate THIS!
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


    # Init variables
    sess.run(tf.global_variables_initializer())

    epoch_train_costs = []
    epoch_train_accs = []
    epoch_val_costs = []
    epoch_val_accs = []

    epoch_train_outputs = []
    epoch_val_outputs = []
    epoch_train_activations = []
    epoch_val_activations = []

    # Train model
    for epoch in range(params.epochs):

        t = time.time()
        # Construct feed dictionary
        if(params.shuffle_negatives):
            shuffle_negatives(y_train)
            train_mask = np.array(np.sum(y_train, axis=1), dtype=np.bool)

        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: params.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.activations], feed_dict=feed_dict)
        epoch_train_costs.append(outs[1])
        epoch_train_accs.append(outs[2])
        epoch_train_outputs.append(outs[3])
        epoch_train_activations.append(outs[4])

        # Validation
        val_cost, val_acc, val_output, val_activations, duration = evaluate(features, support, y_val, val_mask, placeholders)
        epoch_val_costs.append(val_cost)
        epoch_val_accs.append(val_acc)
        epoch_val_outputs.append(val_output)
        epoch_val_activations.append(val_activations)

        # Print results
        if(verbose): print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(val_cost),
              "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > params.early_stopping and epoch_val_costs[-1] > np.mean(epoch_val_costs[-(params.early_stopping+1):-1]):
            if(verbose): print("Early stopping...")
            break

    if(verbose): print("Optimization Finished!")

    sess.close()
    tf.reset_default_graph()

    return epoch_val_outputs  

