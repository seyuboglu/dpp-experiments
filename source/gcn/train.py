from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_bool('shuffle_negatives', False, 'Shuffle Negative Samples on each Epoch')

LAYER_SPEC = [('gcl', 128), ('sfl', 32), ('fcl', 16)]

m = FLAGS.model
hyper_parameters = FLAGS.__flags
hyper_parameters['layer_spec'] = str(LAYER_SPEC)


def perform_train(adj, features, side_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, sess = None, verbose = True):
    """
    Perform training process

    Returns the outputs from the last training pass
    """

    # Some preprocessing
    features = preprocess_features(features)
    #side_features = preprocess_features(side_features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'side_features': tf.placeholder(tf.float32, shape=side_features.shape, name='side_features'),
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64, name='features_shape'), name='features'),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], layer_spec=LAYER_SPEC, logging=True)

    #Initialize Session 
    if (not sess): 
        sess = tf.Session() 

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, side_features, support, labels, mask, placeholders)
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
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        if(FLAGS.shuffle_negatives):
            shuffle_negatives(y_train)
        np.asarray(side_features)
        feed_dict = construct_feed_dict(features, side_features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

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

        if epoch > FLAGS.early_stopping and epoch_val_costs[-1] > np.mean(epoch_val_costs[-(FLAGS.early_stopping+1):-1]):
            if(verbose): print("Early stopping...")
            break

    if(verbose): print("Optimization Finished!")


    return epoch_val_outputs, epoch_val_activations, epoch_train_accs, epoch_val_accs, hyper_parameters  

if __file__ == "__main__":
    # Load data
    print("Load data...")
    data = load_data(FLAGS.dataset)
    perform_train(*data)