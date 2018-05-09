"""
Output pickle files based on our protein-protein interaction dataset

Create files that replicate the standardized train-test splits as follows

There are three different types of sets here
    Training (labeled) / Test (labeled) / Other (unlabeled)


x, the feature vectors of the labeled training instances,
    [examples, features]
y, the one-hot labels of the labeled training instances,
    [examples, hot-labels]
allx, the feature vectors of both labeled and unlabeled training instances (a superset of x),
    [all example, features].
graph, a dict in the format {index: [index_of_neighbor_nodes]}.

tx, the feature vectors of the test instances,
    [examples, features]
ty, the one-hot labels of the test instances,
    [examples, hot-labels]
test.index, the indices of test instances in graph, for the inductive setting,
    separate lines of test examples
ally, the labels for instances in allx.
    [all examples, hot-labels]
"""
from __future__ import division
import csv
import os
from collections import defaultdict
from random import shuffle
from multiprocessing import Pool

import numpy as np
import scipy.sparse
from sklearn.metrics import recall_score, precision_recall_curve, average_precision_score, accuracy_score
from sklearn.model_selection import KFold
import tensorflow as tf 

from gcn.train import perform_train, LAYER_SPEC
from gcn.utils import format_data, sample_mask, inverse_sample_mask
from features import build_ontology_feature_matrix, build_identity_matrix, build_motif_feature_matrix
from analysis import recall_at, auroc, average_precision, mean_rank, plot_auc
from output import ExperimentResults

def compute_gcn_scores(ppi_adj, training_nodes, )
class Disease: 
    def __init__(self, id, name, proteins):
        self.id = id
        self.name = name
        self.proteins = proteins

def load_disease_subset(disease_subset_file):
    subset = set()
    with open(disease_subset_file) as file: 
        for line in file: 
            if line[0] == '#': continue 
            subset.add(line[:-1])
    return subset 

def load_diseases(diseases_subset = None): 
    mapping = defaultdict(set)
    diseases_dict = {} 
    with open(ASSOCIATIONS_PATH) as associations_file:
        reader = csv.DictReader(associations_file)
        for row in reader:
            disease_id = row["Disease ID"]
            if(diseases_subset and disease_id not in diseases_subset):
                continue  
            disease_name = row["Disease Name"]
            disease_proteins = [int(a.strip()) for a in row["Associated Gene IDs"].split(",")]
            diseases_dict[disease_id] = Disease(disease_id, disease_name, disease_proteins)
    return diseases_dict 

def build_ppi_adjacency(protein_mapping):
    """
    Returns sparse [n,n] adjacency matrix of protein network described by the given file

    Node indices are indicated by the passed protein_mapping dictionary
    """
    adjacency = np.zeros((len(protein_mapping), len(protein_mapping)))

    with open(NETWORK_PATH) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]

            if p1 not in protein_mapping or p2 not in protein_mapping:
                continue

            n1, n2 = protein_mapping[p1], protein_mapping[p2]
            adjacency[n1, n2] = 1

    return scipy.sparse.coo_matrix(adjacency)

def build_graph(protein_mapping):
    """
    Returns dictionary {node_id, neighbor_ids}

    Node indicies are indicated by the passed protein_mapping dictionary
    """
    graph = {node_id: list() for node_id in protein_mapping.values()}

    with open(NETWORK_PATH) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]

            if p1 not in protein_mapping or p2 not in protein_mapping:
                continue

            n1, n2 = protein_mapping[p1], protein_mapping[p2]
            graph[n1].append(n2)
            graph[n2].append(n1)

    return graph

def generate_kfold_ids(total_indicies, n_splits):
    """
    Creates generator that splits indicies with k_fold validation

    This is a compliment to sklearn's KFold which works on actual matricies
    instead of indicies
    """
    kf = KFold(n_splits=n_splits)

    total_indicies = np.array(total_indicies)
    for train_index, test_index in kf.split(total_indicies):
        yield total_indicies[train_index].tolist(), total_indicies[test_index].tolist()

def fetch_disease_data(current_disease):
    # Find the proteins that are known to be within this disease
    all_nodes = set(protein_to_node.values())
    in_disease_nodes = set([protein_to_node[protein] for protein in current_disease.proteins if protein in protein_to_node])
    out_disease_nodes = all_nodes - in_disease_nodes 
    in_disease_nodes, out_disease_nodes = list(in_disease_nodes), list(out_disease_nodes)

    # Determine the proportion of negative examples that we want to include
    # within our disease data set - we do this to keep the proportion relative
    # to the disease size since this can vary wildly
    shuffle(out_disease_nodes)
    quantity_negative = int(FRACTION_NEGATIVE*len(in_disease_nodes) / (1-FRACTION_NEGATIVE))
    out_disease_nodes = out_disease_nodes[:quantity_negative]

    n = len(graph)
    k = 2 # binary classification of in-disease

    # Now build up the desired prediction classes
    # [examples, classes]
    Y = np.zeros((n, k))

    # Set the values depending on in_disease vs. out_of_disease
    # Leave the "unknown quantities blank"
    Y[out_disease_nodes, 0] = 1
    Y[in_disease_nodes, 1] = 1

    # Full protein ids under consideration
    # Re-map the protein to node mappings with this
    # We use the labeled ids as our training/test set
    labeled_ids = list(set(in_disease_nodes) | set(out_disease_nodes))
    #shuffle(labeled_ids)

    training_quantity = int(PROPORTION_TRAIN*len(labeled_ids))
    test_quantity = int(PROPORTION_TEST*len(labeled_ids))

    # Select the nodes that we'll use for our training+test sets
    training_ids = labeled_ids[:training_quantity]
    test_ids = labeled_ids[training_quantity:training_quantity+test_quantity]
    return X, XS, Y, training_ids, test_ids

def compute_numeric_metrics(metrics, y_true, prob_pred, training_ids, validation_ids):
    y_pred = np.array(map(float, prob_pred > 0.5))
    train_mask = sample_mask(training_ids, y_true.shape[0])
    no_train_mask = inverse_sample_mask(training_ids, y_true.shape[0])
    validation_mask = sample_mask(validation_ids, y_true.shape[0])
    metrics["Val Accuracy"].append(accuracy_score(y_true[validation_ids], y_pred[validation_ids])) 
    metrics["Train Accuracy"].append(accuracy_score(y_true[training_ids], y_pred[training_ids])) 
    metrics["Train Recall-at-100"].append(recall_at(y_true, prob_pred, 100, validation_ids))
    metrics["Train Recall-at-25"].append(recall_at(y_true, prob_pred, 25, validation_ids))
    metrics["Val Recall-at-100"].append(recall_at(y_true, prob_pred, 100, training_ids))
    metrics["Val Recall-at-25"].append(recall_at(y_true, prob_pred, 25, training_ids))
    metrics["Val AUROC"].append(auroc(y_true, prob_pred, training_ids))
    metrics["Val Average Precision"].append(average_precision(y_true, prob_pred, training_ids))
    metrics["Val Mean Rank"].append(mean_rank(y_true, prob_pred, training_ids))

def compute_visual_metrics(metrics, y_true, prob_pred, training_ids, validation_ids):
    plot_auc(y_true, prob_pred, validation_ids) 
    
def initialize_metrics():
    return {"Train Accuracy": [],
            "Val Accuracy": [],
            "Train Recall-at-100": [],
            "Train Recall-at-25": [], 
            "Val Recall-at-100": [],
            "Val Recall-at-25": [], 
            "Val AUROC": [],
            "Val Average Precision": [],
            "Val Mean Rank": []}

def has_pos_labels(Y):
    Y = np.rint(Y).astype(bool)
    return any(Y)

def train_on_disease(disease):
    X, XS, Y, training_ids, test_ids = fetch_disease_data(disease)
    metrics = initialize_metrics()
    fold_number = 1 
    hyper_parameters = {}
    #Initialize tensorflow session 
    sess = tf.Session()
    for training_ids, validation_ids in generate_kfold_ids(training_ids, n_splits=K_FOLDS):
        try: 
            data = format_data(X, XS, Y, graph, training_ids, validation_ids, test_ids)
            if(not has_pos_labels(Y[validation_ids, 1]) or not has_pos_labels(Y[training_ids, 1])): continue 
            epoch_outputs, epoch_activations, epoch_train_accs, epoch_val_accs, hyper_parameters = perform_train(*data, sess = sess, verbose=False)
            compute_numeric_metrics(metrics, Y[:,1], epoch_outputs[-1][:,1], training_ids, validation_ids)
            #compute_visual_metrics(metrics, Y[:,1], epoch_outputs[-1][:,1], training_ids, validation_ids)
        except Exception as e: 
            print "Exception on Disease Pathway: ", disease.name
            print "Exception Message: ", str(e)
            print "Returning Empty Metrics for Disease Pathway"
        if VERBOSE:
            print("{} - Fold {}  || Train Accuracy: {} -- Val Accuracy: {} -- Val Recall-at-100: {}".format(disease.id,
                                                                                                            fold_number,
                                                                                                            metrics["Train Accuracy"][-1], 
                                                                                                            metrics["Val Accuracy"][-1], 
                                                                                                            metrics["Val Recall-at-100"][-1]))
        fold_number += 1 
    sess.close()
    tf.reset_default_graph()
    # Return average recall
    avg_metrics = {name: np.mean(values) for name, values in metrics.iteritems()} 
    return (disease, avg_metrics, hyper_parameters)