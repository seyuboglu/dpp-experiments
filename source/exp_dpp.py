"""Run experiment"""

import argparse
import logging
import os
from multiprocessing import Pool


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import networkx as nx

from ppi_matrix import compute_matrix_scores
from random_walk import compute_random_walk_scores
from diamond import compute_diamond_scores
from disease import load_diseases, load_network
from output import ExperimentResults, write_dict_to_csv
from analysis import recall_at, recall, mean_rank, auroc, average_precision

from scipy.stats import rankdata


from util import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

def initialize_metrics():
    """Initialize the metrics container for the on one disease. 
    Each iteration should append its results to the lists in the metrics
    dictionary. 
    """
    metrics = {}

    for k in [100, 50, 25, 10]:
        metrics["Fold Recall-at-{}".format(k)] = []
    metrics["Fold Mean Rank"] = []
    metrics["Fold AUROC"] = []
    metrics["Fold Mean Average Precision"] = []

    for k in [100, 50, 25, 10]:
        metrics["Full Recall-at-{}".format(k)] = []
    metrics["Full Mean Rank"] = []
    metrics["Full AUROC"] = []
    metrics["Full Mean Average Precision"] = []

    return metrics

def compute_metrics(metrics, labels, scores, train_nodes, test_nodes):
    """Synthesize the metrics for one disease. 
    Args: 
        metrics: (dictionary) 
        labels: (ndarray) binary array indicating in-disease nodes
        scores: (ndarray) scores assigned to each node
        train_node: (ndarray) array of train nodes
        test_nodes: (ndarray) array of test nodes
    """

    for k in [100, 50, 25, 10]: 
        metrics["Full Recall-at-{}".format(k)].append(recall_at(labels, scores, k, train_nodes))
    metrics["Full Mean Rank"].append(mean_rank(labels, scores, train_nodes))
    metrics["Full AUROC"].append(auroc(labels, scores, train_nodes))
    metrics["Full Mean Average Precision"].append(average_precision(labels, scores, train_nodes))

    # Sample down to one-folds-worth of negative examples 
    out_of_fold = np.random.choice(np.arange(len(scores)), int(len(scores) * (1- 1.0/params.n_folds)), replace=False)
    fold_scores = scores.copy()
    fold_scores[out_of_fold] = 0.0
    fold_scores[test_nodes] = scores[test_nodes]

    for k in [100, 50, 25, 10]: 
        metrics["Fold Recall-at-{}".format(k)].append(recall_at(labels, fold_scores, k, train_nodes))
    metrics["Fold Mean Rank"].append(mean_rank(labels, fold_scores, train_nodes))
    metrics["Fold AUROC"].append(auroc(labels, fold_scores, train_nodes))
    metrics["Fold Mean Average Precision"].append(average_precision(labels, fold_scores, train_nodes))

def write_metrics(directory, disease_to_metrics):
    """Synthesize the metrics for one disease. 
    Args: 
        metrics: (dictionary) 
        directory: (string) directory to save results
        disease: (Disease)
    """
    # Output metrics to csv

    output_results = ExperimentResults()
    for disease, metrics in disease_to_metrics.items():  
        output_results.add_disease_row(disease.id, disease.name)
        output_results.add_data_row_multiple(disease.id, metrics)
    output_results.add_statistics()
    output_results.output_to_csv(os.path.join(directory, 'results.csv'))

def compute_node_scores(training_nodes):
    """ Get score 
    Args:
        disease: (Disease) A disease object
    """
    scores = None
    if params.method == 'ppi_matrix':
        scores = compute_matrix_scores(ppi_matrix, training_nodes)
    
    elif params.method == 'random_walk':
        scores = compute_random_walk_scores(ppi_networkx, training_nodes, alpha = params.rw_alpha)
    
    elif params.method == 'diamond':
        scores = compute_diamond_scores(ppi_networkx, training_nodes, 
                                            max_nodes = params.max_nodes, alpha = params.dm_alpha)
    else: 
        logging.error("No method" + params.method)
    
    return scores 

def run_dpp(disease):
    """ Perform k-fold cross validation on disease protein prediction on disease
    Args:
        disease: (Disease) A disease object
    """

    disease_nodes = disease.to_node_array(protein_to_node)
    labels = np.zeros((len(protein_to_node), 1))
    labels[disease_nodes, 0] = 1 
    metrics = initialize_metrics()

    # Perform k-fold cross validation
    n_folds = disease_nodes.size if params.n_folds < 0 or params.n_folds > len(disease_nodes) else params.n_folds
    kf = KFold(n_splits = n_folds, shuffle=False)
    for train_indices, test_indices in kf.split(disease_nodes):
        train_nodes = disease_nodes[train_indices]
        test_nodes = disease_nodes[test_indices]

        # Compute node scores 
        scores = compute_node_scores(train_nodes)

        # Compute the metrics of target node
        compute_metrics(metrics, labels, scores, train_nodes, test_nodes)

    # Create directory for disease 
    #disease_directory = os.path.join(args.experiment_dir, 'diseases', disease.id)
    #if not os.path.exists(disease_directory):
    #    os.makedirs(disease_directory)
    avg_metrics = {name: np.mean(values) for name, values in metrics.iteritems()} 
    return disease, avg_metrics 


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.update(json_path)

    # Set the logger
    set_logger(os.path.join(args.experiment_dir, 'experiment.log'), level=logging.INFO, console=True)

    # Log Title 
    logging.info("Disease Protein Prediction in the PPI Network")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    # Log Parameters
    #TODO

    # Load data from params file
    logging.info("Loading PPI Network...")
    ppi_network, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    ppi_networkx = nx.from_numpy_matrix(ppi_network_adj)
    logging.info("Loading Disease Associations...")
    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)
    if(params.method == "ppi_matrix"):
        logging.info("Loading PPI Matrix...")
        ppi_matrix = np.load(params.ppi_matrix)

    #Run Experiment
    logging.info("Running Experiment...")
    disease_to_metrics = {}
    if params.n_processes > 1: 
        p = Pool(params.n_processes)
        for n_finished, (disease, metrics) in enumerate(p.imap(run_dpp, diseases_dict.values()), 1):
            logging.info("Experiment Progress: {.1f}% -- {}/{}".format(100.0*n_finished/len(diseases_dict), 
                                                                   n_finished, len(diseases_dict)))
            disease_to_metrics[disease] = metrics
    else: 
        for i, disease in enumerate(diseases_dict.values()): 
            logging.info("Experiment Progress: {.1f}% -- {}/{}".format(100.0*n_finished/len(diseases_dict), n_finished, len(diseases_dict)))
            metrics = run_dpp(disease)
            disease_to_metrics[disease] = metrics
        
    write_metrics(args.experiment_dir, disease_to_metrics)