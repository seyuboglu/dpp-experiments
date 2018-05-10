"""Run experiment"""

import argparse
import logging
import os
import csv
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import KFold
import networkx as nx

from method_ppi_matrix import compute_matrix_scores
from method_random_walk import compute_random_walk_scores
from method_diamond import compute_diamond_scores
from method_gcn import compute_gcn_scores
from disease import load_diseases, load_network
from output import ExperimentResults, write_dict_to_csv
from analysis import positive_rankings, recall_at, recall, auroc, average_precision
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

    for k in [100, 25]:
        metrics["Recall-at-{}".format(k)] = []
    metrics["Ranks"] = []
    metrics["AUROC"] = []
    metrics["Mean Average Precision"] = []

    for k in [100]:
        metrics["Fold Recall-at-{}".format(k)] = []

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
    for k in [100, 25]: 
        metrics["Recall-at-{}".format(k)].append(recall_at(labels, scores, k, train_nodes))
    metrics["AUROC"].append(auroc(labels, scores, train_nodes))
    metrics["Mean Average Precision"].append(average_precision(labels, scores, train_nodes))
    metrics["Ranks"].extend(positive_rankings(labels, scores, train_nodes))


    # Sample down to one-folds-worth of negative examples 
    out_of_fold = np.random.choice(np.arange(len(scores)), int(len(scores) * (1- 1.0/params.n_folds)), replace=False)
    fold_scores = scores.copy()
    fold_scores[out_of_fold] = 0.0
    fold_scores[test_nodes] = scores[test_nodes]
    for k in [100]: 
        metrics["Fold Recall-at-{}".format(k)].append(recall_at(labels, fold_scores, k, train_nodes))

def write_metrics(directory, disease_to_metrics):
    """Synthesize the metrics for one disease. 
    Args: 
        directory: (string) directory to save results
        disease_to_metrics: (dict)
    """
    # Output metrics to csv

    output_results = ExperimentResults()
    for disease, metrics in disease_to_metrics.items():  
        output_results.add_disease_row(disease.id, disease.name)
        output_results.add_data_row_multiple(disease.id, metrics)
    output_results.add_statistics()
    output_results.output_to_csv(os.path.join(directory, 'metrics.csv'))

def write_ranks(directory, disease_to_ranks):
    """Write out the ranks for the proteins for one . 
    Args: 
        directory: (string) directory to save results
        disease_to_ranks: (dict)
    """
    # Output metrics to csv
    with open(os.path.join(directory, 'ranks.csv'), 'w') as file:
        ranks_writer = csv.writer(file)
        ranks_writer.writerow(['Disease ID', 'Disease Name', 'Protein Ranks'])
        for curr_disease, curr_ranks in disease_to_ranks.items():
            ranks_writer.writerow([curr_disease.id, curr_disease.name] + curr_ranks)

def compute_node_scores(train_nodes, val_nodes):
    """ Get score 
    Args:
        disease: (Disease) A disease object
    """
    scores = None
    if params.method == 'ppi_matrix':
        scores = compute_matrix_scores(ppi_matrix, train_nodes, params)
    
    elif params.method == 'random_walk':
        scores = compute_random_walk_scores(ppi_networkx, train_nodes, params)
    
    elif params.method == 'diamond':
        scores = compute_diamond_scores(ppi_networkx, train_nodes, params)

    elif params.method == 'gcn':
        scores = compute_gcn_scores(ppi_network_adj, train_nodes, val_nodes, params)

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
        val_nodes = disease_nodes[test_indices]

        # Compute node scores 
        scores = compute_node_scores(train_nodes, val_nodes)

        # Compute the metrics of target node
        compute_metrics(metrics, labels, scores, train_nodes, val_nodes)

    # Create directory for disease 
    #disease_directory = os.path.join(args.experiment_dir, 'diseases', disease.id)
    #if not os.path.exists(disease_directory):
    #    os.makedirs(disease_directory)
    avg_metrics = {name: np.mean(values) for name, values in metrics.iteritems()}
    ranks = metrics["Ranks"] 
    return disease, avg_metrics, ranks 


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
    ppi_networkx, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    logging.info("Loading Disease Associations...")
    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)
    if(params.method == "ppi_matrix"):
        logging.info("Loading PPI Matrix...")
        ppi_matrix = np.load(params.ppi_matrix)

    #print("COMP: ")
    #print(ppi_matrix.shape)
    #print(np.count_nonzero(ppi_matrix))

    #print("COMP Clipped:")
    #ppi_matrix_clipped = np.clip(ppi_matrix - (np.mean(ppi_matrix, axis=1) + 2 * np.std(ppi_matrix, axis=1)), 
    #                  a_min = 0,
    #                  a_max = None)
    #print(np.count_nonzero(ppi_matrix_clipped))


    #print("ADJ: ")
    #print(ppi_network_adj.shape)
    #print(np.count_nonzero(ppi_network_adj))

    #Run Experiment
    logging.info("Running Experiment...")
    disease_to_metrics, disease_to_ranks = {}, {}
    if params.n_processes > 1: 
        p = Pool(params.n_processes)
        for n_finished, (disease, metrics, ranks) in enumerate(p.imap(run_dpp, diseases_dict.values()), 1):
            logging.info("Experiment Progress: {:.1f}% -- {}/{}".format(100.0*n_finished/len(diseases_dict), 
                                                                   n_finished, len(diseases_dict)))
            disease_to_metrics[disease] = metrics
            disease_to_ranks[disease] = ranks 
    else: 
        for n_finished, disease in enumerate(diseases_dict.values()): 
            logging.info("Experiment Progress: {:.1f}% -- {}/{}".format(100.0*n_finished/len(diseases_dict), n_finished, len(diseases_dict)))
            try:
                disease, avg_metrics, ranks = run_dpp(disease)
                disease_to_metrics[disease] = avg_metrics
                disease_to_ranks[disease] = ranks 
            except Exception as e:
                print "Exception on GCN Execution:", str(e)
        
    write_metrics(args.experiment_dir, disease_to_metrics)
    write_ranks(args.experiment_dir, disease_to_ranks)
