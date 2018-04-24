"""Run experiment"""

import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt 

from ppi_matrix import compute_comp_scores
from disease import load_diseases, load_network
from output import write_dict_to_csv
from analysis import compute_ranking

from util import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

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
    logging.info("Complementarity Sharing in Disease Pathways")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    # Load data from params file
    logging.info("Loading PPI Network...")
    ppi_network, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    logging.info("Loading Disease Associations...")
    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)
    logging.info("Loading PPI Matrices...")
    ppi_matrices = {}
    for name, path in params.ppi_matrices.items():
        ppi_matrices[name] = np.load(path)

    #Run Experiment
    logging.info("Running Experiment...")

    #Compute mean and standard deviation
    means = {}
    stds = {}
    for name, ppi_matrix in ppi_matrices.items():
        means[name] = ppi_matrix.mean()
        stds[name] = ppi_matrix.std()

    #Initialize Metrics
    metrics = {}
    for name, ppi_matrix in ppi_matrices.items():
        metrics[name] = []

    #Analyze diseases 
    for i, disease in enumerate(diseases_dict.values()): 
        logging.info(str(i) + ": " + disease.name)
        for name, ppi_matrix in ppi_matrices.items():
            disease_nodes = disease.to_node_array(protein_to_node)
            disease_scores = ppi_matrix[disease_nodes, :][:,disease_nodes]
            mean_disease_score = disease_scores.mean() 
            std_from_mean = (mean_disease_score - means[name]) / stds[name] 
            metrics[name].append(std_from_mean)
    
    #Plot results 
    for name, results in metrics.items(): 
        sorted_std_from_mean = np.sort(metrics[name])
        plt.semilogy(sorted_std_from_mean, label=name, )
    plot_path = os.path.join(args.experiment_dir, 'all_std_from_mean_log.png')
    plt.ylabel('Z-score')
    plt.xlabel('Diseases Sorted by Z-score')
    plt.title('Complementarity vs. DNS Disease Z-score')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    
        
