"""Run experiment"""

import argparse
import logging
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm

from data import load_diseases, load_network
from output import write_dict_to_csv
from analysis import compute_ranking

from util import Params, set_logger, prepare_sns

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
    _, _, protein_to_node = load_network(params.ppi_network)

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
    with tqdm(total=len(diseases_dict)) as t: 
        for i, disease in enumerate(diseases_dict.values()): 
            for name, ppi_matrix in ppi_matrices.items():
                disease_nodes = disease.to_node_array(protein_to_node)
                disease_scores = ppi_matrix[disease_nodes, :][:, disease_nodes]
                mean_disease_score = disease_scores.mean() 
                std_from_mean = (mean_disease_score - means[name]) / stds[name] 
                metrics[name].append(std_from_mean)
            t.update()
    
    #Plot results 
    prepare_sns(sns, params)

    for name, results in metrics.items(): 
        #sorted_std_from_mean = np.sort(metrics[name])
        #plt.semilogy(sorted_std_from_mean, label=name)
        #plt.plot(sorted_std_from_mean, label=name)
        sns.distplot(results, hist = False, kde_kws = {"shade": True}, label = name)

    time_string = datetime.datetime.now().strftime("%m-%d_%H%M")
    plot_path = os.path.join(args.experiment_dir, 'zscores_' + time_string + '.pdf')
    plt.xlim(xmax = 10)
    plt.ylabel('Density (estimated with KDE)')
    plt.xlabel('Average z-score')
    plt.legend()
    sns.despine(left = True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    plt.clf()