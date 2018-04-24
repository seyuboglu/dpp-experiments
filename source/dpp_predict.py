"""Run experiment"""

from sets import Set 
import csv
import argparse
import logging
import os
from multiprocessing import Pool

import numpy as np 
import snap 

from output import ExperimentResults
from disease import Disease, load_diseases, load_network
from ppi_matrix import compute_matrix_scores, protein_to_node, compute_comp_scores, compute_adj_set, compute_diamond_set, compute_direct_neighborhood_scores
from random_walk import compute_random_walk_scores
from analysis import recall_at, recall, mean_rank, auroc, average_precision, plot_prc
from util import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_dir', default='predictions/base_model',
                    help="Directory containing params.json")

MULTIPROCESS = False 
N_PROCESSES = 30

def disease_prediction(disease): 
    disease_nodes = disease.to_node_array(protein_to_node)

    if(params.method == "ppi_matrix"):
        # Compute Predictions using ppi_matrix 
        ppi_matrix_scores = compute_matrix_scores(ppi_matrix, disease_nodes)
        output_predictions(ppi_matrix_scores, disease)

    elif(params.method == "random_walk"):
        # Compute Predictions using: Random Walk
        rw_scores = compute_random_walk_scores(disease_nodes)
        output_predictions(rw_scores, disease)

    else:
        logging.error("No method " + params.method)

def output_predictions(scores, disease):
    disease_directory = os.path.join(args.prediction_dir, 'diseases', disease.id)
    if not os.path.exists(disease_directory):
        os.makedirs(disease_directory)
    scores_path = os.path.join(disease_directory, disease.id + "_dns.csv") 
    with open(scores_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Protein ID', params.method +' Score'])
        for protein, node in protein_to_node.iteritems(): 
            writer.writerow([protein, scores[node]])

if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.prediction_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.update(json_path)

    # Set the logger
    set_logger(os.path.join(args.prediction_dir, 'experiment.log'), level=logging.INFO, console=True)

    # Log Title 
    logging.info("Disease protein Prediction")
    logging.info("Sabri Eyuboglu  -- SNAP Group -- Stanford University")
    logging.info("======================================")

    # Load Data
    logging.info("Loading Diseases...")
    diseases = load_diseases(params.diseases_path, params.disease_subset)
    logging.info("Loading PPI Network...")
    ppi_network, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    if(params.method == "ppi_matrix"):
        logging.info("Loading PPI Matrix...")
        ppi_matrix = np.load(params.ppi_matrix)

    # Run Predictions
    if MULTIPROCESS:
        p = Pool(N_PROCESSES)
        for n_finished, result in enumerate(p.imap(disease_prediction, diseases.values()), 1):
            print("Experiment Progress: {} -- {}/{}".format(1.0*n_finished/len(diseases), n_finished, len(diseases)))
    else: 
        for i, disease in enumerate(diseases.values(), 1): 
            logging.info(str(i) + ": Running Prediction for Disease " + disease.name)
            disease_prediction(disease)