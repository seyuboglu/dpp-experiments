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
from data import Disease, load_diseases, load_network
from method.ppi_matrix import compute_matrix_scores
from method.random_walk import compute_random_walk_scores
from method.graph_cn import GCN
from util import Params, set_logger
from exp_dpp import compute_metrics, write_metrics, write_ranks

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_dir', default='predictions/base_model',
                    help="Directory containing params.json")

MULTIPROCESS = False 
N_PROCESSES = 30

def disease_prediction(disease): 
    disease_nodes = disease.to_node_array(protein_to_node)
    validation_nodes = disease.to_node_array(protein_to_node, validation=True)
    labels = np.zeros((len(protein_to_node), 1))
    labels[disease_nodes, 0] = 1 
    labels[validation_nodes, 0] = 1

    if(params.method == "ppi_matrix"):
        # Compute Predictions using ppi_matrix 
        scores = compute_matrix_scores(ppi_matrix, disease_nodes, params)

    elif(params.method == "random_walk"):
        # Compute Predictions using: Random Walk
        scores = compute_random_walk_scores(ppi_networkx, disease_nodes, params)

    elif(params.method == "gcn"):
        scores = gcn_method(disease_nodes, validation_nodes)

    else:
        logging.error("No method " + params.method)
    
    output_predictions(scores, disease)
    
    if disease.validation_proteins != None:
        # get metrics
        metrics = {}
        compute_metrics(metrics, labels, scores, disease_nodes, validation_nodes)

        # compute proteins_to_rank
        proteins = [node_to_protein[node] for node in metrics["Nodes"]]
        ranks = metrics["Ranks"]
        proteins_to_ranks = {protein: ranks for protein, ranks in zip(proteins, ranks)}

        # average metrics 
        disease_directory = os.path.join(args.prediction_dir, 'diseases', disease.id)
        metrics = {name: np.mean(values) for name, values in metrics.items()}

        # write results to file 
        write_metrics(disease_directory, {disease: metrics})
        write_ranks(disease_directory, {disease: proteins_to_ranks})

def output_predictions(scores, disease):
    disease_directory = os.path.join(args.prediction_dir, 'diseases', disease.id)
    if not os.path.exists(disease_directory):
        os.makedirs(disease_directory)
    scores_path = os.path.join(disease_directory, disease.id + "_dns.csv") 
    with open(scores_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Protein ID', params.method +' Score'])
        for protein, node in protein_to_node.items(): 
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
    ppi_networkx, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    node_to_protein = {node: protein for protein, node in protein_to_node.items()}

    if(params.method == "ppi_matrix"):
        logging.info("Loading PPI Matrix...")
        ppi_matrix = np.load(params.ppi_matrix)

    elif (params.method == 'gcn'):
        gcn_method = GCN(params, ppi_network_adj)

    # Run Predictions
    if MULTIPROCESS:
        p = Pool(N_PROCESSES)
        for n_finished, result in enumerate(p.imap(disease_prediction, diseases.values()), 1):
            print("Experiment Progress: {} -- {}/{}".format(1.0*n_finished/len(diseases), n_finished, len(diseases)))
    else: 
        for i, disease in enumerate(diseases.values(), 1): 
            logging.info(str(i) + ": Running Prediction for Disease " + disease.name)
            disease_prediction(disease)