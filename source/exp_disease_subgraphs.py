"""Run experiment"""

import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx

from disease import load_diseases, load_network
#from ppi_matrix import compute_matrix_scores 

from util import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

def compute_complementarity(query_node, node_to_node_to_common, disease_nodes):
    """ Compute the complementarity of the query_node with respect to the disease
    Args: 
        query_node: (int) the query node
        node_to_node_to_common: a dictionary of dictionaries giving common neighbors 
        of disease node pairs
    """
    comps = []
    for _, common_nbrs in node_to_node_to_common[query_node].items():
        curr_comp = 0.0
        for common_nbr in common_nbrs:
            curr_comp += 1.0 / np.sqrt(ppi_networkx.degree(common_nbr))
        curr_comp *= 1.0 / np.sqrt(ppi_networkx.degree(query_node))
        comps.append(curr_comp)
    comp = np.mean(comps)

    # Sanity check
    if(sanity_check):
        training_nodes = np.delete(disease_nodes, np.argwhere(disease_nodes == query_node))
        matrix_comp = np.load(ppi_matrix, training_nodes)[query_node]

        logging.info('Comp - Matrix Comp Difference: ' + str(comp-matrix_comp))
        logging.info('Comp - Matrix Relative Difference: ' + str(comp / (matrix_comp)))
        logging.info('Comp:' + str(comp))
        logging.info('Matrix Comp:' + str(matrix_comp))
        logging.info('Elementwise-Difference:')

    return comp

def get_disease_subgraph(disease, disease_directory):
    """ Get the disease subgraph of 
    Args:
        disease: (Disease) A disease object
    """
    disease_nodes = disease.to_node_array(protein_to_node)
    node_to_nbrs = {node: set(ppi_networkx.neighbors(node)) for node in disease_nodes}

    # Dictionary of dictionaries where node_to_node_to_common[a][b] 
    # gives the common neighbors of a and b
    node_to_node_to_common = {node: {} for node in disease_nodes}

    # The set of nodes intermediate between nodes in the 
    intermediate_nodes = set([])

    for a, node_a in enumerate(disease_nodes):
        for b, node_b in enumerate(disease_nodes):
            # avoid repeat pairs
            if a >= b:
                continue
            common_nbrs = node_to_nbrs[node_a] & node_to_nbrs[node_b]
            intermediate_nodes.update(common_nbrs)
            node_to_node_to_common[node_a][node_b] = common_nbrs 
            node_to_node_to_common[node_b][node_a] = common_nbrs
     

    # Get Induced Subgraph 
    induced_subgraph = ppi_networkx.subgraph(intermediate_nodes | set(disease_nodes))

    return induced_subgraph, node_to_node_to_common

def write_disease_subgraph(subgraph, disease, directory):
    """ Output the disease subgraph to a file
    Args:
        subgraph: (networkx) the networkx subgraph object
        directory: (string) the directory to write the subgraph
    """
    disease_nodes = set(disease.to_node_array(protein_to_node))
    with open(os.path.join(directory, 'comp_subgraph.txt'), "wb") as subgraph_file:
        for edge in subgraph.edges():
            for i in range(2): 
                # Add edge terminals
                line = str(edge[0]) + " " + str(edge[1])
                # Add interaction type
                interaction_type = (1 if edge[0] in disease_nodes else 0) + (1 if edge[1] in disease_nodes else 0) 
                line += " " + str(interaction_type)
                # Add disease indicator
                line += " " + str(1 if edge[0] in disease_nodes else 0) 
                line += " " + str(1 if edge[1] in disease_nodes else 0) 
                # Add inverse degree indicator 
                line += " " + str(1.0/np.sqrt(ppi_networkx.degree(edge[0])))
                line += " " + str(1.0/np.sqrt(ppi_networkx.degree(edge[1])))
                subgraph_file.write(line + '\n')
                # Reverse Edge
                edge = edge[::-1]

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
    logging.info("Subgraphs in Disease Pathways")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    # Load data from params file
    logging.info("Loading PPI Network...")
    ppi_network, ppi_network_adj, protein_to_node = load_network(params.ppi_network)
    ppi_networkx = nx.from_numpy_matrix(ppi_network_adj)
    logging.info("Loading Disease Associations...")
    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)
    logging.info("Loading PPI Matrix...")
    ppi_matrix = np.load(params.ppi_matrix)

    #Run Experiment
    logging.info("Running Experiment...")
    all_metrics = []
    for i, disease in enumerate(diseases_dict.values()): 
        logging.info(str(i+1) + ": " + disease.name)
        # Create directory for disease 
        disease_directory = os.path.join(args.experiment_dir, 'diseases', disease.id)
        if not os.path.exists(disease_directory):
            os.makedirs(disease_directory)
        induced_subgraph,_ = get_disease_subgraph(disease, disease_directory)
        with open(os.path.join(disease_directory, 'comp_subgraph.txt'), "wb") as subgraph_file: 
            write_disease_subgraph(induced_subgraph,  disease, disease_directory)