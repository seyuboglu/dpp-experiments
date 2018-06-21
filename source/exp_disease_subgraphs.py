"""Run experiment"""

import argparse
import logging
import os
import csv

import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx

from disease import load_diseases, load_network, load_gene_names
from output import ExperimentResults
from exp_dpp import load_ranks

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
    training_nodes = np.delete(disease_nodes, np.argwhere(disease_nodes == query_node))
    matrix_comp = np.load(ppi_matrix, training_nodes)[query_node]

    logging.info('Comp - Matrix Comp Difference: ' + str(comp-matrix_comp))
    logging.info('Comp - Matrix Relative Difference: ' + str(comp / (matrix_comp)))
    logging.info('Comp:' + str(comp))
    logging.info('Matrix Comp:' + str(matrix_comp))
    logging.info('Elementwise-Difference:')

    return comp

def compute_disease_subgraph(disease, disease_directory):
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

    return induced_subgraph, intermediate_nodes, node_to_node_to_common

def initialize_metrics():
    """ Initialize metrics dictionary
    Args:
        None
    """
    metric_names = ["Disease ID", "Disease Name", "# of Disease Nodes", "# of Intermediate Nodes", 
                   "Density of Disease Subgraph", "Density of Disease-Intermediate Subgraph", "Density of Intermediate Subgraph",
                   "Conductance of Disease Nodes", "Conductance of Disease-Intermediate Nodes", "Conductance of Intermediate Nodes",
                   "Conductance of Intermediate Nodes deg < " + str(params.degree_cutoff), 
                   "Mean Frac. of Intermediate-Disease Interactions"]
    metrics = {field_name: None for field_name in metric_names}
    return metric_names, metrics

def write_metrics(path, metric_names, metrics):
    """ Initialize metrics dictionary
    Args:
        metric_names: (List of String) List of the metric names used as header of csv
        metrics: (List of Dictionaries) List of dictionaries 
    """
    with open(path, 'w') as file:
        metric_writer = csv.DictWriter(file, metric_names)
        metric_writer.writeheader()
        for curr_metrics in metrics:
            metric_writer.writerow(curr_metrics)

def analyze_disease_subgraph(disease, subgraph, intermediate_nodes, node_to_node_to_common):
    """ Output the disease subgraph to a file
    Args:
        disease: (Disease) disease object
        subgraph: (networkx) the networkx subgraph object
    """
    metric_names, metrics = initialize_metrics()
    disease_nodes = set(disease.to_node_array(protein_to_node))
    intermediate_nodes = set(intermediate_nodes)
    intermediate_nodes_k = set([node for node in intermediate_nodes if ppi_networkx.degree(node) < params.degree_cutoff])

    # Fill in disease info. 
    metrics["Disease ID"] = disease.id 
    metrics["Disease Name"] = disease.name

    # Compute the number of intermediate and disease nodes 
    metrics["# of Intermediate Nodes"] = len(intermediate_nodes)
    metrics["# of Disease Nodes"] = len(disease_nodes)

    # Compute the density of the disease pathway and intermediate subgraph
    metrics["Density of Disease Subgraph"] = nx.density(subgraph.subgraph(disease_nodes))
    metrics["Density of Disease-Intermediate Subgraph"] = nx.density(subgraph)
    metrics["Density of Intermediate Subgraph"] = nx.density(subgraph.subgraph(intermediate_nodes))

    # Compute the conductance of the disease pathway and intermediate subgraph
    metrics["Conductance of Disease Nodes"] = nx.algorithms.cuts.conductance(ppi_networkx, disease_nodes)
    metrics["Conductance of Disease-Intermediate Nodes"] = nx.algorithms.cuts.conductance(ppi_networkx, 
                                                                                          intermediate_nodes | disease_nodes)
    metrics["Conductance of Intermediate Nodes"] = nx.algorithms.cuts.conductance(ppi_networkx, intermediate_nodes)
    metrics["Conductance of Intermediate Nodes deg < " + str(params.degree_cutoff)] = nx.algorithms.cuts.conductance(ppi_networkx, intermediate_nodes_k | disease_nodes)

    # Compute average number of connections for intermediate node
    metrics["Mean Frac. of Intermediate-Disease Interactions"] = np.mean([1.0*len(set(subgraph.neighbors(intermediate_node)) & disease_nodes) / len(disease_nodes) 
                           for  intermediate_node in intermediate_nodes])

    return metrics 


def write_disease_subgraph(disease, subgraph, directory):
    """ Output the disease subgraph to a file
    Args:
        disease: (Disease) disease object
        subgraph: (networkx) the networkx subgraph object
        directory: (string) the directory to write the subgraph
    """
    disease_nodes = set(disease.to_node_array(protein_to_node))
    with open(os.path.join(directory, disease.id + '_comp_subgraph.txt'), "wb") as subgraph_file:
        for edge in subgraph.edges():
            # Add edge terminals
            line = str(edge[0]) + " " + str(edge[1])

            # Add interaction type
            interaction_type = (1 if edge[0] in disease_nodes else 0) + (1 if edge[1] in disease_nodes else 0) 
            line += " " + str(interaction_type)

            subgraph_file.write(line + '\n')

    # Write node data into a csv file
    with open(os.path.join(directory, disease.id + '_node_data.csv'), 'w') as node_data_file:
        fieldnames = ['Node ID', 'Protein ID', 'Protein Name', 'Disease Node', 'Degree']
        fieldnames.extend([method_name for method_name, _ in params.method_exp_dirs.items()]) 
        node_data_writer = csv.DictWriter(node_data_file, fieldnames)
        node_data_writer.writeheader()
        for node in subgraph.nodes():
            protein = node_to_protein[node]
            row_dict = {'Node ID': node, 
                        'Protein ID': protein,
                        'Protein Name': protein_to_name.get(protein, ""),
                        'Disease Node': 1 if node in disease_nodes else 0,
                        'Degree': ppi_networkx.degree(node)}
            for  method_name, _ in params.method_exp_dirs.items():
                protein_to_rank = method_to_ranks[method_name][disease.id]
                if protein in protein_to_rank:
                    row_dict[method_name] = protein_to_rank[protein]
            node_data_writer.writerow(row_dict)

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
    node_to_protein = {node: protein for protein, node in protein_to_node.items()}
    ppi_networkx = nx.from_numpy_matrix(ppi_network_adj)
    logging.info("Loading Disease Associations...")
    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)

    logging.info("Loading Protein Names...")
    protein_to_name, _ = load_gene_names(params.protein_names_path)

    logging.info("Loading PPI Matrix...")
    ppi_matrix = np.load(params.ppi_matrix)

    logging.info("Loading Protein Ranks")
    method_to_ranks = {}
    for method_name, method_dir in params.method_exp_dirs.items(): 
        disease_to_ranks = load_ranks(method_dir)
        method_to_ranks[method_name] = disease_to_ranks

    #Run Experiment
    logging.info("Running Experiment...")
    all_metrics = []
    metric_names, _ = initialize_metrics()
    for i, disease in enumerate(diseases_dict.values()): 
        logging.info(str(i+1) + ": " + disease.name)
        # Create directory for disease 
        disease_directory = os.path.join(args.experiment_dir, 'diseases', disease.id)
        if not os.path.exists(disease_directory):
            os.makedirs(disease_directory)
        induced_subgraph, intermediate_nodes, node_to_node_to_common = compute_disease_subgraph(disease, disease_directory)
        metrics = analyze_disease_subgraph(disease, induced_subgraph, intermediate_nodes, node_to_node_to_common)
        all_metrics.append(metrics)
        write_disease_subgraph(disease, induced_subgraph, disease_directory)
        write_metrics(os.path.join(disease_directory, disease.id + '_metrics.csv'), metric_names, [metrics])
    
    write_metrics(os.path.join(args.experiment_dir, 'subgraph_metrics.csv'), metric_names, all_metrics)
     
        