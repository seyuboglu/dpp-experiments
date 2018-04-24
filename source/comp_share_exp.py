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

def initialize_metrics():
    """Initialize the metrics container for the on one disease. 
    Each iteration should append its results to the lists in the metrics
    dictionary. 
    """
    metrics = {
        'top_comp_share': [], 
        'top_'+str(params.k)+'comp_share': [], 
        'comp_GINI': [],
        'comp_ranking': [],
        'comps': []
    }
    return metrics

def synthesize_metrics(directory, metrics, disease):
    """Synthesize the metrics for one disease. 
    Args: 
        metrics: (dictionary) 
        directory: (string) directory to save results
        disease: (Disease)
    """
    # Output metrics to csv
    results_path = os.path.join(directory, 'results.csv')
    write_dict_to_csv(results_path, metrics)

    # Plot GINI against target_comp_ranking
    plot_path = os.path.join(directory, 'gini_v_rank.png')
    plt.scatter(metrics['comp_ranking'], metrics['comp_GINI'])
    plt.ylabel('comp_GINI')
    plt.xlabel('comp_ranking')
    plt.title('Comp. GINI vs. Comp. Ranking')
    plt.savefig(plot_path)
    plt.close()

def synthesize_all_metrics(directory, all_metrics):
    """Synthesize the  metrics passed in with relevant graphs and 
    Args: 
        metrics: (list of dictionary) 
        directory: (string) directory to save results
    """
    # Plot the GINI by score percentiles
    rankings = []
    GINIs = []
    for metrics in all_metrics:
        GINIs.extend(metrics['comp_GINI'])
        rankings.extend(metrics['comp_ranking'])
    rankings, GINIs = np.array(rankings), np.array(GINIs)
    ranking_order = np.argsort(rankings)
    bucket_rankings, bucket_GINIs = [],[]
    for i, bucket_indices in enumerate(np.array_split(ranking_order, params.n_buckets)):
        mean_ranking = np.mean(rankings[bucket_indices])
        mean_GINI = np.mean(GINIs[bucket_indices])
        bucket_rankings.append(mean_ranking)
        bucket_GINIs.append(mean_GINI)

    # Plot the GINI by ranking buckets
    plot_path = os.path.join(args.experiment_dir, 'gini_v_rank.png')
    plt.plot(bucket_rankings, bucket_GINIs)
    plt.ylabel('Average Bucket GINI')
    plt.xlabel('Average Bucket Ranking')
    plt.title('Bucketed Ranking vs. GINI')
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def analyze_comp_share(disease):
    """ Repeatedly hold out one disease protein, and assess its relationship to the
    rest of the pathway. Also compute its complementarity ranking. 
    Args:
        disease: (Disease) A disease object
    """

    all_disease_nodes = disease.to_node_array(protein_to_node)
    metrics = initialize_metrics()

    # Hold out one disease protein at a time
    for target_index, target_node in enumerate(all_disease_nodes):
        disease_nodes = np.delete(all_disease_nodes, target_index)

        # Get comps between target and disease
        all_disease_comps = comp_matrix[:, disease_nodes]
        target_disease_comps = all_disease_comps[target_node, :]
        metrics['comps'].append(" ".join(str(comp) for comp in np.round(target_disease_comps, 2)))

        # Get sum of comps and then sort
        target_disease_comp_sum = np.sum(target_disease_comps)
        sorted_target_disease_comps = np.sort(target_disease_comps)

        #Ensure that at least one comp is nonnegative
        if (np.isclose(target_disease_comp_sum, 0)):
            continue

        # Compute the share of the complementarity coming from the top disease node 
        top_comp = sorted_target_disease_comps[-1]
        metrics['top_comp_share'].append((1.0 * top_comp) / target_disease_comp_sum)

        # Compute the share of complementarity coming from the top k% of disease nodes 
        top_k_comp_sum = np.sum(sorted_target_disease_comps[::-1][:int(np.round(params.k * len(disease_nodes)))])
        metrics['top_'+str(params.k)+'comp_share'].append(top_k_comp_sum/target_disease_comp_sum)

        # Compute the GINI coefficient 
        cumulative_sum_target_disease_comps = np.cumsum(sorted_target_disease_comps/target_disease_comp_sum)
        auc = np.trapz(cumulative_sum_target_disease_comps, dx = 1.0/len(target_disease_comps))
        gini_coeff = 1.0 - 2.0*auc
        metrics['comp_GINI'].append(gini_coeff)

        # Compute the Ranking of target node
        comp_scores = compute_comp_scores(disease_nodes)
        comp_ranking = compute_ranking(comp_scores)
        metrics['comp_ranking'].append(comp_ranking[target_node])
    
    # Create directory for disease 
    disease_directory = os.path.join(args.experiment_dir, 'diseases', disease.id)
    if not os.path.exists(disease_directory):
        os.makedirs(disease_directory)

    synthesize_metrics(disease_directory, metrics, disease)
    return metrics 


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
    logging.info("Loading Complementarity Matrix...")
    comp_matrix = np.load(params.comp_matrix)

    #Run Experiment
    logging.info("Running Experiment...")
    all_metrics = []
    for i, disease in enumerate(diseases_dict.values()): 
        logging.info(str(i) + ": " + disease.name)
        metrics = analyze_comp_share(disease)
        all_metrics.append(metrics)
    
    synthesize_all_metrics(args.experiment_dir, all_metrics)
        
