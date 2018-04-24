# dpp_hop 
# Execute disease protein prediction experiments using k-hop based methods

from sets import Set 
import csv
from multiprocessing import Pool

import numpy as np 
import snap 

from sklearn.model_selection import KFold

from output import ExperimentResults
from disease import Disease, load_diseases, load_network
from ppi_matrix import compute_comp_scores, compute_adj_set, compute_diamond_set, compute_direct_neighborhood_scores
from analysis import recall_at, recall, mean_rank, auroc, average_precision

OUTPUT_FILENAME = "results/dpp-diamond-comp.csv"
K_FOLDS = 5

MULTIPROCESS = True 
N_PROCESSES = 50

def sample_mask(ids, l):
    mask = np.zeros((l,1))
    mask[ids] = 1
    return np.array(mask, dtype=np.bool)

def get_labels(protein_to_node, disease): 
    labels = np.zeros((len(protein_to_node), 1))
    pos_indices = [protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node]
    labels[pos_indices, 0] = 1
    return labels
    
def initialize_metrics():
    metrics = {}
    for k in [100, 50, 25, 10]:
        metrics["Val Recall-at-{}".format(k)] = []
    metrics["Val Mean Rank"] = []
    metrics["Val AUROC"] = []
    metrics["Val Mean Average Precision"] = []
    return metrics

def initialize_comparison_metrics():
    metrics = {}
    metrics["Comp Val Recall_at_Adj"] = []
    metrics["Adj Val Recall_at_Adj"] = []
    metrics["Recall Diff: Comp - Adj"] = []
    return metrics

def compute_metrics(metrics, labels, scores, training_nodes, validation_nodes):
    for k in [100, 25]: 
        metrics["Val Recall-at-{}".format(k)].append(recall_at(labels, scores, k, training_nodes))
    metrics["Val Mean Rank"].append(mean_rank(labels, scores, training_nodes))
    metrics["Val AUROC"].append(auroc(labels, scores, training_nodes))
    metrics["Val Mean Average Precision"].append(average_precision(labels, scores, training_nodes))

def compute_comparison_metrics(metrics, labels, comp_scores, adj_scores, training_nodes, validation_nodes, k):
    metrics["Comp Val Recall_at_Adj"].append(recall_at(labels, comp_scores, k, training_nodes))
    metrics["Adj Val Recall_at_Adj"].append(recall_at(labels, adj_scores, k, training_nodes))
    metrics["Recall Diff: Comp - Adj"].append(metrics["Comp Val Recall_at_Adj"][-1]-metrics["Adj Val Recall_at_Adj"][-1])

def generate_kfold_ids(total_indicies, n_splits):
    kf = KFold(n_splits=n_splits)
    total_indicies = np.array(total_indicies)
    for train_index, test_index in kf.split(total_indicies):
        yield total_indicies[train_index].tolist(), total_indicies[test_index].tolist()

def disease_experiment(disease): 
    #metrics = initialize_metrics()
    metrics = {}
    metrics["Val Recall_at_100"] = []
    disease_nodes = np.array([protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node])
    labels = get_labels(protein_to_node, disease)
    for training_ids, validation_ids in generate_kfold_ids(range(len(disease_nodes)), n_splits=K_FOLDS):
        # Split nodes into training and validation sets 
        training_nodes = disease_nodes[training_ids]
        validation_nodes = disease_nodes[validation_ids]

        # Compute predictions using: DIAMOnD
        diamond_set = compute_diamond_set(training_nodes, 100)
        metrics["Val Recall_at_100"].append(recall(labels, diamond_set, training_nodes))
        print "Val Recall_at_100", metrics["Val Recall_at_100"][-1]

    avg_metrics = {name: np.mean(values) for name, values in metrics.iteritems()} 
    return disease, avg_metrics

def output_results(all_results):
    output_results = ExperimentResults()
    for disease, metrics in all_results:  
        output_results.add_disease_row(disease.id, disease.name)
        output_results.add_data_row_multiple(disease.id, metrics)
    output_results.add_statistics()
    output_results.output_to_csv(OUTPUT_FILENAME)


if __name__ == "__main__":
    print("Disease Protein Prediction with k-hop Neighborhoods")
    print("Sabri Eyuboglu and Pierce Freeman -- Stanford University")
    print("============================================================")
    print "Loading Diseases..."
    diseases = load_diseases()
    print "Loading PPI Network..."
    ppi_network, ppi_adj, protein_to_node = load_network() 
    n_nodes = ppi_adj.shape[0]

    all_results = []
    
    if MULTIPROCESS:
        p = Pool(N_PROCESSES)
        for n_finished, result in enumerate(p.imap(disease_experiment, diseases.values()), 1):
            all_results.append(result)
            print("Experiment Progress: {} -- {}/{}".format(1.0*n_finished/len(diseases), n_finished, len(diseases)))
    else: 
        for i, disease in enumerate(diseases.values(), 1): 
            print i, ": Running Experiment for Disease ", disease.name 
            disease_nodes = np.array([protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node])
            if(len(disease_nodes) < K_FOLDS): 
                continue
            results = disease_experiment(disease)
            all_results.append(results) 
    
    output_results(all_results)
