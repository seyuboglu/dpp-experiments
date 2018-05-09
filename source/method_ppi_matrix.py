"""
Provides methods for generating matrices that describe pairwise relationships 
between proteins in the protein-protein interaction network. 
"""
import os

from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import snap 
import networkx as nx

from disease import Disease, load_diseases, load_network
from output import ExperimentResults

PPI_COMP_PATH = "data/ppi_matrices/ppi_comp_sqrt_qnorm.npy"
OUTPUT_PATH = "results/comp_results.csv"

#Loading PPI Complementarity Matrix
#====================================================
#ppi_comp = np.load(PPI_COMP_PATH)
#ppi_network, ppi_adj, protein_to_node = load_network() 
#ppi_networkx = nx.from_numpy_matrix(ppi_adj)

#Functions for Computing DPP Scores
#====================================================
def compute_matrix_scores(ppi_matrix, training_ids, params):
    scores = np.mean(ppi_matrix[:, training_ids], axis = 1)
    return scores 

def compute_direct_neighborhood_scores(training_ids):
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    scores = np.mean((ppi_adj*ppi_inv_deg)[:, training_ids], axis = 1)
    return scores 

def compute_comp_scores(training_ids): 
    #Compute score by taking mean of complementarity with training Set 
    scores = np.mean(ppi_comp[:, training_ids], axis = 1)
    return scores

def compute_adj_scores(training_ids): 
    #Compute score by taking mean of adjacency with training Set 
    scores = np.mean(ppi_adj[:, training_ids], axis = 1)
    return scores

def compute_adj_set(training_ids): 
    #Compute score by taking mean of complementarity with training Set 
    set = np.any(ppi_adj[:, training_ids], axis = 1)
    return set, np.sum(set)

#Complementarity Analysis
#====================================================
def disease_comp_analysis(disease_ids, target_id):
    disease_comp = ppi_comp[:, disease_ids]
    target_disease_comp = disease_comp[target_id, :]
    print target_disease_comp
    plt.bar(range(len(target_disease_comp)), target_disease_comp)
    plt.show()

def compute_node_set_comp(node_ids):
    n = node_ids.size
    non_diag_mask = (np.ones((n,n)) - np.diag(np.ones(n))).astype(bool)
    disease_comp = ppi_comp[node_ids, :][:, node_ids][non_diag_mask]
    return np.mean(disease_comp)

def compute_average_comp():
    n = ppi_comp.shape[0]
    non_diag_mask = (np.ones((n,n)) - np.diag(np.ones(n))).astype(bool)
    disease_comp = ppi_comp[non_diag_mask]
    return np.mean(disease_comp)

def build_codisease_matrix(diseases_dict):
    n_nodes = len(protein_to_node.keys())
    codisease_matrix = np.zeros((n_nodes, n_nodes))
    for disease in diseases_dict.values():
        disease_nodes = np.array([protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node])
        codisease_matrix[np.ix_(disease_nodes, disease_nodes)] += 1
    return codisease_matrix

def plot_pairwise_disease_protein_analysis(results):
    data = []
    for name, codisease_probs in results: 
        line = go.Scatter(x = codisease_probs)
        data.append(line)
    py.iplot(data, filename = 'figures/pairwise_disease_protein_analysis')

def pairwise_disease_protein_analysis(scores_matrices, diseases_dict, n_buckets = 1000, top_k=None):
    codisease_matrix = build_codisease_matrix(diseases_dict)
    codisease_flat = codisease_matrix.flatten()
    for name, scores_matrix in scores_matrices: 
        scores_flat = scores_matrix.flatten() 
        ranked_flat = np.argsort(scores_flat)
        if top_k: 
            print("TOPK")
            ranked_flat = ranked_flat[-top_k:]
        codisease_probs = []
        for i, bucket_indices in enumerate(np.array_split(ranked_flat, n_buckets)):
            codisease_prob = 1.0*np.count_nonzero(codisease_flat[bucket_indices])/bucket_indices.size 
            codisease_probs.append(codisease_prob)
        plt.plot(codisease_probs, label = name)
    plt.title('Co-disease Probability vs. Score')
    plt.legend()
    plt.ylabel('Prob. Protein Pair Share Disease')
    plt.xlabel('Percentile for Pairwise Protein Metric')
    plt.savefig('figures/pairwise_15b_10000t_n_qn_sq_l3')    

#Functions for building complementarity matrices
#====================================================
def build_ppi_comp_matrix(deg_fn = 'id', row_norm = False, col_norm = False):
    name = 'comp'
    # Build vector of node degrees
    deg_vector = np.sum(ppi_adj, axis = 1, keepdims=True)

    # Apply the degree function
    name += '_' + deg_fn
    if deg_fn == 'log':
        # Take the natural log of the degrees. Add one to avoid division by zero
        deg_vector = np.log(deg_vector) + 1
    elif deg_fn == 'sqrt':
        # Take the square root of the degrees
        deg_vector = np.sqrt(deg_vector) 

    # Take the inverse of the degree vector
    inv_deg_vector = np.power(deg_vector, -1)

    # Build the complementarity matrix
    comp_matrix = np.dot((inv_deg_vector*ppi_adj).T, ppi_adj)

    if(row_norm):
        # Normalize by the degree of the query node. (row normalize)
        name += '_rnorm'
        comp_matrix = inv_deg_vector * comp_matrix
    
    if(col_norm):
        # Normalize by the degree of the disease node. (column normalize)
        name += '_cnorm'
        comp_matrix = (comp_matrix.T * inv_deg_vector).T
    
    np.save(os.path.join('data', 'ppi_matrices', name + ".npy"), comp_matrix)
    return comp_matrix 


def build_ppi_comp():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    ppi_comp = np.dot((ppi_adj*ppi_inv_deg).T, ppi_adj)
    np.save("data/ppi_matrices/ppi_comp.npy", ppi_comp)
    return ppi_comp 

def build_ppi_comp_sqrt():
    ppi_sqrt_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -(0.5))
    ppi_comp = np.dot((ppi_adj*ppi_sqrt_inv_deg).T, ppi_adj)
    np.save("data/ppi_matrices/ppi_comp_sqrt.npy", ppi_comp)
    return ppi_comp 

def build_ppi_comp_sqrt_query_normalized():
    ppi_sqrt_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -(0.5))
    ppi_comp = np.dot((ppi_adj*ppi_sqrt_inv_deg).T, ppi_adj)*ppi_sqrt_inv_deg
    np.save("data/ppi_matrices/ppi_comp_sqrt_qnorm.npy", ppi_comp)
    return ppi_comp 

def build_ppi_comp_query_normalized():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    ppi_comp = np.dot((ppi_adj*ppi_inv_deg).T, ppi_adj)*ppi_sqrt_inv_deg
    np.save("data/ppi_matrices/ppi_comp_qnorm.npy", ppi_comp)
    return ppi_comp 

def build_ppi_comp_sqrt_normalized(): 
    ppi_sqrt_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -(0.5))
    ppi_comp = (np.dot((ppi_adj*ppi_sqrt_inv_deg).T, ppi_adj)*ppi_sqrt_inv_deg).T * ppi_sqrt_inv_deg
    np.save("data/ppi_matrices/ppi_comp_sqrt_norm.npy", ppi_comp)
    return ppi_comp 

def build_dn_normalized():
    ppi_sqrt_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -(0.5))
    dn_norm = (ppi_adj*ppi_sqrt_inv_deg).T * ppi_sqrt_inv_deg
    np.save("data/ppi_matrices/dn_norm.npy", dn_norm)
    return dn_norm

def build_dn_query_normalized():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    dn_query_norm = ppi_adj*ppi_inv_deg
    np.save("data/ppi_matrices/dn_query_norm.npy", dn_query_norm)
    return dn_query_norm

def build_l3():
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    ppi_l3 = np.dot((np.dot((ppi_inv_deg*ppi_adj).T, ppi_adj) * ppi_inv_deg).T, ppi_adj)
    np.save("data/ppi_matrices/ppi_l3.npy", ppi_l3)
    return ppi_l3

def build_l3_query_normalized():
    l3 = np.load("data/ppi_matrices/ppi_l3.npy")
    ppi_inv_deg = np.power(np.sum(ppi_adj, axis = 1, keepdims=True), -1)
    l3_qnorm = l3 *ppi_inv_deg
    np.save("data/ppi_matrices/ppi_l3_qnorm.npy", l3_qnorm)
    return l3_qnorm


#Main
#====================================================
if __name__ == "__main__":
    print("Complementarity of Disease Pathways in PPI Network")
    print("Sabri Eyuboglu  -- Stanford University")
    print("======================================")

    build_ppi_comp_matrix(deg_fn = 'sqrt', row_norm = True, col_norm = False)

