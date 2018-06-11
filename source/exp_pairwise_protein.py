"""
Provides methods for generating matrices that describe pairwise relationships 
between proteins in the protein-protein interaction network. 
"""

from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import networkx as nx

from disease import Disease, load_diseases, load_network
from output import ExperimentResults
from diamond import DIAMOnD

PPI_COMP_PATH = "data/ppi_matrices/ppi_comp_sqrt_qnorm.npy"
OUTPUT_PATH = "results/comp_results.csv"

#Complementarity Analysis
#====================================================
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
            ranked_flat = ranked_flat[-top_k:]
        codisease_probs = []
        for i, bucket_indices in enumerate(np.array_split(ranked_flat, n_buckets)):
            codisease_prob = 1.0*np.count_nonzero(codisease_flat[bucket_indices])/bucket_indices.size 
            codisease_probs.append(codisease_prob)
        percentiles = np.linspace(100.0 - (1.0*top_k/len(codisease_flat)), 100.0, n_buckets)
        plt.plot(percentiles, codisease_probs, label = name)
    plt.title('Co-disease Probability vs. Score')
    plt.legend()
    plt.ylabel('Prob. Protein Pair Share Disease')
    plt.xlabel('Percentile for Pairwise Protein Metric')
    plt.savefig('figures/pairwise_16b_20000t_n_qn_sq_l3')    

if __name__ == "__main__":
    print("Complementarity of Disease Pathways in PPI Network")
    print("Sabri Eyuboglu  -- Stanford University")
    print("======================================")

    print("Loading PPI Complementarity...") 
    ppi_comp = np.load(PPI_COMP_PATH)
    print("Loading PPI Network...")
    ppi_network, ppi_adj, protein_to_node = load_network() 
    ppi_networkx = nx.from_numpy_matrix(ppi_adj)

    diseases = load_diseases()
    ppi_comp_sqrt = np.load("data/ppi_matrices/ppi_comp_sqrt.npy")
    ppi_comp_sqrt_qnorm = np.load("data/ppi_matrices/ppi_comp_sqrt_qnorm.npy")
    ppi_comp_l3= np.load("data/ppi_matrices/ppi_l3.npy")
    ppi_dn_norm = np.load("data/ppi_matrices/dn_norm.npy")

    pairwise_disease_protein_analysis([('Comp. Sqrt Query Normalized', ppi_comp_sqrt_qnorm),
                                       ('Length-3 Normalized', ppi_comp_l3),
                                       ('Direct Neighbor Normalized', ppi_dn_norm)], 
                                       diseases, n_buckets = 8, top_k = 20000)




