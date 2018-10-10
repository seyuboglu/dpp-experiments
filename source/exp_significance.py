"""Run experiment"""

import argparse
import logging
import os
import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, rankdata
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data import load_diseases, load_network
from exp import Experiment

from util import Params, set_logger, prepare_sns

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

def sort_by_degree(network):
    """
    Sort and argsort nodes by degree. Note: assumes that node numbers match index.  
    args:
        network (networkx graph)
    return:
        node_sorted_by_deg (ndarray) List of all nodes sorted by degree
        node_to_rank_by_deg (ndarray) Argsort of nodes by degree 
    """
    # get degrees and nodes 
    degrees =  np.array(network.degree())[:, 1]
    nodes_sorted_by_deg = degrees.argsort()
    nodes_ranked_by_deg = rankdata(degrees, method='ordinal') - 1

    return nodes_sorted_by_deg, nodes_ranked_by_deg

def get_pathway_scores(ppi_matrix, disease_pathway):
    """
    Gets the scores between nodes in a disease pathway.
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray) 
    """
    inter_pathway = ppi_matrix[disease_pathway, :][:, disease_pathway]

    # ignore diagonal
    above_diag = inter_pathway[np.triu_indices(len(disease_pathway))]
    below_diag = inter_pathway[np.tril_indices(len(disease_pathway))]
    pathway_scores = np.concatenate((above_diag, below_diag))
    return pathway_scores

class SignficanceExp(Experiment):
    """
    Class for running experiment that assess the significance of a network metric
    between disease proteins. Uses the metghod described in Guney et al. for generating
    random subgraph. 
    """
    def __init__(self, dir):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super(SignficanceExp, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # unpack parameters 
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}

        # Log title 
        logging.info("Metric Significance of Diseases in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
         
        logging.info("Loading Network...")
        self.ppi_networkx, self.ppi_adj, self.protein_to_node = load_network(self.params.ppi_network) 

        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, self.params.disease_subset)

        logging.info("Loading PPI Matrices...")
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}

        logging.info("Sorting Nodes by Degree...")
        self.nodes_sorted_by_deg, self.nodes_ranked_by_deg = sort_by_degree(self.ppi_networkx)
    
    def get_null_pathways(self, pathway, quantity = 1, stdev = 25):
        """
        Given a reference pathway, generate quantity 
        """
        null_pathways = [set() for _ in range(quantity)] 
        for node in pathway:
            node_rank = self.nodes_ranked_by_deg[node]
            a = (0 - node_rank ) / stdev
            b = (len(self.nodes_sorted_by_deg) - node_rank) / stdev
            rank_dist = truncnorm(a, b, loc = node_rank, scale = stdev)
            for null_pathway in null_pathways:
                while True:
                    sample_rank = int(rank_dist.rvs())
                    sample_node = self.nodes_sorted_by_deg[sample_rank]
                    print("Node:", node)
                    print("Node Degree:", self.ppi_networkx.degree(node))
                    print("Node_Rank:", node_rank)
                    print("Sample_Rank:", sample_rank)
                    print("Sample_Node:", sample_node)
                    print("Sample Node Degree:", self.ppi_networkx.degree(sample_node))
                    print("________")
                    # guarantee that the same node is not added twice 
                    if sample_node not in null_pathway:
                        null_pathway.add(sample_node)
                        break
        print("Pathway:", np.mean([self.ppi_networkx.degree(node) for node in pathway]))
        for null_pathway in null_pathways:
            print("Null Pathway:", np.mean([self.ppi_networkx.degree(node) for node in null_pathway]))
        return map(list, null_pathways) 

    def process_disease(self, disease):
        """
        Generates null model for disease and computes 
        Args:
            disease (Disease) the current disease 
        """
        disease_pathway = disease.to_node_array(self.protein_to_node)
        null_pathways = self.get_null_pathways(disease_pathway, 
                                               self.params.n_random_pathways, 
                                               self.params.sd_sample)
        results = {"disease_id": disease.id,
                   "disease_name": disease.name}
        for name, ppi_matrix in self.ppi_matrices.items():
            disease_mean = np.mean(get_pathway_scores(ppi_matrix, disease_pathway))
            null_means = np.array([np.mean(get_pathway_scores(ppi_matrix, null_pathway)) 
                                     for null_pathway in null_pathways])
            null_means_mean = null_means.mean()
            null_means_std = null_means.std()
            disease_zscore = 1.0 * (disease_mean - null_means_mean) / null_means_std
            results.update({"disease_zscore_" + name: disease_zscore,
                            "disease_median_" + name: disease_mean,
                            "null_medians+" + name: null_means}) 
        return results
    
    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Running Experiment...")
        self.results = []

        if self.params.n_processes > 1:
            with tqdm(total=len(self.diseases)) as t: 
                p = Pool(self.params.n_processes)
                for results in p.imap(self.diseases.values()):
                    self.results.append(results)
                    t.update()
        else:
            with tqdm(total=len(self.diseases)) as t: 
                for disease in self.diseases.values():
                    results  = self.process_disease(disease)
                    self.results.append(results)
                    t.update()
    
    def save_results(self):
        results_df = pd.DataFrame(self.results)
        summary_df =  results_df.describe()
        results_df.to_csv(os.path.join(self.dir, 'results.csv'))
        summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
    
    def load_results(self):
        self.results.read_csv(os.path.join(self.dir, 'results.csv'))
    
def process_disease_wrapper(disease):
    return exp.run_disease(disease)

if __name__ == "__main__":
    np.seterr(all='raise')
    args = parser.parse_args()
    exp = SignficanceExp(args.experiment_dir)
    if exp.run():
        exp.save_results()