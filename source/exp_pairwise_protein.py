"""
Provides methods for generating matrices that describe pairwise relationships 
between proteins in the protein-protein interaction network. 
"""
import argparse
import logging
import os
from collections import defaultdict
import datetime
import pickle

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.signal import savgol_filter

from exp import Experiment
from data import Disease, load_diseases, load_network
from output import ExperimentResults
from util import set_logger, parse_id_rank_pair, prepare_sns


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

class CodiseaseProbExp(Experiment):
    """
    Class for the codisease probability experiment. 
    """
    def __init__(self, dir):
        super(CodiseaseProbExp, self).__init__(dir)

        # Set the logger
        
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # Log title 
        logging.info("Co-disease probability in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
         
        logging.info("Loading Network...")
        self.ppi_networkx, self.ppi_adj, self.protein_to_node = load_network(self.params.ppi_network) 

        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, self.params.disease_subset)

        # unpack params
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}
        self.top_k = self.params.top_k
        self.n_buckets = self.params.n_buckets 
        self.window_length = self.params.window_length
        self.smooth = self.params.smooth
        self.plots = self.params.plots

        if hasattr(self.params, 'codisease_matrix'):
            logging.info("Loading Codisease Matrix...")
            self.codisease_matrix = np.load(self.params.codisease_matrix)
        else:
            logging.info("Building Codisease Matrix...")
            self.codisease_matrix  = self.build_codisease_matrix()

    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Running Experiment...")
        self.results = {}

        codisease_flat = self.codisease_matrix.flatten()
        for name, matrix in self.ppi_matrices.items(): 
            np.fill_diagonal(matrix, 0)  
            scores_flat = matrix.flatten() 
            ranked_flat = np.argsort(scores_flat)
            if self.top_k: 
                ranked_flat = ranked_flat[-self.top_k:]
            codisease_probs = []
            for i, bucket_indices in enumerate(np.array_split(ranked_flat, self.n_buckets)):
                codisease_prob = 1.0*np.count_nonzero(codisease_flat[bucket_indices])/bucket_indices.size 
                codisease_probs.append(codisease_prob)
            percentiles = np.linspace(100.0 - (1.0*self.top_k/len(codisease_flat)), 100.0, self.n_buckets)
            self.results[name] = (percentiles, codisease_probs)
    
    def build_codisease_matrix(self):
        """
        Build an n_nodes x n_nodes 
        Args: 
            diseases_dict (dict) dictionary of diseases
        Return: 
            codisease_matrix (ndarray)
        """
        n_nodes = len(self.protein_to_node.keys())
        codisease_matrix = np.zeros((n_nodes, n_nodes))
        for disease in self.diseases.values():
            disease_nodes = disease.to_node_array(self.protein_to_node)
            codisease_matrix[np.ix_(disease_nodes, disease_nodes)] += 1
        
        np.save(os.path.join("data","disease_data","codisease_"+str(n_nodes)+".npy"), codisease_matrix)
        return codisease_matrix
    
    def load_results(self):
        with open(os.path.join(self.dir, "results.p"), "rb" ) as file: 
            self.results = pickle.load(file)

    def save_results(self): 
        with open(os.path.join(self.dir, "results.p"), "wb" ) as file:
            pickle.dump(self.results, file)
    
    def plot_results(self):
        """
        Outputs the results as a plot
        """
        assert(self.results != None)

        prepare_sns(sns, self.params)
        for name in self.plots:
            _, codisease_probs = self.results[name]
            if self.smooth: 
                codisease_probs = np.maximum(0, savgol_filter(codisease_probs, window_length=self.window_length-1, polyorder=3))
            bucket_size = self.top_k / self.n_buckets
            plt.plot(np.arange(1, len(codisease_probs) * bucket_size + 1, bucket_size), 
                     codisease_probs[::-1], 
                     label = name,
                     linewidth = 2.0 if name == self.params.primary else 1.0)
            #plt.xticks(np.arange(1, len(codisease_probs) * bucket_size, bucket_size))

        plt.legend()
        plt.ylabel('Codisease Probability')
        plt.xlabel('Protein Pair Rank')

        figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        time_string = datetime.datetime.now().strftime("%m-%d_%H%M")
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'codisease_' + time_string + '.pdf'))
        
if __name__ == "__main__":
     # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    exp = CodiseaseProbExp(args.experiment_dir)
    exp.run()
    #exp.load_results()
    exp.save_results()
    exp.plot_results()