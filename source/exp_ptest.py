"""Run experiment"""

import argparse
import logging
import os
import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm, rankdata
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data import load_diseases, load_network
from exp import Experiment
from method.ppi_matrix import load_ppi_matrices
from util import Params, set_logger, prepare_sns, string_to_list, fraction_nonzero

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='experiments/base_model',
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

def get_pairwise_scores(ppi_matrix, pathway):
    """
    Gets the scores between nodes in a disease pathway.
    args:
        ppi_matrix  (ndarray)
        pathway (ndarray) 
    """
    inter_pathway = ppi_matrix[pathway, :][:, pathway]

    # ignore diagonal
    above_diag = inter_pathway[np.triu_indices(len(pathway))]
    below_diag = inter_pathway[np.tril_indices(len(pathway))]
    pathway_scores = np.concatenate((above_diag, below_diag))
    return pathway_scores

def get_loo_scores(ppi_matrix,  pathway):
    """
    Gets the scores between nodes in a disease pathway.
    args:
        ppi_matrix  (ndarray)
        pathway (ndarray) 
    """
    loo = LeaveOneOut()
    scores = []
    for train_index, test_index in loo.split(pathway):
        train = pathway[train_index]
        test = pathway[test_index]

        node_score = np.sum(ppi_matrix[test, train])
        scores.append(node_score)
    return np.array(scores)
    
def loo_median(ppi_matrix, pathway):
    """
    Computes the median score for each node w.r.t
    the rest of the pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.median(get_loo_scores(ppi_matrix, pathway))

def pairwise_mean(ppi_matrix, pathway):
    """
    Computes average score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.mean(get_pairwise_scores(ppi_matrix, pathway))

def pairwise_median(ppi_matrix, pathway):
    """
    Computes median score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return np.median(get_pairwise_scores(ppi_matrix, pathway))

def pairwise_nonzero(ppi_matrix, pathway):
    """
    Computes fraction nonzero score between nodes in a pathway. 
    args:
        ppi_matrix  (ndarray)
        disease_pathway (ndarray)
    """
    return fraction_nonzero(get_pairwise_scores(ppi_matrix, pathway))

class PermutationTest(Experiment):
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
        super(PermutationTest, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # unpack parameters 
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}

        # Log title 
        logging.info("Metric Significance of Diseases in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, self.params.disease_subset)

        self.metric_fn = globals()[self.params.metric_fn]

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
                
                    # gaurantee that the same node is not added twice 
                    if sample_node not in null_pathway:
                        null_pathway.add(sample_node)
                        break
        
        return map(np.array, (map(list, null_pathways)))

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
                   "disease_name": disease.name,
                   "disease_size": len(disease_pathway)}
        for name, ppi_matrix in self.ppi_matrices.items():
            
            disease_result = self.metric_fn(ppi_matrix, disease_pathway)
            null_results = np.array([self.metric_fn(ppi_matrix, null_pathway) 
                                     for null_pathway in null_pathways])

            disease_pvalue = (null_results > disease_result).mean()
            results.update({"pvalue_" + name: disease_pvalue,
                            self.metric_fn.__name__ + "_" + name: disease_result,
                            "null_" + self.metric_fn.__name__ + "s_" + name: null_results})
            
        return results
    
    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Loading Network...")
        self.ppi_networkx, self.ppi_adj, self.protein_to_node = load_network(self.params.ppi_network) 

        logging.info("Loading PPI Matrices...")
        self.ppi_matrices = load_ppi_matrices(self.params.ppi_matrices)

        logging.info("Sorting Nodes by Degree...")
        self.nodes_sorted_by_deg, self.nodes_ranked_by_deg = sort_by_degree(self.ppi_networkx)

        logging.info("Running Experiment...")
        self.results = []

        if self.params.n_processes > 1:
            with tqdm(total=len(self.diseases)) as t: 
                p = Pool(self.params.n_processes)
                for results in p.imap(process_disease_wrapper, self.diseases.values()):
                    self.results.append(results)
                    t.update()
        else:
            with tqdm(total=len(self.diseases)) as t: 
                for disease in self.diseases.values():
                    results  = self.process_disease(disease)
                    self.results.append(results)
                    t.update()
        self.results = pd.DataFrame(self.results)
    
    def save_results(self, summary = True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        summary_df =  self.results.describe()
        self.results.to_csv(os.path.join(self.dir, 'results.csv'))
        summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
    
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'results.csv'))
    
    def plot_disease(self, disease):
        """
        Plots one disease 
        """
        row = self.results.loc[self.results['disease_id'] == disease.id]
        disease_dir = os.path.join(self.dir, 'figures', 'diseases', disease.id)

        if not os.path.exists(disease_dir):
            os.makedirs(disease_dir)

        for name in self.ppi_matrices.keys():
            null_means = row["null_means_" + name].values[0]

            if type(null_means) == str:
                null_means = string_to_list(null_means, float)

            sns.kdeplot(null_means, shade=True, kernel="gau",  color = "grey", clip=(0,1), label = "Random Pathways")

            disease_mean = row["mean_" + name]
            sns.scatterplot(disease_mean, 0, label = disease.name)

            plt.ylabel('Density (estimated with KDE)')
            plt.xlabel('mean ' + name)
            sns.despine(left = True)

            plt.tight_layout()
            plt.savefig(os.path.join(disease_dir, name + "_mean.pdf"))
            plt.close()
            plt.clf()

    
    def plot_all_diseases(self):
        """
        Estimates the distribution of z-scores for each metric across all diseases
        then plots the estimated distributions on the same plot. 
        """
        for name in self.ppi_matrices.keys(): 
            series = self.results["pvalue_" + name]
            series = np.array(series)
            sns.kdeplot(series, shade=True, kernel="gau", clip=(0,1), label = name)
        
        time_string = datetime.datetime.now().strftime("%m-%d_%H%M")

        figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        plot_path = os.path.join(figures_dir, 'pvalue_dist_' + time_string + '.pdf')
        #plt.xlim(xmax = 30, xmin = -10)

        plt.ylabel('Density (estimated with KDE)')
        plt.xlabel('p-value')
        plt.legend()
        sns.despine(left = True)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plt.clf()

    def plot_results(self):
        """
        Plots the results 
        """
        print("Plotting Results...")
        prepare_sns(sns, self.params)
        self.plot_all_diseases()

        for disease_id in tqdm(self.params.disease_plots):
            self.plot_disease(self.diseases[disease_id])
    
def process_disease_wrapper(disease):
    return exp.process_disease(disease)

if __name__ == "__main__":
    args = parser.parse_args()
    exp = PermutationTest(args.dir)
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()