"""Run experiment"""

import argparse
import logging
import os
import csv
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import KFold
import networkx as nx
import scipy.sparse as sp
import scipy.stats as stats
from tqdm import tqdm

from exp import Experiment
from method.ppi_matrix import compute_matrix_scores
from method.random_walk import compute_random_walk_scores, L2RandomWalk
from method.diamond import compute_diamond_scores
#from method.graph_cn import GCN
from method.learned_cn_method import LearnedCN
from method.pathway_expansion import PathwayExpansion
from method.lr import compute_lr_scores, build_embedding_feature_matrix
from method.dns import compute_dns_scores
from data import load_diseases, load_network
from output import ExperimentResults, write_dict_to_csv
from analysis import positive_rankings, recall_at, recall, auroc, average_precision
from util import Params, set_logger, parse_id_rank_pair

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='experiments/base_model',
                    help="Directory containing params.json")

class DPPExperiment(Experiment):
    """
    Class for the disease protein prediction experiment
    """
    def __init__(self, dir):
        """ Initialize the disease protein prediction experiment 
        Args: 
            dir (string) The directory where the experiment should be run
        """
        super(DPPExperiment, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(args.dir, 'experiment.log'), level=logging.INFO, console=True)

        # Log Title 
        logging.info("Disease Protein Prediction in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")

        # Load data from params file
        logging.info("Loading PPI Network...")
        self.ppi_networkx, self.ppi_network_adj, self.protein_to_node = load_network(self.params.ppi_network)
        self.node_to_protein = {node: protein for protein, node in self.protein_to_node.items()}
        logging.info("Loading Disease Associations...")
        self.diseases_dict = load_diseases(self.params.diseases_path, self.params.disease_subset)

        # Load method specific data 
        # TODO: Build class for each method 
        if(self.params.method == "ppi_matrix"):
            logging.info("Loading PPI Matrix...")
            self.ppi_matrix = np.load(self.params.ppi_matrix)
            # normalize columns of ppi_matrix
            if(self.params.normalize):
                if hasattr(self.params, "norm_type"):
                    if self.params.norm_type == "frac":
                        self.ppi_matrix = self.ppi_matrix / np.sum(self.ppi_matrix, 
                                                                   axis=0)
                    elif self.params.norm_type == "zscore":
                        self.ppi_matrix = (self.ppi_matrix - np.mean(self.ppi_matrix, axis=0)) / np.std(self.ppi_matrix, axis=0)
                else:
                    self.ppi_matrix = (self.ppi_matrix - np.mean(self.ppi_matrix, axis=0)) / np.std(self.ppi_matrix, axis=0)
         
            # zero out the diagonal
            np.fill_diagonal(self.ppi_matrix, 0)  

        elif (self.params.method == 'lr'):
            logging.info("Loading Feature Matrices...")
            self.feature_matrices = []
            for features_filename in self.params.features:
                self.feature_matrices.append(
                    build_embedding_feature_matrix(self.protein_to_node, 
                                                   features_filename))
        elif (self.params.method == 'l2_rw'):
            self.method = L2RandomWalk(self.params)

        elif (self.params.method == 'pathway_expansion'):
            self.method = PathwayExpansion(self.params, 
                                           self.ppi_networkx, 
                                           self.ppi_network_adj)
        
        elif (self.params.method == "learned_cn"):
            self.method = LearnedCN(self.dir,
                                    self.params,
                                    self.ppi_network_adj,
                                    self.diseases_dict,
                                    self.protein_to_node)

        elif (self.params.method == 'gcn'):
            self.method = GCN(self.params, self.ppi_network_adj)
  
    def _run(self):
        """ Run the disease protein prediction experiment
        Args: 
            dir (string) The directory where the experiment should be run
        """
        logging.info("Running Experiment...")
        disease_to_metrics, disease_to_ranks = {}, {}
        if self.params.n_processes > 1: 
            p = Pool(self.params.n_processes)
            with tqdm(total=len(self.diseases_dict)) as t:
                for n_finished, (disease, metrics, ranks) in enumerate(p.imap(run_dpp_wrapper, self.diseases_dict.values()), 1):
                    if metrics != None or ranks != None:
                        disease_to_ranks[disease] = ranks 
                        disease_to_metrics[disease] = metrics
                        t.set_postfix(str="{} Recall-at-100: {:.2f}%".format(disease.id, 100 * metrics["Recall-at-100"]))
                    else:
                        t.set_postfix(str="{} Not Recorded".format(disease.id))
                    t.update()
                
        else: 
            with tqdm(total=len(self.diseases_dict)) as t:
                for n_finished, disease in enumerate(self.diseases_dict.values()): 
                    disease, metrics, ranks = self.run_dpp(disease)
                    if metrics != None or ranks != None:
                        disease_to_metrics[disease] = metrics
                        disease_to_ranks[disease] = ranks 
                        t.set_postfix(str="{} Recall-at-100: {:.2f}%".format(disease.id, 100 * metrics["Recall-at-100"]))
                    else:
                        t.set_postfix(str="{} Not Recorded".format(disease.id))
                    t.update()
     
        self.results = {"metrics": disease_to_metrics,
                        "ranks": disease_to_ranks}
   
    def compute_node_scores(self, train_nodes, val_nodes):
        """ Get score 
        Args:
            disease: (Disease) A disease object
        """
        scores = None
        if self.params.method == 'ppi_matrix':
            scores = compute_matrix_scores(self.ppi_matrix, train_nodes, self.params)

        elif self.params.method == 'direct_neighbor':
            scores = compute_dns_scores(self.ppi_network_adj, train_nodes, self.params)

        elif self.params.method == 'random_walk':
            scores = compute_random_walk_scores(self.ppi_networkx, train_nodes, self.params)

        elif self.params.method == 'diamond':
            scores = compute_diamond_scores(self.ppi_networkx, train_nodes, self.params)

        elif self.params.method == 'gcn':
            scores = self.method(train_nodes, val_nodes)

        elif self.params.method == 'lr':
            scores = compute_lr_scores(self.feature_matrices, train_nodes, self.params)

        elif self.params.method == 'l2_rw':
            scores = self.method(train_nodes, val_nodes)

        elif self.params.method == 'pathway_expansion':
            scores = self.method(train_nodes, val_nodes)
        
        elif self.params.method == 'learned_cn':
            scores = self.method(train_nodes, val_nodes)
            
        else:
            logging.error("No method " + self.params.method)
        return scores


    def run_dpp(self, disease):
        """ Perform k-fold cross validation on disease protein prediction on disease
        Args:
            disease: (Disease) A disease object
        """
        # create directory for disease 
        disease_directory = os.path.join(args.dir, 'diseases', disease.id)
        if not os.path.exists(disease_directory):
            os.makedirs(disease_directory)

        disease_nodes = disease.to_node_array(self.protein_to_node)
        # Ensure that there are at least 2 proteins
        if disease_nodes.size <= 1:
            return disease, None, None 
        labels = np.zeros((len(self.protein_to_node), 1))
        labels[disease_nodes, 0] = 1 
        metrics = {}

        if getattr(self.params, 'saliency_map', False): 
            rank_corrs = []
            p_values = []

        # Perform k-fold cross validation
        n_folds = disease_nodes.size if self.params.n_folds < 0 or self.params.n_folds > len(disease_nodes) else self.params.n_folds
        kf = KFold(n_splits = n_folds, shuffle=False)

        for train_indices, test_indices in kf.split(disease_nodes):
            train_nodes = disease_nodes[train_indices]
            val_nodes = disease_nodes[test_indices]

            # compute node scores 
            scores = self.compute_node_scores(train_nodes, val_nodes)

            # compute the metrics of target node
            compute_metrics(metrics, labels, scores, train_nodes, val_nodes)

            # compute saliency maps
            if getattr(self.params, 'saliency_map', False): 
                rank_corr, p_value = self.method.analyze_saliency_maps(disease_directory, self.node_to_protein)
                rank_corrs.extend(rank_corr)
                p_values.extend(p_value)

        avg_metrics = {name: np.mean(values) for name, values in metrics.items()}
        proteins = [self.node_to_protein[node] for node in metrics["Nodes"]]
        ranks = metrics["Ranks"]

        if getattr(self.params, 'saliency_map', False):
            avg_metrics["GCN-Comp Rank Correlation"] = np.mean(rank_corrs)
            print(avg_metrics["GCN-Comp Rank Correlation"])
            avg_metrics["GCN-Comp P-Value"] = stats.combine_pvalues(p_values)[1]

        proteins_to_ranks = {protein: ranks for protein, ranks in zip(proteins, ranks)}
        return disease, avg_metrics, proteins_to_ranks 

    def save_results(self):
        write_metrics(self.dir, self.results["metrics"])
        write_ranks(self.dir, self.results["ranks"])


def compute_metrics(metrics, labels, scores, train_nodes, test_nodes):
    """Synthesize the metrics for one disease. 
    Args: 
        metrics: (dictionary) 
        labels: (ndarray) binary array indicating in-disease nodes
        scores: (ndarray) scores assigned to each node
        train_node: (ndarray) array of train nodes
        test_nodes: (ndarray) array of test nodes
    """
    for k in [100, 25]: 
        metrics.setdefault("Recall-at-{}".format(k), []).append(recall_at(labels, scores, k, train_nodes))
    metrics.setdefault("AUROC", []).append(auroc(labels, scores, train_nodes))
    metrics.setdefault("Mean Average Precision", []).append(average_precision(labels, scores, train_nodes))
    metrics.setdefault("Ranks", []).extend(positive_rankings(labels, scores, train_nodes))
    metrics.setdefault("Nodes", []).extend(test_nodes)


def run_dpp_wrapper(disease):
    return exp.run_dpp(disease)


def write_metrics(directory, disease_to_metrics):
    """Synthesize the metrics for one disease. 
    Args: 
        directory: (string) directory to save results
        disease_to_metrics: (dict)
    """
    # Output metrics to csv
    output_results = ExperimentResults()
    for disease, metrics in disease_to_metrics.items():  
        output_results.add_disease_row(disease.id, disease.name)
        output_results.add_data_row_multiple(disease.id, metrics)
    output_results.add_statistics()
    output_results.output_to_csv(os.path.join(directory, 'metrics.csv'))


def write_ranks(directory, disease_to_ranks):
    """Write out the ranks for the proteins for one . 
    Args: 
        directory: (string) directory to save results
        disease_to_ranks: (dict)
    """
    # Output metrics to csv
    with open(os.path.join(directory, 'ranks.csv'), 'w') as file:
        ranks_writer = csv.writer(file)
        ranks_writer.writerow(['Disease ID', 'Disease Name', 'Protein Ranks', 'Protein Ids'])
        for curr_disease, curr_ranks in disease_to_ranks.items():
            curr_ranks = [str(protein) + "=" + str(rank) for protein, rank in curr_ranks.items()]
            ranks_writer.writerow([curr_disease.id, curr_disease.name] + curr_ranks)


def load_ranks(directory):
    """Load ranks from a rankings file output for one method. 
    """
    disease_to_ranks = {}
    with open(os.path.join(directory, 'ranks.csv'),'r') as file: 
        ranks_reader = csv.reader(file)
        for i, row in enumerate(ranks_reader):
            if (i == 0): 
                continue 
            disease_id = row[0]
            protein_to_ranks = {id: rank for id, rank in map(parse_id_rank_pair, row[2:])}
            disease_to_ranks[disease_id] = protein_to_ranks
        
    return disease_to_ranks


if __name__ == "__main__":
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    exp = DPPExperiment(args.dir)
    if exp.run():
        exp.save_results()
