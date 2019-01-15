"""Run experiment"""

import argparse
import logging
import os
import datetime
import time
from multiprocessing import Pool
from collections import defaultdict

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


def compute_pvalue(result, null_results):
    """
    """
    null_results = np.array(null_results)
    return np.logical_or((null_results > result), 
                            np.isclose(null_results, result)).mean()


def build_degree_buckets(network, min_len=500):
    """
    Buckets nodes by degree such that no bucket has less than min_size nodes.
    args:
        min_len (int)  minimum bucket size
    return:
        degree_to_bucket  (dict)    map from degree to the corresponding bucket
    """
    degrees = np.array(network.degree())[:, 1]

    # build degree to buckets
    degree_to_buckets = defaultdict(list)
    max_degree = np.max(degrees)
    for node, degree in enumerate(degrees):
        degree_to_buckets[degree].append(node)

    # enforce min_len
    curr_bucket = None
    prev_bucket = None
    curr_degrees = []
    for degree in range(max_degree + 1):
        # skip nonexistant degrees
        if degree not in degree_to_buckets:
            continue
        
        curr_degrees.append(degree)

        # extend current bucket if necessary
        if curr_bucket is not None:
            curr_bucket.extend(degree_to_buckets[degree])
            degree_to_buckets[degree] = curr_bucket
        else: 
            curr_bucket = degree_to_buckets[degree]
            
        if(len(curr_bucket) >= min_len):
            prev_bucket = curr_bucket
            curr_bucket = None
            curr_degrees = []

    if curr_bucket is not None and prev_bucket is not None and len(curr_bucket) < min_len:
        prev_bucket.extend(curr_bucket)
        for degree in curr_degrees:
            degree_to_buckets[degree] = prev_bucket

    return degree_to_buckets


class LOOAnalysis(Experiment):
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
        super(LOOAnalysis, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), level=logging.INFO, console=True)

        # unpack parameters 
        self.ppi_matrices = {name: np.load(file) for name, file in self.params.ppi_matrices.items()}
        self.exclude = set(self.params.exclude) if hasattr(self.params, "exclude") else set()

        # Log title 
        logging.info("Metric Significance of Diseases in the PPI Network")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, 
                                      self.params.disease_subset,
                                      exclude_splits=['none'])
    
    def get_null_nodes(self, node, quantity=1):
        """
        """
        degree = self.ppi_networkx.degree[node]
        return np.random.choice(self.degree_to_bucket[degree], 
                                size=quantity, 
                                replace=True)
    
    def frac_direct_interactions(self, test_node, train_nodes):
        """
        """
        return np.sum(self.ppi_adj[test_node, :][train_nodes]) / len(train_nodes)

    def frac_common_interactions(self, test_node, train_nodes):
        """
        """
        return 1.0*np.count_nonzero(np.dot(self.ppi_adj[test_node, :], 
                      self.ppi_adj[train_nodes, :].T)) / len(train_nodes)
    
    def common_interactor_score(self, test_node, train_nodes):
        """
        """
        return np.sum(self.ppi_matrices["CI"][test_node, train_nodes])

    def process_disease(self, disease):
        """
        Generates null model for disease and computes 
        Args:
            disease (Disease) the current disease 
        """
        indices = []
        results = []
        disease_pathway = disease.to_node_array(self.protein_to_node)
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(disease_pathway):
            train_nodes = disease_pathway[train_index]
            test_node = disease_pathway[test_index][0]

            indices.append((disease.id, self.node_to_protein[test_node]))
            metrics = {}
            metrics["degree"] = self.ppi_networkx.degree[test_node]

            null_nodes = self.get_null_nodes(test_node, 
                                             quantity=self.params.n_random_nodes)
            
            for metric_fn in self.params.metric_fns:
                result = getattr(self, metric_fn)(test_node, train_nodes)
                null_results = [getattr(self, metric_fn)(null_node, train_nodes)
                                for null_node in null_nodes]
                p_value = compute_pvalue(result, null_results)
                metrics[metric_fn] = result
                metrics["{}_pvalue".format(metric_fn)] = p_value
                
            results.append(metrics)
        return indices, results
    
    def _run(self):
        """
        Run the experiment.
        """
        logging.info("Loading Network...")
        self.ppi_networkx, self.ppi_adj, self.protein_to_node = load_network(
            self.params.ppi_network) 
        self.node_to_protein = {node: protein for protein, node in 
                                self.protein_to_node.items()}

        logging.info("Loading PPI Matrices...")
        self.ppi_matrices = load_ppi_matrices(self.params.ppi_matrices)

        logging.info("Building Degree Buckets...")
        self.degree_to_bucket = build_degree_buckets(self.ppi_networkx,
                                                     min_len=self.params.min_bucket_len)
        for degree, bucket in self.degree_to_bucket.items():
            print("Degree: {}, Size: {}".format(degree, len(bucket)))

        logging.info("Running Experiment...")
        self.results = []
        self.indices = []

        if self.params.n_processes > 1:
            with tqdm(total=len(self.diseases)) as t: 
                p = Pool(self.params.n_processes)
                for indices, results in p.imap(process_disease_wrapper, 
                                               self.diseases.values()):
                    self.indices.extend(indices)
                    self.results.extend(results)
                    t.update()
        else:
            with tqdm(total=len(self.diseases)) as t: 
                for disease in self.diseases.values():
                    indices, results = self.process_disease(disease)
                    self.indices.extend(indices)
                    self.results.extend(results)
                    t.update()

        index = pd.MultiIndex.from_tuples(self.indices, names=['disease', 'protein'])
        self.results = pd.DataFrame(self.results, index=index)
    
    def summarize_results(self):
        """
        Creates a dataframe summarizing the results across
        diseases. 
        return:
            summary_df (DataFrame)
        """
        summary_df =  self.results.describe()

        frac_significant = {} 
        for col_name in self.results:
            if "pvalue" not in col_name:
                continue
            col = self.results[col_name]
            frac_significant[col_name] = np.mean(np.array(col) <= 0.05)
        summary_df = summary_df.append(pd.Series(frac_significant, name="<= 0.05"))

        return summary_df
    
    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'results.csv'), index=True)

        if summary:
            summary_df = self.summarize_results()
            summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
    
    def load_results(self):
        """
        Loads the results from a csv to a pandas Data Frame.
        """
        print("Loading Results...")
        self.results = pd.read_csv(os.path.join(self.dir, 'results.csv'), index_col=[0,1])

    def plot_full_distribution(self, name, metrics, 
                               plot_type="bar", xlabel="", ylabel="",
                               yscale="linear", bins=100,
                               xmin=0.0, xmax=1.0):
        """
        """
        for metric_name in metrics:
            metric = np.array(self.results[metric_name])
            if plot_type == "bar":
                sns.distplot(metric, bins=bins, kde=False, 
                            hist_kws={'range': (xmin, xmax)}, 
                            label=metric_name)
                plt.ylabel("Associations [count{}]".format(r' $\log_{10}$' 
                                                        if yscale == "log" 
                                                        else ""))

            elif plot_type == "kde":
                sns.kdeplot(metric, shade=True, kernel="gau", clip=(0, 1), 
                            label=metric_name)
                plt.ylabel("Associations [KDE{}]".format(r' $\log_{10}$' 
                                                        if yscale == "log" 
                                                        else ""))
                plt.yticks([])

            elif plot_type == "bar_kde":
                sns.distplot(metric, bins=40, kde=True, 
                            kde_kws={'clip': (xmin, xmax)}, label=metric_name)
                plt.ylabel("Associations [count{}]".format(r' $\log_{10}$' 
                                                        if yscale == "log" 
                                                        else ""))
            
            elif plot_type == "":
                pass
        
        plt.xlabel(xlabel)
        sns.despine()
        plt.xticks(np.arange(0.0, 1.0, 0.05))
        if plot_type == "kde": 
            plt.yticks()
        plt.legend()
        # plt.tight_layout()
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.yscale(yscale)

        time_string = datetime.datetime.now().strftime("%m-%d_%H%M")
        plot_path = os.path.join(self.figures_dir, 
                                 '{}_{}_'.format(name, 
                                                    yscale) + time_string + '.pdf')
        plt.show()
        plt.savefig(plot_path)
        plt.close()
        plt.clf()

    def plot_results(self):
        """
        Plots the results 
        """
        print("Plotting Results...")
        prepare_sns(sns, self.params)
        self.figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
        
        for plot_name, params in self.params.plots_to_params.items():
            plot_fn = params["plot_fn"]
            del params["plot_fn"]
            getattr(self, plot_fn)(name=plot_name, **params)


def process_disease_wrapper(disease):
    return exp.process_disease(disease)


if __name__ == "__main__":
    args = parser.parse_args()
    exp = LOOAnalysis(args.dir)
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()