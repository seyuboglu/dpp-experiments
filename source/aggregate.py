"""Run experiment"""

import argparse
import logging
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data import load_diseases, load_network
from exp import Experiment
from util import Params, set_logger, prepare_sns

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='experiments/base_model',
                    help="Directory containing params.json")


class Aggregate(Experiment):
    """
    
    """
    def __init__(self, dir):
        """
        Constructor 
        Args: 
            dir (string) directory of the experiment to be run
        """
        super(Aggregate, self).__init__(dir)

        # Set the logger
        set_logger(os.path.join(self.dir, 'experiment.log'), 
                   level=logging.INFO, 
                   console=True)

        # Unpack parameters
        self.experiments = self.params.experiments
        self.plots = self.params.plots

        # Log title 
        logging.info("Aggregating Experiments")
        logging.info("Sabri Eyuboglu  -- SNAP Group")
        logging.info("======================================")
        logging.info("Loading Disease Associations...")
        self.diseases = load_diseases(self.params.diseases_path, 
                                      self.params.disease_subset)
    
    def process_experiment(self, exp_dict):
        """
        Dictionary of experiment info. 
        args:
            exp_dict
        """
        df = pd.read_csv(exp_dict["path"],
                         index_col=0)
        return df[exp_dict["cols"]]
        
    def _run(self):
        """
        Run the aggregation.
        """
        print("Running Experiment...")
        self.results = pd.concat([self.process_experiment(exp)
                                  for exp in self.experiments],
                                 axis=1,
                                 keys=[exp["name"]
                                       for exp in self.experiments])
    
    def save_results(self, summary=True):
        """
        Saves the results to a csv using a pandas Data Fram
        """
        print("Saving Results...")
        self.results.to_csv(os.path.join(self.dir, 'results.csv'))

        if summary:
            summary_df = self.summarize_results()
            summary_df.to_csv(os.path.join(self.dir, 'summary.csv'))
    
    def plot_results(self):
        """
        Plot the results 
        """
        logging.info("Plotting results...")

        figures_dir = os.path.join(self.dir, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        prepare_sns(sns, self.params)
        for plot in self.plots:
            
            if plot["type"] == "scatter":
                series_a = self.results[tuple(plot["cols"][0])]
                series_b = self.results[tuple(plot["cols"][1])]
                sns.scatterplot(x=series_a, y=series_b)
            
            elif plot["type"] == "regplot":
                series_a = self.results[tuple(plot["cols"][0])]
                series_b = self.results[tuple(plot["cols"][1])]
                sns.regplot(x=series_a, y=series_b)

            plt.ylabel(plot["y_label"])
            plt.xlabel(plot["x_label"])
            sns.despine(left=True)

            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, plot["name"] + ".pdf"))
            plt.close()
            plt.clf()


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Aggregate(args.dir)
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()
