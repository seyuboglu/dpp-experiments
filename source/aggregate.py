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
         
    def initalize_dataframe(self):
        """
        Initializes the dataframe with diseases.
        """
        df = pd.DataFrame()
        for disease_id, disease in self.diseases.items(): 
            df = df.append(pd.Series({"name": disease.name}, name=disease_id))
        return df
        
    def _run(self):
        """
        Run the aggregation.
        """
        print("Running Experiment...")
        df = self.initalize_dataframe() 
        self.results = pd.concat([df] + [self.process_experiment(exp)
                                         for exp in self.experiments],
                                 axis=1,
                                 keys=[exp["name"] <- ISSUE HERE 
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

if __name__ == "__main__":
    args = parser.parse_args()
    exp = Aggregate(args.dir)
    if exp.is_completed():
        exp.load_results()
    elif exp.run():
        exp.save_results()
    exp.plot_results()
