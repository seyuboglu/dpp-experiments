"""Draw recall_curve"""

import argparse
import logging
import os
import csv
from multiprocessing import Pool


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import seaborn as sns

#from disease import load_diseases, load_network

from scipy.stats import rankdata

from util import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")

if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.update(json_path)  

    # Set the logger
    set_logger(os.path.join(args.experiment_dir, 'experiment.log'), level=logging.INFO, console=True)

    # Log Title 
    logging.info("Recall-Curve Generator")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    sns.set_style("whitegrid")
    for name, method_exp_dir in params.method_exp_dirs.items():
        ranks  = [] 
        with open(os.path.join(method_exp_dir, 'ranks.csv'), 'r') as ranks_file:
            rank_reader = csv.reader(ranks_file)
            for i, row in enumerate(rank_reader):
                if i == 0: continue
                ranks.extend(map(float, row[2:]))
        ranks = np.array(ranks).astype(int)
        rank_bin_count = np.bincount(ranks)
        recall_curve = np.cumsum(rank_bin_count) / len(ranks)
        plt.plot(recall_curve[:250], label = name)
    plt.title("Protein Rankings across DPP Methods")
    plt.ylabel("Number of Proteins")
    plt.xlabel("Protein Rank")
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'recall_curve_250.jpg'))
