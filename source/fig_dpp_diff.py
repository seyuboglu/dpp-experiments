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

from util import Params, set_logger, prepare_sns

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

    prepare_sns(sns, params)

    method_to_scores = {}
    for method_name, method_exp_dir in params.method_exp_dirs.items():
        method_to_scores[method_name]  = {} 
        with open(os.path.join(method_exp_dir, 'metrics.csv'), 'r') as metrics_file:
            metrics_reader = csv.DictReader(metrics_file)
            for i, row in enumerate(metrics_reader):
                method_to_scores[method_name][row["Disease ID"]] = float(row[params.metric])
    
    ref_name = params.reference_exp
    ref_scores = method_to_scores[ref_name]

    mean_abs_diffs = {}

    for method_name, method_scores in method_to_scores.items():
        # Skip comparing reference to itself
        if method_name == ref_name: continue

        # Compute differences in scores
        disease_to_diffs = {key: ref_scores[key] - method_scores[key] for key in ref_scores.keys()}
        diffs = np.sort(np.array(list(disease_to_diffs.values())))

        mean_abs_diffs[method_name] = np.mean(np.abs(diffs))

        # Split into positive and negative 
        diffs = diffs * 100
        mid = np.where(diffs > 0.0)[0][0]
        plt.bar(np.arange(mid, len(diffs)), diffs[mid:], 1.5, label = ref_name, alpha = 0.75)
        plt.bar(np.arange(mid), diffs[:mid], 1.5, label = method_name,  alpha = 0.75)
        plt.title(ref_name + " " + params.metric + " Gain over " + method_name )
        plt.ylabel(params.metric + " Difference")
        plt.xlabel("Diseases Sorted by Difference")
        plt.legend()
        plt.savefig(os.path.join(args.experiment_dir, ref_name + "-" + method_name + ".pdf"))
        plt.clf()
    
    plt.bar(np.arange(len(mean_abs_diffs.values())), mean_abs_diffs.values())
    plt.xticks(np.arange(len(mean_abs_diffs.values())), mean_abs_diffs.keys())
    plt.ylabel("Mean Absolute Difference in Recall-at-100")
    plt.xlabel("Methods")
    plt.savefig(os.path.join(args.experiment_dir, "mean_absolute_diff.pdf"))
    plt.clf()
