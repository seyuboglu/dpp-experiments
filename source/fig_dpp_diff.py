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

from data import load_diseases, is_disease_id

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
    logging.info("DPP-Diff Generator")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    prepare_sns(sns, params)

    diseases_dict = load_diseases(params.diseases_path, params.disease_subset)

    method_to_scores = {}
    for method_name, method_exp_dir in params.method_exp_dirs.items():
        method_to_scores[method_name]  = {} 
        with open(os.path.join(method_exp_dir, 'metrics.csv'), 'r') as metrics_file:
            metrics_reader = csv.DictReader(metrics_file)
            print(method_exp_dir)
            for i, row in enumerate(metrics_reader):
                if row[params.metric] == params.metric: continue
                if not is_disease_id(row["Disease ID"]): continue 
                if (diseases_dict[row["Disease ID"]].split == "none"):
                        continue 
                method_to_scores[method_name][row["Disease ID"]] = float(row[params.metric])
    
    ref_name = params.reference_exp
    ref_scores = method_to_scores[ref_name]

    mean_abs_diffs = {}

    for method_name, method_scores in method_to_scores.items():
        # Skip comparing reference to itself
        if method_name == ref_name: 
            continue

        # Compute differences in scores
        disease_to_diffs = {key: ref_scores[key] - method_scores[key] 
                            for key in ref_scores.keys()}
        diffs = np.sort(np.array(list(disease_to_diffs.values())))

        mean_abs_diffs[method_name] = np.mean(np.abs(diffs))

        # Split into positive and negative 
        # diffs = diffs * 100
        ref_start = np.where(diffs > 0.0)[0][0]
        method_end = np.where(diffs < 0.0)[-1][-1]

        plt.plot(np.arange(ref_start, len(diffs)), diffs[ref_start:],
                 label=ref_name, alpha=0.6)
        plt.fill_between(np.arange(ref_start, len(diffs)), 
                         0, diffs[ref_start:], alpha=0.6)
        plt.plot(np.arange(method_end), diffs[:method_end], 
                 label=method_name, alpha=0.6)
        plt.plot(np.arange(len(diffs)), np.zeros(len(diffs)), color='k', linewidth=0.8)
        plt.fill_between(np.arange(method_end), 0, diffs[:method_end], alpha=0.6)

        plt.ylabel("Difference in recall-at-100") 
        plt.xlabel("Diseases sorted by difference")
        plt.legend()

        sns.despine()

        plt.savefig(os.path.join(args.experiment_dir, ref_name + "-" + method_name + ".pdf"))
        plt.clf()
    
    plt.bar(np.arange(len(mean_abs_diffs.values())), mean_abs_diffs.values())
    plt.xticks(np.arange(len(mean_abs_diffs.values())), mean_abs_diffs.keys())
    plt.ylabel("Mean Absolute Difference in Recall-at-100")
    plt.xlabel("Methods")
    plt.savefig(os.path.join(args.experiment_dir, "mean_absolute_diff.pdf"))
    plt.clf()
