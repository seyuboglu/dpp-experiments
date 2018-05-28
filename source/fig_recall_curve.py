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

from util import Params, set_logger, parse_id_rank_pair

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
    sns.set_palette([sns.xkcd_rgb["bright red"]] + sns.color_palette("GnBu_d"))
    recall_curves = {}
    for name, method_exp_dir in params.method_exp_dirs.items():
        if(params.by_disease):
            recall_curve_sum = np.zeros(params.length)
            with open(os.path.join(method_exp_dir, 'ranks.csv'), 'r') as ranks_file:
                rank_reader = csv.reader(ranks_file)
                for i, row in enumerate(rank_reader):
                    if i == 0: continue
                    ranks = [parse_id_rank_pair(rank_str)[1] for rank_str in row[2:]]
                    ranks = np.array(ranks).astype(int)
                    rank_bin_count = np.bincount(ranks)
                    recall_curve = np.cumsum(rank_bin_count) / len(ranks)
                    if len(recall_curve) < params.length:
                        recall_curve = np.pad(recall_curve, 
                                              (0, params.length - len(recall_curve)), 
                                              'edge')
                    recall_curve_sum += recall_curve[:params.length]
            recall_curve = 100*(recall_curve_sum/(i+1))
            recall_curves[name] = recall_curve
            plt.plot(recall_curve, label = name)
        else: 
            ranks = []
            with open(os.path.join(method_exp_dir, 'ranks.csv'), 'r') as ranks_file:
                rank_reader = csv.reader(ranks_file)
                for i, row in enumerate(rank_reader):
                    if i == 0: continue
                    ranks.extend(map(float, row[2:]))
            ranks = np.array(ranks).astype(int)
            rank_bin_count = np.bincount(ranks)
            recall_curve = 100*(np.cumsum(rank_bin_count) / len(ranks))
            plt.plot(recall_curve[:params.length], label = name)
        
    # plot percent differences
    for k in params.percent_increase:
        recalls_at_k = [] 
        for name, recall_curve in recall_curves.items():
            recalls_at_k.append(recall_curve[k])
        recalls_at_k.sort(reverse=True)
        plt.plot([k, k], [recalls_at_k[0] - params.offset, recalls_at_k[1] + params.offset], color = 'green', alpha = 0.5)
        percent_inrease = round(100 * (recalls_at_k[0] - recalls_at_k[1]) / recalls_at_k[1], 1)
        plt.text(x = k + 2, y = recalls_at_k[1] + (recalls_at_k[0] - recalls_at_k[1]) / 2 - 0.001, 
                 s = '+' + str(percent_inrease) + '%',
                 fontsize = 9, weight = 'bold', alpha = .75, color='green')
    
    #Plot 
    if(params.title):
        plt.title("Recall-at-K (%) across DPP Methods")

    plt.ylabel("Recall-at-K (%)")
    plt.xlabel("K")
    plt.legend()
    plt.savefig(os.path.join(args.experiment_dir, 'recall_curve_' + str(params.length) + '.jpg'))
