"""Draw subgraph metrics"""

import argparse
import logging
import os
import csv

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt 
import seaborn as sns

from output import ExperimentReader
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
    logging.info(" Generator")
    logging.info("Sabri Eyuboglu  -- SNAP Group")
    logging.info("======================================")

    prepare_sns(sns, params)
     
    experiment_reader = ExperimentReader(os.path.join(params.exp_dir, 'subgraph_metrics.csv'))
    for metric in params.metrics:
        header = metric
        col = experiment_reader.get_col(header)
        plt.hist(col, params.n_bins, range=params.range,  label = header, alpha=params.alpha)

        plt.ylabel(params.y_label)
        plt.xlabel(params.x_label)
        if params.legend:
            plt.legend()
        print(metric.lower())
        plt.savefig(os.path.join(args.experiment_dir, metric.lower() + ".pdf"))