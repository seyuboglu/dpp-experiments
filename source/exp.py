"""
Provides base class for all experiments 
"""
import argparse
import os
import pickle
from util import Params, parse_id_rank_pair

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='experiments/base_model',
                    help="Directory containing params.json")

class Experiment(object):
    """ 
    Base class for all disease protein prediction methods.
    """
    def __init__(self, dir):
        """ Initialize the 
        Args: 
            dir (string) The directory where the experiment should be run
        """
        self.dir = dir 

        # load experiment params
        json_path = os.path.join(dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        params.update(json_path)  
        self.params = params

    def run(self): 
        pass

    def __call__(self): 
        return self.run()

    def load_results(self):
        with open(os.path.join(self.dir, "results.p"), "rb" ) as file: 
            self.results = pickle.load(file)

    def save_results(self): 
        with open(os.path.join(self.dir, "results.p"), "wb" ) as file:
            pickle.dump(self.results, file)

    def output_results(self):
        pass


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.update(json_path)  

    assert(hasattr(params, "source"))
    
    os.system(params.source + '--experiment_dir ' + args.dir)
