# PPI 
# Provides classes and methods for reading in ppi network and disease_pathways 
import csv
import random
import os

import numpy as np 
import networkx as nx

NETWORK_PATH = "data/networks/bio-pathways-network.txt"
ASSOCIATIONS_PATH = "data/bio-pathways-associations.csv"

class Disease: 
    def __init__(self, id, name, proteins):
        self.id = id
        self.name = name
        self.proteins = proteins
    
    def to_node_array(self, protein_to_node):
        """ Translates the diseases protein list to an array of node ids using
        the protein_to_node dictionary.
        Args: 
            protein_to_node (dictionary)
        """
        return np.array([protein_to_node[protein] for protein in self.proteins if protein in protein_to_node])

def is_disease_id(str):
    """ Returns bool indicating whether or not the passed in string is 
    a valid disease id. 
    Args: 
        str (string)
    """
    return len(str) == 8 and str[0] == 'C' and str[1:].isdigit()

def split_diseases(split_fractions, path):
    """ Splits a set of disease assocation into data sets i.e. train, test, and dev sets 
    Args: 
        split_fractions (dictionary) dictionary mapping split name to fraction. fractions should sum to 1.0
        path (string) 
    """
    with open(path) as file:
        reader = csv.DictReader(file)
        disease_rows = [row for row in reader if is_disease_id(row["Disease ID"])]
        header = reader.fieldnames
    
    random.seed(360)
    random.shuffle(disease_rows)

    split_rows = {}
    curr_start = 0
    N = len(disease_rows)
    for name, fraction in split_fractions.items():
        curr_end = curr_start + int(N*fraction)
        split_rows[name] = disease_rows[curr_start : curr_end]
        curr_start = curr_end
    
    for name, rows in split_rows.items():
        directory, filename = os.path.split(path)
        split_path = os.path.join(directory, name + '_' + filename)
        with open(split_path, 'w') as file:
            writer = csv.DictWriter(file, header)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    
def load_diseases(associations_path = ASSOCIATIONS_PATH, diseases_subset = []): 
    diseases_dict = {} 
    with open(associations_path) as associations_file:
        reader = csv.DictReader(associations_file)
        for row in reader:
            disease_id = row["Disease ID"]
            if(diseases_subset and disease_id not in diseases_subset):
                continue  
            disease_name = row["Disease Name"]
            disease_proteins = set([int(a.strip()) for a in row["Associated Gene IDs"].split(",")])
            diseases_dict[disease_id] = Disease(disease_id, disease_name, disease_proteins)
    return diseases_dict 

def build_codisease_matrix(diseases_dict, protein_to_node):
    n_nodes = len(protein_to_node.keys())
    codisease_matrix = np.zeros((n_nodes, n_nodes))
    for disease in diseases_dict.values():
        disease_nodes = np.array([protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node])
        codisease_matrix[np.ix_(disease_nodes, disease_nodes)] += 1
    return codisease_matrix

def load_network(network_path = NETWORK_PATH):
    protein_ids = set()
    with open(network_path) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]
            protein_ids.add(p1)
            protein_ids.add(p2)
    protein_to_node = {protein_id: i for i, protein_id in enumerate(protein_ids)}
    adj = np.zeros((len(protein_to_node), len(protein_to_node)))
    with open(network_path) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]
            n1, n2 = protein_to_node[p1], protein_to_node[p2]
            adj[n1,n2] = 1
            adj[n2,n1] = 1
    return nx.from_numpy_matrix(adj), adj, protein_to_node


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir

    # Log Title 
    print("Disease Set Splitting")
    print("Sabri Eyuboglu  -- SNAP Group -- Stanford University")
    print("====================================================")

    print("Splitting diseases...")
    split_fractions = {'val': 0.40,
                       'test': 0.60}
    split_diseases(split_fractions, 'experiments/associations/bio-pathways-associations.csv')
    print("Done.")
