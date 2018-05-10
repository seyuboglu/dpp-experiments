# PPI 
# Provides classes and methods for reading in ppi network and disease_pathways 

from sets import Set 
import csv
import numpy as np 

NETWORK_PATH = "data/bio-pathways-network.txt"
ASSOCIATIONS_PATH = "data/bio-pathways-associations.csv"
MOTIF_PATH = "data/bio-pathways-proteinmotifs.csv"

class Disease: 
    def __init__(self, id, name, proteins):
        self.id = id
        self.name = name
        self.proteins = proteins
    
    def to_node_array(self, protein_to_node):
        """Translates the diseases protein list to an array of node ids using
        the protein_to_node dictionary.
        """
        return np.array([protein_to_node[protein] for protein in self.proteins if protein in protein_to_node])

def load_diseases(disease_associations_path = ASSOCIATIONS_PATH, diseases_subset = []): 
    diseases_dict = {} 
    with open(disease_associations_path) as associations_file:
        reader = csv.DictReader(associations_file)
        for row in reader:
            disease_id = row["Disease ID"]
            if(diseases_subset and disease_id not in diseases_subset):
                continue  
            disease_name = row["Disease Name"]
            disease_proteins = Set([int(a.strip()) for a in row["Associated Gene IDs"].split(",")])
            diseases_dict[disease_id] = Disease(disease_id, disease_name, disease_proteins)
    return diseases_dict 

def build_codisease_matrix(diseases_dict, protein_to_node):
    n_nodes = len(protein_to_node.keys())
    codisease_matrix = np.zeros((n_nodes, n_nodes))
    for disease in diseases_dict.values():
        disease_nodes = np.array([protein_to_node[protein] for protein in disease.proteins if protein in protein_to_node])
        codisease_matrix[np.ix_(disease_nodes, disease_nodes)] += 1
    return codisease_matrix

def load_snap_network(network_path = NETWORK_PATH):
    protein_ids = set()
    with open(network_path) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]
            protein_ids.add(p1)
            protein_ids.add(p2)
    node_mapping = {protein_id:i for i, protein_id in enumerate(protein_ids)}
    network = snap.TUNGraph.New() 
    adj = np.zeros((len(node_mapping), len(node_mapping)))
    with open(network_path) as network_file:
        for line in network_file:
            p1, p2 = [int(a) for a in line.split()]
            n1, n2 = node_mapping[p1], node_mapping[p2]
            adj[n1,n2] = 1
            adj[n2,n1] = 1
            if(not network.IsNode(n1)): network.AddNode(n1)
            if(not network.IsNode(n2)): network.AddNode(n2) 
            network.AddEdge(n1, n2)
    return network, adj, {protein_id:i for i, protein_id in enumerate(protein_ids)}

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

