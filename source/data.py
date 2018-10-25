# PPI 
# Provides classes and methods for reading in ppi network and disease_pathways 
import argparse
import csv
import random
import os

import numpy as np 
import pandas as pd 
import networkx as nx
from goatools.obo_parser import GODag

from method.ppi_matrix import build_ppi_comp_matrix, build_ppi_dn_matrix
from util import print_title

NETWORK_PATH = "data/networks/bio-pathways-network.txt"
ASSOCIATIONS_PATH = "data/bio-pathways-associations.csv"
GENE_NAMES_PATH = "data/protein_data/protein_names.txt"

parser = argparse.ArgumentParser()
parser.add_argument('--job', default='split_diseases',
                    help="which job should be performed")


class Disease: 
    def __init__(self, id, name, proteins, validation_proteins=None):
        """ Initialize a disease. 
        Args:
            id (string) 
            name (string)
            proteins (list of ints)
            valdiation_proteins (list of ints)
        """
        self.id = id
        self.name = name
        self.proteins = proteins
        self.validation_proteins = validation_proteins

        self.doids = []
        self.parents = []
        self.class_doid = None 
        self.class_name = None
    
    def to_node_array(self, protein_to_node, validation=False):
        """ Translates the diseases protein list to an array of node ids using
        the protein_to_node dictionary.
        Args: 
            protein_to_node (dictionary)
        """
        proteins = self.validation_proteins if validation else self.proteins
        return np.array([protein_to_node[protein] for protein in proteins if protein in protein_to_node])


def is_disease_id(str):
    """ Returns bool indicating whether or not the passed in string is 
    a valid disease id. 
    Args: 
        str (string)
    """
    return len(str) == 8 and str[0] == 'C' and str[1:].isdigit()


def load_doid(diseases,
              hdo_path='data/raw/disease_ontology.obo'):
    """
    Adds a doids attribute to disease objects.
    Allows for mapping between mesh ID (e.g. C003548), like those
    used in the association files, and DOID (e.g. 0001816). 
    args:
        hdo_path    (string)
        diseases    (dict)  meshID to disease object 
    """
    with open(hdo_path) as hdo_file:
        for line in hdo_file:
            if line.startswith('id:'):
                doid = line.strip().split('id:')[1].strip()
            elif line.startswith('xref: UMLS_CUI:'):
                mesh_id = line.strip().split('xref: UMLS_CUI:')[1].strip()
                if mesh_id in diseases:
                    diseases[mesh_id].doids.append(doid)


def assign_classes(class_doid, class_diseases, class_doid_to_disease, diseases, min, max):
    """
    """
    if len(class_diseases) < min:
        for disease in class_diseases:
            disease.class_id = None
    elif len(class_diseases) > max:
        for disease_id in class_diseases:
            disease = diseases[disease_id]
            level = disease.parents.index(class_doid) + 1
            if level >= len(disease.parents):
                disease.class_id = None 
                continue
            disease.class_doid = disease.parents[level]
            class_doid_to_diseases.setdefault(class_doid, set()).add(disease.id)
            assign_classes(class_doid, class_diseases, )


def load_disease_classes(diseases, 
                         hdo_path='data/raw/disease_ontology.obo',
                         min_level=2, 
                         min=10,
                         max=100):
    """
    Adds a classes attribute to disease objects.
    """
    obo = GODag(hdo_path)
    load_doid(diseases, hdo_path)

    class_doid_to_diseases = {}
    for disease in diseases.values():
        if not disease.doids: 
            continue
        doid = disease.doids[0]
        disease.parents = [parent for parent in obo[doid].get_all_parents()]
        if len(disease.parents) >= min_level:
            disease.class_doid = disease.parents[min_level-1]
        class_doid_to_diseases.setdefault(disease.class_doid, set()).add(disease.id)

    num_classified = 0
    while len(class_doid_to_diseases) > 0:
        for class_doid, class_diseases in class_doid_to_diseases.items():
            if len(class_diseases) < min:
                for disease_id in class_diseases:
                    disease = diseases[disease_id]
                    disease.class_id = None
                del class_doid_to_diseases[class_doid]
            elif len(class_diseases) > max:
                for disease_id in class_diseases:
                    disease = diseases[disease_id]
                    level = disease.parents.index(class_doid) + 1
                    if level >= len(disease.parents):
                        disease.class_id = None 
                        continue
                    disease.class_doid = disease.parents[level]
                    class_doid_to_diseases.setdefault(disease.class_doid, set()).add(disease.id)
                del class_doid_to_diseases[class_doid]
            else:
                num_classified += len(class_doid_to_diseases[class_doid])
                for disease_id in class_diseases:
                    print(obo[class_doid].name)
                    diseases[disease_id].class_name = obo[class_doid].name
                del class_doid_to_diseases[class_doid]
    
    print("Classified {:.2f}% ({}/{}) of diseases".format(
          100.0 * num_classified / len(diseases),
          num_classified,
          len(diseases)))


def output_diseases(diseases, output_path):
    """
    Output disease objects to csv. 
    args:
        diseases    (dict)
        output_path (string)
    """
    df = pd.DataFrame([{"name": disease.name,
                        "class": "None" if  disease.class_name is None 
                                 else disease.class_name,
                        "size": len(disease.proteins)} 
                       for disease in diseases.values()],
                      index=[disease.id for disease in diseases.values()],
                      columns=["name", "class", "size"])
    df.index.name = "id"
    df.to_csv(output_path)                


def split_diseases(split_fractions, path):
    """ Splits a set of disease assocation into data sets i.e. train, test, and dev sets 
    Args: 
        split_fractions (dictionary) dictionary mapping split name to fraction. fractions 
                                     should sum to 1.0
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
        curr_end = curr_start + int(N * fraction)
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
    

def load_diseases(associations_path=ASSOCIATIONS_PATH, 
                  diseases_subset=[], 
                  gene_names_path=GENE_NAMES_PATH): 
    """ Load a set of disease-protein associations
    Args:
        assoications_path (string)
        diseases_subset (set) 
    Returns:
        diseases (dict)
    """
    diseases = {} 
    total = 0
    with open(associations_path) as associations_file:
        reader = csv.DictReader(associations_file)

        has_ids = "Associated Gene IDs" in reader.fieldnames
        assert(has_ids or gene_names_path != None)

        if not has_ids:
            _, name_to_protein = load_gene_names(gene_names_path)

        for row in reader:
            disease_id = row["Disease ID"]
            if(diseases_subset and disease_id not in diseases_subset):
                continue  
            disease_name = row["Disease Name"]

            if has_ids:
                disease_proteins = set([int(a.strip()) 
                                        for a in row["Associated Gene IDs"].split(",")])
            else:
                disease_proteins = set([int(name_to_protein[a.strip()]) 
                                        for a in row["Associated Genes Names"].split(",")
                                        if a.strip() in name_to_protein])

            validation_proteins = None 
            if "Validation Gene IDs" in row:
                validation_proteins = set([int(a.strip()) 
                                          for a in row["Validation Gene IDs"].split(",")])

            elif "Validation Gene Names" in row: 
                validation_proteins = set([int(name_to_protein[a.strip()]) 
                                           for a in row["Validation Gene Names"].split(",")
                                           if a.strip() in name_to_protein])
            total += len(disease_proteins)
            diseases[disease_id] = Disease(disease_id, disease_name, 
                                           disease_proteins, validation_proteins)
    print(total)
    return diseases 


def write_associations(diseases, associations_path, threshold=10): 
    """ Write a set of disease-protein associations to a csv
    Args:
        diseases
        assoications_path (string)
        diseases_subset (set) 
    """
    disease_list = [{"Disease ID": disease.id,
                     "Disease Name": disease.name,
                     "Associated Gene IDs": ",".join(map(str, disease.proteins))} 
                    for _, disease in diseases.items() if len(disease.proteins) >= threshold] 

    with open(associations_path, 'w') as associations_file:
        writer = csv.DictWriter(associations_file, fieldnames=["Disease ID", 
                                                               "Disease Name", 
                                                               "Associated Gene IDs"])
        writer.writeheader()
        for disease in disease_list:
            writer.writerow(disease)


def load_network(network_path=NETWORK_PATH):
    """ Load a network. Returns numpy adjacency matrix, networkx network and 
    dictionary mapping entrez protein_id to node index in network and adjacency
    matrix.
    Args:
        network_path (string)
    """
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
            adj[n1, n2] = 1
            adj[n2, n1] = 1
    return nx.from_numpy_matrix(adj), adj, protein_to_node


def build_random_network(model_path, name="random-network.txt"):
    """
    Generates a random network with degree sequence matching the network
    at model_path.
    Args:
        model_path (string) 
    """
    model_networkx, _, protein_to_node = load_network(model_path)
    node_to_protein = {node: protein for protein, node in protein_to_node.items()}
    deg_sequence = np.array(model_networkx.degree())[:, 1]
    random_network = nx.configuration_model(deg_sequence, create_using=nx.Graph)
    with open(os.path.join("data/networks/", name), 'w') as file:
        for edge in random_network.edges():
            node_1, node_2 = edge[0], edge[1]
            protein_1, protein_2 = node_to_protein[node_1], node_to_protein[node_2]
            file.write(str(protein_1) + " " + str(protein_2) + '\n')


def load_gene_names(file_path, a_converter=lambda x: x, b_converter=lambda x: x):
    """ Load a mapping between entrez_id and gene_names.
    Args:
        file_path (string)
    Return:
        protein_to_name (dict)
        name_to_protein (dict)
    """
    a_to_b = {}
    b_to_a = {}
    with open(file_path) as file:
        for line in file:
            if line[0] == '#': continue
            line_elems = line.split()
            if (len(line_elems) != 2): continue
            a, b = line.split()
            a = a_converter(a)
            b = b_converter(b)
            a_to_b[a] = b
            b_to_a[b] = a 
    return a_to_b, b_to_a

""" Biogrid Homo-Sapiens ID """
HOMO_SAPIENS_ID = 9606


def build_biogrid_network(biogrid_path, name='biogrid-network.txt'):
    """ Converts a biogrid PPI network into a list of entrez_ids. 
    Args:
        biogrid+path (string)
    """
    name_to_protein, _ = load_gene_names('data/protein_data/symbol_to_entrez.txt', b_converter=int)

    interactions = []

    with open(biogrid_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:

            # only include interactions between two human proteins
            if int(row['ORGANISM_A_ID']) != HOMO_SAPIENS_ID or int(row['ORGANISM_B_ID']) != HOMO_SAPIENS_ID:
                continue

            # only include interactions for which we have an entrez id
            if row['OFFICIAL_SYMBOL_A'] not in name_to_protein or row['OFFICIAL_SYMBOL_B'] not in name_to_protein:
                continue 
            
            interactions.append((str(name_to_protein[row['OFFICIAL_SYMBOL_A']]), 
                                 str(name_to_protein[row['OFFICIAL_SYMBOL_B']])))
    
    with open(os.path.join("data", "networks", name), 'w') as file:
        for interaction in interactions: 
            file.write(' '.join(interaction) + '\n')


def build_string_network(string_path, name='string-network.txt'):
    """ Converts a biogrid PPI network into a list of entrez_ids. 
    Args:
        biogrid+path (string)
    """
    _, name_to_protein = load_gene_names('data/protein_data/entrez_to_string.tsv', 
                                         a_converter=int)

    interactions = []

    with open(string_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        for row in reader:

            # only include interactions with nonzero experimental 
            if int(row['experiments']) < 1:
                continue

            # only include interactions for which we have an entrez id
            if (row['protein1'] not in name_to_protein or 
                row['protein2'] not in name_to_protein):
                continue 
            
            interactions.append((str(name_to_protein[row['protein1']]), 
                                 str(name_to_protein[row['protein2']])))
    
    with open(os.path.join("data", "networks", name), 'w') as file:
        for interaction in interactions: 
            file.write(' '.join(interaction) + '\n') 


def build_disgenet_associations(disgenet_path, name='disgenet-associations.csv'):
    """ Converts a disgenet file of associations into the accepted format for
    gene-disease associations.
    Args: 
        disgenet_path
    """
    with open(disgenet_path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')
        
        diseases = {}
        for row in reader:
            disease_id = row["diseaseId"]
            disease_name = row["diseaseName"]
            gene_id = row["geneId"]

            disease = diseases.setdefault(disease_id, Disease(disease_id, disease_name, []))
            disease.proteins.append(int(gene_id))
    
    write_associations(diseases, os.path.join("data", "associations", name))  


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    # Log Title 
    if(args.job == split_diseases):
        print_title("Disease Set Splitting")

        print("Splitting diseases...")
        split_fractions = {'val': 0.40,
                        'test': 0.60}
        split_diseases(split_fractions, 'experiments/associations/bio-pathways-associations.csv')
        print("Done.")
    
    elif(args.job == "build_biogrid"):
        print_title("Building Biogrid Network")

        build_biogrid_network('data/networks/biogrid-raw.txt')
    
    elif (args.job == "build_string"):
        print_title("Building String Network")
        build_string_network('data/raw/string-raw.txt')
    
    elif(args.job == "build_disgenet"):
        print_title("Building Disgenet Associations")

        build_disgenet_associations("data/associations/disgenet_raw.tsv")
    
    elif(args.job == "build_ppi_matrix"):
        print_title("Build PPI Matrices with PPI Network")

        print("Loading PPI Network...")
        _, ppi_network_adj, _ = load_network("data/networks/bio-pathways-network.txt")

        print("Building PPI Matrix...")
        build_ppi_comp_matrix(ppi_network_adj, deg_fns='sqrt', row_norm=False, col_norm=False, network_name="bio-pathways")
    
    elif(args.job == "build_disease_classes"):
        print_title("Build Disease Classes")

        diseases = load_diseases('data/associations/disgenet-associations.csv')
        load_disease_classes(diseases, hdo_path='data/raw/disease_ontology.obo')
        output_diseases(diseases, 'data/disease_data/disgenet-classes.csv')

    elif(args.job == "generate_random_network"):
        print_title("Generating Random Network")
        build_random_network("data/networks/bio-pathways-network.txt")
    
    elif(args.job == "print_data"):
        
        diseases = load_diseases('data/associations/disgenet-associations.csv')

        ppi_networkx, _, _ = load_network("data/networks/string-network.txt")
        print("Nodes: {}".format(len(ppi_networkx)))
        print("Edges: {}".format(ppi_networkx.number_of_edges()))



    else:
        print ("Job not recognized.")
