"""
"""
import os 

import goatools
from goatools.associations import read_ncbi_gene2go
from goatools.base import download_ncbi_associations
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag


class GOParser():

    def __init__(self, go_path="data/go/go-basic.obo", gene2go_path="data/go/gene2go"):
        self.obo = GODag(go_path)
        self.gene_to_go = read_ncbi_gene2go(gene2go_path, go2geneids=False)
 
    def get_annotations(self, genes, level=2):
        """
        """
        genes_to_annotations = {gene: set() for gene in genes}
        for gene in genes:
            doids = self.gene_to_go[gene]
            for doid in doids:
                for parent in self.obo[doid].get_all_parents():
                    if self.obo[parent].level == level:
                        genes_to_annotations[gene].add(self.obo[parent].name)
                break
        return genes_to_annotations
