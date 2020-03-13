from Bio.PDB import *
import Bio.PDB.Polypeptide as polypep
import pandas as pd
import numpy as np
import networkx as nx
import pygraphviz as pgv
import datetime
import itertools
import seaborn
import matplotlib.pyplot as plt

now = datetime.datetime.now()
month = str(now.strftime("%b"))
day = str(now.strftime("%d"))
year = str(now.strftime("%y"))


def get_adjacency_matrix(pdb_id, pdb_file):

    parser = PDBParser()   # initialize biopython PDB parser
    structure = parser.get_structure(pdb_id, pdb_file)  # get PDB parsed by providing id and file name

    # deriving all amino acids based on presence of beta carbon
    amino_acids = [res for res in structure[0]['A'] if 'CB' in res]

    # set up df based on num. of amino acids. All amino acid pair interaction values will be appended.
    adj_df_values = np.zeros(shape=(len(amino_acids), len(amino_acids)))

    for i, r1 in enumerate(amino_acids):
        for j, r2 in enumerate(amino_acids):
            if i != j:
                # looking through all non-self AA interactions
                distance = r1['CB'] - r2['CB']  # distance in Angstrom, 3D space between beta carbons on 2 amino acids

                # if 3D distance < 8 Angstrom, then 3D contact is assumed.
                # Adjancency matrix has a 1 for amino acids with 3D contact (8 A limit) and 0 for not.
                if distance <= 8:
                    adj_df_values[i][j] = 1.0

                else:
                    adj_df_values[i][j] = 0
            else:
                adj_df_values[i][j] = 0

    # df with rows and cols having aa name and position; values from appended df
    adjacency_df = pd.DataFrame(index=[polypep.three_to_one(r.get_resname()) for r in amino_acids],
                                columns=[polypep.three_to_one(r.get_resname()) for r in amino_acids],
                                data=adj_df_values)

    return adjacency_df


def get_distance_matrix(pdb_id, pdb_file):
    parser = PDBParser()  # initialize biopython PDB parser
    structure = parser.get_structure(pdb_id, pdb_file)  # get PDB parsed by providing id and file name

    # deriving all amino acids based on presence of beta carbon
    amino_acids = [res for res in structure[0]['A'] if 'CB' in res]

    # set up df based on num. of amino acids. All amino acid pair interaction values will be appended.
    dist_df_values = np.zeros(shape=(len(amino_acids), len(amino_acids)))

    for i, r1 in enumerate(amino_acids):
        for j, r2 in enumerate(amino_acids):
            if i != j:
                # looking through all non-self AA interactions
                dist = r1['CB'] - r2['CB']  # distance in Angstrom, 3D space between beta carbons on 2 amino acids
                dist_df_values[i][j] = dist  # distance matrix just has 3D distance values. No cutoff required.
            else:
                dist_df_values[i][j] = 0

    # df with rows and cols having aa name and position; values from appended df
    distance_df = pd.DataFrame(index=[polypep.three_to_one(r.get_resname()) for r in amino_acids],
                               columns=[polypep.three_to_one(r.get_resname()) for r in amino_acids],
                               data=dist_df_values)

    return distance_df

