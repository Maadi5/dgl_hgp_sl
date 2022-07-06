import dgl
from dgl.data import DGLDataset
import torch
import torch as th
import os
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import Chem
import numpy as np
import pandas as pd

df=pd.read_csv('/content/drive/MyDrive/dgl_hgp_sl/odour2/train.csv')


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    A = Chem.GetAdjacencyMatrix(mol)
    A = np.asmatrix(A)
    nnodes=len(A)
    nz = np.nonzero(A)
    edge_list=[]
    src=[]
    dst=[]

    for i in range(nz[0].shape[0]):
      src.append(nz[0][i])
      dst.append(nz[1][i])

    u, v = src, dst
    g = dgl.graph((u, v))
    bg=dgl.to_bidirected(g)

    return bg

def feat_vec(smiles_string):
    """
    Returns atom features for a molecule given a smiles string
    """
    # atoms
    mol = Chem.MolFromSmiles(smiles_string)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    return x

lista_senten=df['SENTENCE'].to_list()
labels=[]

for olor in lista_senten:
  olor=olor.split(",")
  if 'fruity' in olor:
    labels.append(1)
  else:
    labels.append(0)

# This block makes a list of graphs
lista_mols = df['SMILES'].to_list()

j = 0
graphs = []
execptions = []
for mol in lista_mols:

    g_mol = smiles2graph(mol)

    try:
        g_mol.ndata['feat'] = torch.tensor(feat_vec(mol))
    except:
        execptions.append(j)

    graphs.append(g_mol)
    j += 1

# Some smiles are not well processed, so they are droped
ii=0
for i in execptions:
  graphs.pop(i-ii)
  labels.pop(i-ii)
  ii+=1

i=0
for grap in graphs:

  try:
    grap.ndata['feat']
  except:
    print(i)
  i+=1

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class OdourDataset2(DGLDataset):
    def __init__(self):
        super().__init__(name='smell2')

    def process(self):
        #edges = pd.read_csv('./graph_edges.csv')
        #properties = pd.read_csv('./graph_properties.csv')
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def statistics(self):
        return 9, 0, 2, len(self.graphs)

# dataset = SyntheticDataset()
#graph, label = dataset[0]
#print(graph, label)