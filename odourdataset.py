from dgl.data import DGLDataset

import pandas as pd
import json
import os
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


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
    g = dgl.graph((u, v), num_nodes = nnodes)
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

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {'Br', 'C', 'Cl', 'F', 'I', 'N', 'Na', 'O', 'P', 'S', 'Zn'},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    # print(smiles)
    try:
        flag = Chem.SanitizeMol(molecule, catchErrors=True)
        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
            Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    except:
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    # Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule, global_node=False):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    max_id = 0
    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
        # print('feat: ', atom_featurizer.encode(atom).shape, type(atom_featurizer.encode(atom)))

        # Add self-loops
        max_id = max(atom.GetIdx(), max_id)
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    if global_node == True:

        # Create and add global node features and global bond feature
        global_node_feat = np.mean(np.array(atom_features), axis=0)
        atom_features.append(global_node_feat)
        global_bond_feat = np.mean(np.array(bond_features), axis=0)

        # Global node connections
        for ix in range(max_id + 1):
            pair_indices.append([max_id + 1, ix])
            bond_features.append(global_bond_feat)  # Should this be None of something else?

        # Global node self connection
        pair_indices.append([max_id + 1, max_id + 1])
        bond_features.append(bond_featurizer.encode(None))  # Should this be None of something else?

        num_nodes = max_id + 2
    else:
        num_nodes = max_id + 1

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices), num_nodes


def create_dgl_graph( edges, num_nodes):
    srcs = []
    dsts = []
    for i in edges:
        src, dst = i
        srcs.append(src)
        dsts.append(dst)
    return dgl.graph((srcs, dsts), num_nodes=num_nodes)


def dgl_graph_from_molecule( molecule, global_node=False): #need to update global node
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []
    max_id = 0
    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
        # print('feat: ', atom_featurizer.encode(atom), type(atom_featurizer.encode(atom)))

        # Add self-loops
        max_id = max(atom.GetIdx(), max_id)
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    if global_node == True:
        # Global node connections
        for ix in range(max_id + 1):
            pair_indices.append([max_id + 1, ix])
            bond_features.append(bond_featurizer.encode(None))  # Should this be None of something else?

        # Global node self connection
        pair_indices.append([max_id + 1, max_id + 1])
        bond_features.append(bond_featurizer.encode(None))  # Should this be None of something else?

        # Create and add global node features
        global_node_feat = np.mean(np.array(atom_features), axis=0)
        atom_features.append(global_node_feat)

        graph = create_dgl_graph(pair_indices, num_nodes=max_id + 2)
        graph.ndata['features'] = torch.tensor(np.array(atom_features))
        graph.edata['features'] = torch.tensor(np.array(bond_features))

    else:
        graph = create_dgl_graph(pair_indices, num_nodes=max_id + 1)
        graph.ndata['features'] = torch.from_numpy(np.array(atom_features, dtype=np.float32))
        graph.edata['features'] = torch.from_numpy(np.array(bond_features, dtype=np.float32))

    return graph


class OdourDataset_train(DGLDataset):
    def __init__(self):
        super(OdourDataset_train, self).__init__(name='odour_dataset')
        # self.all_graphs, self.all_labels = dgl.load_graphs(graphs_path)
        # self.csv_path = csv_path
        # self.mode = mode
        self.atom_featurizer = AtomFeaturizer(
            allowable_sets={
                "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
                "n_valence": {0, 1, 2, 3, 4, 5, 6},
                "n_hydrogens": {0, 1, 2, 3, 4},
                "hybridization": {"s", "sp", "sp2", "sp3"},
            }
        )

        self.bond_featurizer = BondFeaturizer(
            allowable_sets={
                "bond_type": {"single", "double", "triple", "aromatic"},
                "conjugated": {True, False},
            }
        )


    def process(self):
        savepath = '/content/drive/MyDrive/dgl_hgp_sl/dataset/'
        df = pd.read_csv(os.path.join(savepath, ('odour_graphs_leff_' + str('train') + '.csv')))
        self.graphs = []
        self.labels = []
        self.labels_set = []
        for idx, row in df.iterrows():
            if row['IsomericSMILES'] != '':

                mol = molecule_from_smiles(row['IsomericSMILES'])
                label = [int(i) for i in list(row)[2:]]
                atom_features, bond_features, pair_indices, num_nodes = graph_from_molecule(mol, global_node=True)
                g = create_dgl_graph(pair_indices, num_nodes=num_nodes)
                g.ndata['features'] = torch.from_numpy(np.array(atom_features, dtype=np.float32))
                g.edata['features'] = torch.from_numpy(np.array(bond_features, dtype=np.float32))
                # g = smiles2graph(row['SMILES'])
                # g.ndata['features'] = torch.tensor(feat_vec(row['SMILES']))
                self.num_atom_feat = atom_features.shape[1]
                self.num_bond_feat = bond_features.shape[1]
                self.graphs.append(g)
                self.labels.append(label)
                self.labels_set.append(label)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def statistics(self):
        # return 9, 0,len(self.labels_set), len(self.graphs)
        return self.num_atom_feat, self.num_bond_feat, len(self.labels_set), len(self.graphs)

class OdourDataset_val(DGLDataset):
    def __init__(self):
        super(OdourDataset_val, self).__init__(name='odour_dataset')
        # self.all_graphs, self.all_labels = dgl.load_graphs(graphs_path)
        # self.csv_path = csv_path
        # self.mode = mode
        self.atom_featurizer = AtomFeaturizer(
            allowable_sets={
                "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
                "n_valence": {0, 1, 2, 3, 4, 5, 6},
                "n_hydrogens": {0, 1, 2, 3, 4},
                "hybridization": {"s", "sp", "sp2", "sp3"},
            }
        )

        self.bond_featurizer = BondFeaturizer(
            allowable_sets={
                "bond_type": {"single", "double", "triple", "aromatic"},
                "conjugated": {True, False},
            }
        )


    def process(self):
        savepath = '/content/drive/MyDrive/dgl_hgp_sl/dataset/'
        df = pd.read_csv(os.path.join(savepath, ('odour_graphs_leff_' + str('valid') + '.csv')))
        self.graphs = []
        self.labels = []
        self.labels_set = []
        for idx, row in df.iterrows():
            if row['IsomericSMILES'] != '':

                mol = molecule_from_smiles(row['IsomericSMILES'])
                label = [int(i) for i in list(row)[2:]]
                atom_features, bond_features, pair_indices, num_nodes = graph_from_molecule(mol, global_node=True)
                g = create_dgl_graph(pair_indices, num_nodes=num_nodes)
                g.ndata['features'] = torch.from_numpy(np.array(atom_features, dtype=np.float32))
                g.edata['features'] = torch.from_numpy(np.array(bond_features, dtype=np.float32))
                # g = smiles2graph(row['SMILES'])
                # g.ndata['features'] = torch.tensor(feat_vec(row['SMILES']))
                self.num_atom_feat = atom_features.shape[1]
                self.num_bond_feat = bond_features.shape[1]
                self.graphs.append(g)
                self.labels.append(label)
                self.labels_set.append(label)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def statistics(self):
        # return 9, 0,len(self.labels_set), len(self.graphs)
        return self.num_atom_feat, self.num_bond_feat, len(self.labels_set), len(self.graphs)

class OdourDataset_test(DGLDataset):
    def __init__(self):
        super(OdourDataset_test, self).__init__(name='odour_dataset')
        # self.all_graphs, self.all_labels = dgl.load_graphs(graphs_path)
        # self.csv_path = csv_path
        # self.mode = mode
        self.atom_featurizer = AtomFeaturizer(
            allowable_sets={
                "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
                "n_valence": {0, 1, 2, 3, 4, 5, 6},
                "n_hydrogens": {0, 1, 2, 3, 4},
                "hybridization": {"s", "sp", "sp2", "sp3"},
            }
        )

        self.bond_featurizer = BondFeaturizer(
            allowable_sets={
                "bond_type": {"single", "double", "triple", "aromatic"},
                "conjugated": {True, False},
            }
        )


    def process(self):
        savepath = '/content/drive/MyDrive/dgl_hgp_sl/dataset/'
        df = pd.read_csv(os.path.join(savepath, ('odour_graphs_leff_' + str('test') + '.csv')))
        self.graphs = []
        self.labels = []
        self.labels_set = []
        for idx, row in df.iterrows():
            if row['IsomericSMILES'] != '':
                mol = molecule_from_smiles(row['IsomericSMILES'])
                label = [int(i) for i in list(row)[2:]]
                atom_features, bond_features, pair_indices, num_nodes = graph_from_molecule(mol, global_node=True)
                g = create_dgl_graph(pair_indices, num_nodes=num_nodes)
                g.ndata['features'] = torch.from_numpy(np.array(atom_features, dtype=np.float32))
                g.edata['features'] = torch.from_numpy(np.array(bond_features, dtype=np.float32))
                # g = smiles2graph(row['SMILES'])
                # g.ndata['features'] = torch.tensor(feat_vec(row['SMILES']))
                self.num_atom_feat = atom_features.shape[1]
                self.num_bond_feat = bond_features.shape[1]
                self.graphs.append(g)
                self.labels.append(label)
                self.labels_set.append(label)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def statistics(self):
        # return 9, 0,len(self.labels_set), len(self.graphs)
        return self.num_atom_feat, self.num_bond_feat, len(self.labels_set), len(self.graphs)

if __name__ == "__main__":
    pass