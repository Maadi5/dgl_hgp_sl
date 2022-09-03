import os
import pandas as pd
from skmultilearn.model_selection import IterativeStratification

pat = '/content/drive/MyDrive/dgl_hgp_sl/dataset/'
dataset_name = 'sharma_logs_shrunk_w_odourless'



def split_by_batches(dataset_name, input_path, num_splits= 2, split = (.8,.2)):
    inputdataset = os.path.join(input_path, (dataset_name + '.csv'))

    datasetdf = pd.read_csv(inputdataset)
    output_dataset = os.path.join(pat, 'stratified')
    if not os.path.exists(output_dataset):
        os.makedirs(output_dataset)
    y = datasetdf.drop(columns=["smiles"]).to_numpy().astype(int)
    X = datasetdf.index.values
    train_fraction = split[0]
    test_fraction = split[1]
    train_all = []
    test_all = []
    skf = IterativeStratification(n_splits=num_splits, order=2, sample_distribution_per_fold=[train_fraction, test_fraction]) #StratifiedKFold
    count = 0
    for train_index, test_index in skf.split(X, y):
        trainitr = datasetdf.iloc[train_index]
        trainitr.to_csv(os.path.join(pat, (dataset_name + '_train_' + str(count) + '.csv')))
        train_all.append(os.path.join(pat, (dataset_name + '_train_' + str(count) + '.csv')))
        testitr = datasetdf.iloc[test_index]
        testitr.to_csv(os.path.join(pat, (dataset_name + '_test_' + str(count) + '.csv')))
        test_all.append(os.path.join(pat, (dataset_name + '_test_' + str(count) + '.csv')))
        count += 1
    return train_all, test_all