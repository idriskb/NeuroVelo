import torch
from torch.utils.data import Dataset
import numpy as np
from anndata import AnnData

def split_data(
    adata: AnnData,
    percent: float,
    sample_obs: float,
    layer = ['spliced','unspliced']
):
    """
    Split the dataset for training and validation

    Parameters
    ----------
    adata
        The `AnnData` object for the whole dataset
    percent
        The percentage to be used for training the model
    sample_obs
        The column of sample/treatment index
    layer
        Layer used for training
        (Default: ['spliced','unspliced'])
    Returns
    ----------
    `AnnData` object for training and validation
    """
    n_cells = adata.n_obs
    n_train = int(np.ceil(n_cells * percent))
    n_val = n_cells-n_train
    
    indices = np.random.permutation(n_cells)
    train_idx = np.random.choice(indices, n_train, replace = False)
    indices2 = np.setdiff1d(indices, train_idx)
    val_idx = np.random.choice(indices2, n_val, replace = False)


    train_data = [adata.layers[layer[0]][train_idx], adata.layers[layer[1]][train_idx], adata.obs[sample_obs][train_idx]]
    val_data = [adata.layers[layer[0]][val_idx], adata.layers[layer[1]][val_idx], adata.obs[sample_obs][val_idx]]
    return train_data, val_data


class MakeDataset(Dataset):
    """
    A class to generate Dataset

    Parameters
    ----------
    adata
        An `AnnData` object
    """

    def __init__(
        self,
        su_data,
    ):

        self.data = su_data
        self.data[0] = torch.Tensor(self.data[0])
        self.data[1] = torch.Tensor(self.data[1])
        self.data[2] = torch.Tensor(self.data[2])

    def __len__(self):
        return self.data[0].size(0)

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]
