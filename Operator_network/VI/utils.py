# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:18:35 2024

This script contains additional utility functions used by other scripts

@author: Ponkrshnan
"""
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset, DataLoader


def get_data(cfg):
    """
    Function to load data from file
    Parameters
    ----------
    cfg : Configuration file of main_bayesin_deeponet.py

    Returns
    -------

    """
    if cfg.dataset == 'Burgers':
        class BurgersDataSet(Dataset):
            """Load the Burgers dataset"""

            def __init__(self, branch_in, trunk_in, output, p):
                self.x = trunk_in.astype(np.float32)
                self.f_x = branch_in.astype(np.float32)
                self.u_x = output.astype(np.float32)
                self.p = p

            def __len__(self):
                return int(self.f_x.shape[0])

            def __getitem__(self, i):
                ind = np.random.choice(range(self.x.shape[0]), self.p, replace=False)
                return np.expand_dims(self.f_x[i, :], axis=0), self.x[ind], self.u_x[i, ind]

        data_mat = scipy.io.loadmat('../Data/DeepOnet_data.mat')
        train_dataset = BurgersDataSet(data_mat['branch_in'][0:cfg.N_train], data_mat['trunk_in'],
                                       data_mat['solution'][0:cfg.N_train], p=cfg.p)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        valid_dataset = BurgersDataSet(data_mat['branch_in'][cfg.N_train:cfg.N_train + cfg.N_valid],
                                       data_mat['trunk_in'],
                                       data_mat['solution'][cfg.N_train:cfg.N_train + cfg.N_valid], p=cfg.p)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size)
        return train_loader, valid_loader, cfg.N_train * data_mat['trunk_in'].shape[0], cfg.N_valid * \
                                           data_mat['trunk_in'].shape[0]

    elif cfg.dataset == 'Cone':
        raise NotImplementedError(f'Dataset: {cfg.dataset} is NOT available. Dataset should be Burgers')
    else:
        raise NotImplementedError(f'Dataset: {cfg.dataset} is NOT implemented. Dataset should be Burgers or Cone')


def normalize_data(feat):
    xp_min = np.array([0.241, 50.], dtype=np.float32)
    xp_max = np.array([3.16e-01, 5.00e+02], dtype=np.float32)
    xf_min = np.array([-3.38642632], dtype=np.float32)
    xf_max = np.array([3.09895004], dtype=np.float32)
    y_min = np.array([-0.66139158], dtype=np.float32)
    y_max = np.array([2.27885358], dtype=np.float32)
    feat['Xf'] = (feat['Xf'] - xf_max) / (xf_max - xf_min)
    feat['Xp'] = (feat['Xp'] - xp_max) / (xp_max - xp_min)
    feat['Y'] = (feat['Y'] - y_max) / (y_max - y_min)
    return feat


def data_normalize(Xf, Xp):
    xp_min = np.array([0.241, 50.], dtype=np.float32)
    xp_max = np.array([3.16e-01, 5.00e+02], dtype=np.float32)
    xf_min = np.array([-3.38642632], dtype=np.float32)
    xf_max = np.array([3.09895004], dtype=np.float32)
    Xf = (Xf - xf_max) / (xf_max - xf_min)
    Xp = (Xp - xp_max) / (xp_max - xp_min)
    return Xf, Xp


def get_trsize(tfrecdataset):
    for i, _ in enumerate(tfrecdataset, 1):
        pass
    return i


def get_numbatches(data_loader):
    for i, _ in enumerate(data_loader, 1):
        pass
    return i


def flatten(model):
    return torch.cat([p.flatten() for p in model.parameters()])


def unflatten(model, flattened_params):
    if flattened_params.dim() != 1:
        print(flattened_params.shape)
        raise ValueError('Expecting a 1d flattened_params')
    params_list = []
    i = 0
    for val in list(model.parameters()):
        length = val.nelement()
        param = flattened_params[i:i + length].view_as(val)
        params_list.append(param)
        i += length

    return params_list
