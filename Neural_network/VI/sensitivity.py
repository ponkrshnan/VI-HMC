#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:16:19 2024

This script performs the sensitivity analysis

@author: ponkrshnan
"""
import torch
import torch.nn as nn
import util
import numpy as np
import os
from matplotlib import pyplot as plt
import config_sens as cfg
from torch.func import jacrev, functional_call
import matplotlib

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
matplotlib.rcParams['text.usetex'] = True


def get_data(N_tr, N_val, noise):
    """
    Function to generate training and validation data
    Parameters
    ----------
    N_tr : int
        Number of points in training
    N_val : int
        Number of points in validation
    noise : float
        Standard deviation of the noise parameter

    Returns
    -------
    tuple
        Training and validation data.
    """
    if cfg.load_data:
        print('Loading data from file')
        x_train = torch.load('Data/x_train')
        y_train = torch.load('Data/y_train')
        x_val = torch.load('Data/x_val')
        y_val = torch.load('Data/y_val')
    else:

        x_val = torch.linspace(-1.2, 1.2, N_val).view(-1, 1)
        # y_val = (0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(3 * x_val)).view(-1, 1)
        y_val = 4 * torch.sin(4 * x_val) + 5 * torch.cos(12 * x_val)

        x_train = torch.cat((torch.linspace(-1, -0.2, N_tr // 2), torch.linspace(0.2, 1, N_tr // 2))).view(-1, 1)
        # y_train = (0.4 * torch.sin(4 * x_train) + 0.5 * torch.cos(3 * x_train)) + torch.randn_like(x_train) * noise
        y_train = 4 * torch.sin(4 * x_train) + 5 * torch.cos(12 * x_train) + torch.randn_like(x_train) * noise

        # torch.save(x_val, 'Data/x_val')
        # torch.save(x_train,'Data/x_train')
        # torch.save(y_val,'Data/y_val')
        # torch.save(y_train,'Data/y_train')

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)
    return (x_train, y_train), (x_val, y_val)


def eval_std_dydw(valid_data, model, mean_params, std_params):
    """
    Function to evaluate sensitivity scores
    Parameters
    ----------
    valid_data : tuple
        Data to compute sensitivity scores
    model : torch.nn.module
        A deterministic model to compute sensitivities
    mean_params : list
        List containing the mean parameters of the model
    std_params : list
        List containing the standard deviation parameters of the model

    Returns
    -------
    numpy.typing.NDArray
        Sensitivity scores of the parameters
    """
    params_unflattened = util.unflatten(model, mean_params)
    cnt = 0
    for param in model.parameters():
        param.data = params_unflattened[cnt]
        cnt = cnt + 1
    x, _ = valid_data
    grads = eval_jac(model, x)
    sensitivity = grads * std_params ** 2
    return sensitivity.detach().numpy()


def eval_jac(model, x):
    """
    Function to evaluate the gradients to compute sensitivity scores
    Parameters
    ----------
    model : Model to evaluate gradients
    x : Inputs to the model

    Returns
    -------
    torch.Tensor
        Mean of the square of gradients for various inputs
    """
    params = dict(model.named_parameters())

    def fmodel(w, inputs):
        return functional_call(model, w, inputs)

    jacobian_output_to_params = jacrev(fmodel, argnums=0)(params, x)
    grads = []
    for key in jacobian_output_to_params.keys():
        temp = jacobian_output_to_params[key]
        temp = torch.mean(temp ** 2, (0, 1))
        grads.append(temp.view(-1))
    grads = torch.cat(grads)
    return grads


def captured_var(imp, var_threshold):
    """
    Function to evaluate the ratio of variance captured
    Parameters
    ----------
    imp : Sensitivity scores
    var_threshold : Sensitivity threshold

    Returns
    -------
    int
        Number of sensitive parameters for the given sensitivity threshold
    """
    tot_var = sum(imp)
    cumilative_sum = np.cumsum(np.sort(imp)[::-1])
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })
    per = cumilative_sum / tot_var
    plt.plot(np.arange(len(per)) + 1, per, color='blue', alpha=0.7, linewidth=2)
    plt.xlabel('No of parameters')
    plt.ylabel('Ratio of variance captured')
    plt.grid(True)

    # Increase the tick parameters
    locs, labels = plt.xticks()
    locs[0] = 1
    labels[0] = '$\\mathdefault{1}$'
    plt.xticks(locs, labels)
    plt.yticks()
    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/captured_variance.pdf', dpi=600)
    plt.show()
    return sum(per <= var_threshold)


def get_model(width, act, bias_on=True):
    """
    Function to construct the neural netwowk
    Parameters
    ----------
    width : List of integers containing width
    act : Activation function
    bias_on : Flag to denote if the bias is on

    Returns
    -------
    torch.nn.Module
        A deterministic neural network model
    """
    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    if act == 'relu':
        act_fn = nn.ReLU()
    elif act == 'tanh':
        act_fn = nn.Tanh()
    elif act == 'sine':
        act_fn = Sin()

    mod_list = []
    mod_list.append(nn.Linear(1, width[0]))
    mod_list.append(act_fn)
    i = -1  # For single layer
    for i in range(len(width) - 1):
        mod_list.append(nn.Linear(width[i], width[i + 1]))
        mod_list.append(act_fn)
    mod_list.append(nn.Linear(width[i + 1], 1, bias=bias_on))
    model = nn.Sequential(
        *mod_list
    )
    return model


def run():
    """
    Function to perform preprocessing, compute sensitivities and postprocessing
    Returns
    -------
    None
    """
    ckpt_dir = cfg.ckpt_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    unique_id = cfg.unique_id
    _, val_data = get_data(cfg.N_tr, cfg.N_val, cfg.noise)
    width = cfg.width
    model = get_model(width, cfg.act, cfg.bias)
    if not os.path.exists(f'{ckpt_dir}/stds_flattened_{unique_id}'):
        mean_params, std_params = util.flatten_mean_std(unique_id)
        torch.save(mean_params, f'{ckpt_dir}/means_flattened_{unique_id}')
        torch.save(std_params, f'{ckpt_dir}/stds_flattened_{unique_id}')
    else:
        mean_params = torch.load(f'{ckpt_dir}/means_flattened_{unique_id}', map_location=device)
        std_params = torch.load(f'{ckpt_dir}/stds_flattened_{unique_id}', map_location=device)

    mean_importance = eval_std_dydw(val_data, model, mean_params, std_params)
    num_params = captured_var(mean_importance, cfg.importance_threshold)
    ind = np.argsort(-mean_importance)[:num_params]
    print('Sensitivity threshold: ', -np.sort(-mean_importance)[num_params])
    ind = np.sort(ind)
    util.plot_hists(mean_importance, cfg)
    np.save(f'{ckpt_dir}/gradient_indices_{unique_id}.npy', ind)
    np.save(f'{ckpt_dir}/mean_importance_{unique_id}.npy', mean_importance)
    print('Number of parameters: ', ind.shape[0])


if __name__ == '__main__':
    run()
