#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:19:05 2024

This is a script to perform the sensitivity analysis on VI parameters

@author: ponkrshnan
"""
import torch
import numpy as np
import os

import config_sens as cfg
import utils as util
from model import DeepONet
from matplotlib import pyplot as plt
from torch.func import jacrev
import matplotlib
from my_make_func import Functional_DeepONet

matplotlib.rcParams['text.usetex'] = True
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model(valid_data, model, mean_params):
    """
    This function evaluates a deterministic model with a given set of weights and biases
    Parameters
    ----------
    valid_data : torch.DataLoader
        Data to evaluate
    model : torch.nn.Module
        A deterministic model
    mean_params : list
        A list of weights and biases of the model

    Returns
    -------
    list
        A list of predictions evaluated at the inputs for the given weights and biases
    """
    params_unflattened = util.unflatten(model, mean_params)
    cnt = 0
    for param in model.parameters():
        param.data = params_unflattened[cnt]
        cnt = cnt + 1

    pred_list = []
    for i, batch_data in enumerate(valid_data):
        if cfg.dataset == 'Cone':
            x = [batch_data['Xf'], batch_data['Xp']]
        else:
            x = [batch_data[0], batch_data[1]]
        y_pred = model(*x)
        pred_list.append(y_pred)
    return pred_list


def eval_std_dydw(valid_data, model, mean_params, std_params):
    """
    Function to evaluate sensitivity scores
    Parameters
    ----------
    valid_data : torch.DataLoader
        Data to compute sensitivity scores
    model : torch.nn.Module
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
    grads_list = 0
    for i, batch_data in enumerate(valid_data):
        if cfg.dataset == 'Cone':
            x = [batch_data['Xf'], batch_data['Xp']]
            y = batch_data['Y']
            num_batches = np.ceil(988 / cfg.batch_size)
        else:
            x = [batch_data[0], batch_data[1]]
            num_batches = len(valid_data)

        grads_list += eval_jac(model, *x) / num_batches

    assert (i + 1 == num_batches)
    sensitivities = grads_list * (std_params ** 2)
    return sensitivities.detach().numpy()


def eval_jac(model, x_branch, x_trunk):
    """
    Function to evaluate the gradients to compute sensitivity scores
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate gradients
    x_branch : torch.Tensor
        input to the branch network
    x_trunk : torch.Tensor
        input to the trunk network

    Returns
    -------
    torch.Tensor
        Mean of the square of gradients for various inputs
    """
    params = util.flatten(model)
    f_deeponet = Functional_DeepONet(depth_branch=cfg.branch_depth, depth_trunk=cfg.trunk_depth, model=model,
                                     activation=cfg.activation, impose_bc=True if cfg.dataset == 'Burgers' else False)
    fmodel = f_deeponet.functional_model
    with torch.no_grad():  # prevents memory leak
        jacobian_output_to_params = jacrev(fmodel, argnums=2)(x_branch, x_trunk, params).detach()
    grads = torch.mean(jacobian_output_to_params ** 2, dim=tuple(range(jacobian_output_to_params.ndim - 1)))
    return grads


def plot_hists(mean_grads):
    """
    Function to plot and save the histogram of sensitivity scores
    Parameters
    ----------
    mean_grads : numpy.typing.NDArray
        sensitivity scores

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })
    plt.hist(mean_grads, bins=np.linspace(0, np.percentile(mean_grads, 99), 100), edgecolor='black', color='blue',
             alpha=0.7)

    plt.yscale('log')
    plt.xlabel('Sensitivity ($S_i^2$)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.xticks()
    plt.yticks()

    plt.tight_layout()
    plt.savefig(f'{cfg.save_loc}/hist1.pdf', dpi=600)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(mean_grads, edgecolor='black', color='blue', alpha=0.7)

    plt.yscale('log')
    plt.xlabel('Sensitivity ($S_i^2$)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.xticks()
    plt.yticks()

    plt.tight_layout()
    plt.savefig(f'{cfg.save_loc}/hist2.pdf', dpi=600)
    plt.show()


def plot_grads(grads_unflattened, vmax):
    """
    Function to visualize the sensitivity scores as matrices of parameter shapes
    Parameters
    ----------
    grads_unflattened : list
        sensitivity scores
    vmax : float
        Maximum value in the colorbar

    Returns
    -------
    None
    """
    for i, mat in enumerate(grads_unflattened):
        if mat.dim() == 1:
            mat = mat.unsqueeze(1)
        elif mat.dim() == 0:
            mat = mat.unsqueeze(0)
            mat = mat.unsqueeze(1)

        plt.imshow(mat, cmap='coolwarm', vmin=0, vmax=vmax)
        plt.axis('off')

        # plt.colorbar(pad=0.1)
        plt.savefig(f'{cfg.save_loc}/sensitivity_layer_{i}.pdf', dpi=600, bbox_inches='tight')
        plt.show()


def captured_var(imp, var_threshold):
    """
    Function to evaluate the ratio of variance captured vs number of parameters
    Parameters
    ----------
    imp : numpy.typing.NDArray
        Sensitivity scores
    var_threshold : float
        Sensitivity threshold

    Returns
    -------
    Number of sensitive parameters for the given threshold
    """
    tot_var = sum(imp)
    cumilative_sum = np.cumsum(np.sort(imp)[::-1])
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })
    plt.plot(np.arange(len(cumilative_sum)) + 1, cumilative_sum / tot_var, color='blue', alpha=0.7, linewidth=2)

    plt.xlabel('No of parameters')
    plt.ylabel('Ratio of variance captured')
    plt.grid(True)

    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig(f'{cfg.save_loc}/captured_variance.pdf', dpi=600)
    plt.show()
    return sum(cumilative_sum / tot_var <= var_threshold)


def run(load_saved_sens=False):
    """
    Function to perform preprocessing, compute sensitivities and postprocessing
    Parameters
    ----------
    load_saved_sens : Boolean
        Flag to indicate if the sensitivity scores are loaded from a saved file.

    Returns
    -------
    None
    """
    train_loader, valid_loader, tr_size, vld_size = util.get_data(cfg)
    model = DeepONet(cfg.layer_width, cfg.in_branch, cfg.in_trunk, cfg.branch_depth, cfg.trunk_depth,
                     cfg.output_neurons, cfg.activation, impose_bc=True if cfg.dataset == 'Burgers' else False)
    try:
        mean_params = torch.load(f'{cfg.save_loc}/means_flattened_{cfg.uid}', map_location=device)
    except:
        saved_mod = torch.load(f'{cfg.model_loc}/max_model_{cfg.uid}.pt', map_location=device)
        mu_list = []
        std_list = []
        for name, param in saved_mod['model'].items():
            if 'mu' in name:
                mu_list.append(param.flatten())
            elif 'rho' in name:
                std_list.append(torch.log1p(torch.exp(param)).flatten())
        mean_params = torch.cat(mu_list)
        if not os.path.exists(cfg.save_loc):
            os.makedirs(cfg.save_loc, exist_ok=True)
        torch.save(mean_params, f'{cfg.save_loc}/means_flattened_{cfg.uid}')
        torch.save(torch.cat(std_list), f'{cfg.save_loc}/stds_flattened_{cfg.uid}')

    std_params = torch.load(f'{cfg.save_loc}/stds_flattened_{cfg.uid}', map_location=device)
    mean_importance = np.load(
        f'{cfg.save_loc}/sensitivity_scores_{cfg.uid}.npy') if load_saved_sens else eval_std_dydw(valid_loader,
                                                                                                  model,
                                                                                                  mean_params,
                                                                                                  std_params)
    num_params = captured_var(mean_importance, cfg.importance_threshold)
    ind = np.argsort(-mean_importance)[:num_params]
    print('Sensitivity threshold: ', -np.sort(-mean_importance)[num_params])
    ind = np.sort(ind)

    np.save(f'{cfg.save_loc}/sensitivity_scores_{cfg.uid}.npy', mean_importance)
    plot_hists(mean_importance)
    print('Min: ', min(mean_importance))
    print('Max: ', max(mean_importance))
    print('95 percentile: ', np.percentile(mean_importance, 95))
    print('No of sensitive parameters: ', len(ind))
    np.save(f'{cfg.save_loc}/gradient_indices_{cfg.uid}.npy', ind)
    importance_unflattened = util.unflatten(model, torch.tensor(mean_importance))
    plot_grads(importance_unflattened, np.percentile(mean_importance, 99))


def print_and_plot_res():
    """
    Function to print and plot saved results
    Returns
    -------
    None
    """
    mean_importance = np.load(f'{cfg.save_loc}/sensitivity_scores_{cfg.uid}.npy')
    captured_var(mean_importance, cfg.importance_threshold)
    ind = np.load(f'{cfg.save_loc}/gradient_indices_{cfg.uid}.npy')
    print('Min: ', min(mean_importance))
    print('Max: ', max(mean_importance))
    print('95 percentile: ', np.percentile(mean_importance, 95))
    print('No of sensitive parameters: ', len(ind))
    plot_hists(mean_importance)


if __name__ == '__main__':
    run(load_saved_sens=cfg.load_saved_sens)
    # print_and_plot_res()
