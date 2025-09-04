#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:35:45 2024

Script to implement HMC for the regression examples

@author: ponkrshnan
"""

import torch
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from datetime import datetime
import config as cfg
import os
import matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.rcParams['text.usetex'] = True


def get_data():
    """
    Function to generate training and validation data
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Training and validation data.
    """
    if os.path.exists('../Data/x_train'):
        x_train = torch.load('../Data/x_train')
        y_train = torch.load('../Data/y_train')
        x_val = torch.load('../Data/x_val')
        y_val = torch.load('../Data/y_val')
    else:
        print('Generating data ...')
        x_val = torch.linspace(-1.2, 1.2, cfg.N_val).view(-1, 1)
        # y_val = (0.4*torch.sin(4*x_val) + 0.5*torch.cos(3*x_val)).view(-1,1)
        y_val = (4 * torch.sin(4 * x_val) + 5 * torch.cos(12 * x_val)).view(-1, 1)

        x_train = torch.cat((torch.linspace(-1, -0.2, cfg.N_tr // 2), torch.linspace(0.2, 1, cfg.N_tr // 2))).view(-1,
                                                                                                                   1)
        # y_train = (0.4*torch.sin(4*x_train) + 0.5*torch.cos(3*x_train)) + torch.randn_like(x_train)*noise
        y_train = (4 * torch.sin(4 * x_train) + 5 * torch.cos(12 * x_train)) + torch.randn_like(x_train) * (
                1 / cfg.tau_out ** 0.5)

        # torch.save(x_val, 'Data/x_val')
        # torch.save(x_train,'Data/x_train')
        # torch.save(y_val,'Data/y_val')
        # torch.save(y_train,'Data/y_train')

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)
    return x_train, y_train, x_val, y_val


def get_model(bias_on=True):
    """
    Function to build a neural network model
    Parameters
    ----------
    bias_on : bool
        Flag indicating if the bias is on

    Returns
    -------
    torch.nn.Module
        A deterministic neural network
    """
    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    if cfg.act == 'relu':
        act_fn = nn.ReLU()
    elif cfg.act == 'tanh':
        act_fn = nn.Tanh()
    elif cfg.act == 'sine':
        act_fn = Sin()
    else:
        raise ValueError('Activation should be relu, sine or tanh')
    mod_list = []
    mod_list.append(nn.Linear(1, cfg.width[0]))
    mod_list.append(act_fn)
    i = -1  # For single layer
    for i in range(len(cfg.width) - 1):
        mod_list.append(nn.Linear(cfg.width[i], cfg.width[i + 1]))
        mod_list.append(act_fn)
    mod_list.append(nn.Linear(cfg.width[i + 1], 1, bias=bias_on))
    net = nn.Sequential(
        *mod_list
    )
    return net


def draw_hmc_samples(dtstring):
    """
    Function to draw HMC samples
    Parameters
    ----------
    dtstring : str
        Unique id to save the parameters

    Returns
    -------
    None
    """
    net = get_model(cfg.bias)
    params_init = hamiltorch.util.flatten(net).to(device).clone()
    print('Parameter size: ', params_init.shape[0])

    tau_list = []
    for w in net.parameters():
        tau_list.append(cfg.tau)
    tau_list = torch.tensor(tau_list).to(device)

    x_train, y_train, x_val, y_val = get_data()
    params_hmc = hamiltorch.sample_model(net, x_train, y_train, model_loss='regression', params_init=params_init,
                                         num_samples=cfg.num_samples, debug=0,
                                         step_size=cfg.step_size, num_steps_per_sample=cfg.L, tau_out=cfg.tau_out,
                                         normalizing_const=cfg.N_tr, tau_list=tau_list)

    np.save(f'{cfg.out_dir}/hmc_params_{dtstring}.npy', params_hmc)


def validate(dtstring):
    """
    Function to evaluate the sampled network parameters
    Parameters
    ----------
    dtstring : str
        Unique id of the saved parameters

    Returns
    -------
    None
    """
    net = get_model(cfg.bias)
    x_train, y_train, x_val, y_val = get_data()

    tau_list = []
    for w in net.parameters():
        tau_list.append(cfg.tau)
    tau_list = torch.tensor(tau_list).to(device)

    params_hmc = torch.tensor(np.load(f'{cfg.out_dir}/hmc_params_{dtstring}.npy'))
    pred_list, log_prob_list = hamiltorch.predict_model(net, x=x_val, y=y_val, model_loss='regression',
                                                        samples=params_hmc[cfg.burn:], tau_out=cfg.tau_out,
                                                        tau_list=tau_list)

    print(tau_list[0])
    print(cfg.tau_out)
    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    print('\nExpected MSE: {:.2f}'.format(((pred_list.mean(0) - y_val) ** 2).mean()))

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(8, 5))
    plt.plot(x_val.cpu().numpy(), pred_list[:].cpu().numpy().squeeze().T, 'C0', alpha=0.051)
    plt.plot(x_val.cpu().numpy(), y_val.cpu().numpy(), 'r', linewidth=3, label='True function')
    plt.plot(x_val.cpu().numpy(), pred_list.mean(0).cpu().numpy().squeeze().T, 'k', alpha=0.9, linewidth=3,
             label='Mean prediction')

    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), '.C3', markersize=30, label='x train', alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'{cfg.out_dir}/hmc_predictions.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    dt_string = datetime.now().strftime("%d%m%y%H%M%S")
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir, exist_ok=True)

    if cfg.test:
        validate(cfg.test_dtstring)
    else:
        os.system(f'cp config.py  {cfg.out_dir}/config_{dt_string}.txt');
        for run_num in range(cfg.num_chains):
            draw_hmc_samples(f'{dt_string}_{run_num}')
            validate(f'{dt_string}_{run_num}')
