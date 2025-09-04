#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:40:50 2024

Script to postprocess the results of Burgers problem

@author: ponkrshnan
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy
from model import DeepONet
from my_make_func import Functional_DeepONet
import config as cfg
from tqdm import tqdm
import matplotlib

matplotlib.rcParams['text.usetex'] = True

plt.rcParams.update({
    'font.size': 22,  # Base font size
})


def get_data():
    """
    Function to get the Burgers data from file
    Returns
    -------
    tuple[torch.Tensor,torch.Tensor,torch.Tensor]
        Branch input, trunk input, output
    """
    data_mat = scipy.io.loadmat('./Data/DeepOnet_data.mat')
    branch_in = torch.tensor(
        np.expand_dims(data_mat['branch_in'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32), axis=1))
    trunk_in = torch.tensor(np.expand_dims(data_mat['trunk_in'].astype(np.float32), axis=0))
    y = torch.tensor(data_mat['solution'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32))
    return branch_in, trunk_in, y


def animate_soln(network, params):
    """
    Animate the solution of the Burgers equation over time
    Parameters
    ----------
    network : torch.nn.Module
        Bayesian DeepONet

    Returns
    -------
    None
    """
    fig = plt.figure()

    axis = plt.axes(xlim=(0, 1),
                    ylim=(-0.2, 0.2))

    # initializing a line variable 
    line, = axis.plot([], [], color='b', linewidth=2.5)
    line1, = axis.plot([], [], '--r', linewidth=2.5)
    line2, = axis.plot([], [], '--k', linewidth=1.5)
    line3, = axis.plot([], [], '--k', linewidth=1.5)

    def init():
        line.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line, line1, line2, line3

    test_ind = 971  # np.random.choice(range(cfg.N_valid))
    branch_input, trunk_input, soln = get_data()
    branch_in = branch_input[test_ind].unsqueeze(dim=0)
    pred_list = []
    for i in range(len(params)):
        prediction = fmodel(branch_in, trunk_input, params[i])
        prediction = prediction.squeeze(dim=0).squeeze(dim=0).detach().numpy()
        pred_list.append(prediction)
    truth = soln[test_ind]
    truth = np.reshape(truth, -1)
    mean_pred = np.mean(pred_list, axis=0)
    std_pred = np.std(pred_list, axis=0)
    Nx = 101

    def animate(tstep):
        line.set_data(np.linspace(0, 1, Nx), mean_pred[Nx * tstep:Nx * (tstep + 1)])
        line1.set_data(np.linspace(0, 1, Nx), truth[Nx * tstep:Nx * (tstep + 1)])
        line2.set_data(np.linspace(0, 1, Nx), mean_pred[Nx * tstep:Nx * (tstep + 1)]
                       + 3 * std_pred[Nx * tstep:Nx * (tstep + 1)])
        line3.set_data(np.linspace(0, 1, Nx), mean_pred[Nx * tstep:Nx * (tstep + 1)]
                       - 3 * std_pred[Nx * tstep:Nx * (tstep + 1)])
        axis.set_title('t= ' + str(round(1e-2 * (tstep + 1), 2)))
        return line, line1

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=100, interval=1, blit=True)
    anim.save('filename.mp4',
              writer='ffmpeg', fps=30)


def l2_relative_error(y_true, y_pred):
    """
    Function to evaluate the relative L2 error
    Parameters
    ----------
    y_true : numpy.typing.NDArray
        True labels
    y_pred : numpy.typing.NDArray
        predictions

    Returns
    -------
    Relative L2 error
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")
    return np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)


def print_error(network, params_list):
    """
    Function to print the min, max and mean relative error
    Parameters
    ----------
    network : torch.nn.Module
        Bayesian DeepONet
    params_list : list
        list of samples of parameters

    Returns
    -------
    None
    """
    branch_input, trunk_input, soln = get_data()
    err = []
    for params in tqdm(params_list):
        for param in params:
            prediction = network(branch_input, trunk_input, param)
            err.append(l2_relative_error(soln, prediction.squeeze()))
    err = np.array(err)
    print("Mean Relative L2 error: ", np.mean(err))
    print("MAP error: ", np.min(np.mean(err, axis=1)))
    print("Min error index: ", np.unravel_index(err.argmin(), err.shape))
    print("Max error index: ", np.unravel_index(err.argmax(), err.shape))


def plot_correlation(f_model, params_list):
    """
    Plot the correlation between error and uncertainties
    Parameters
    ----------
    f_model : function
        functional Bayesian DeepONet
    params_list: list
        List of samples of parameters
    Returns
    -------
    None
    """

    def eval_corr(t):
        time_ind = [t * 101, (t + 1) * 101]
        branch_input, trunk_input, soln = get_data()
        trunk_in = trunk_input[:, time_ind[0]:time_ind[1], :]
        truth = soln[:, time_ind[0]:time_ind[1]]
        pred_tensor = torch.empty(len(params_list[0]) * len(params_list), truth.shape[0], truth.shape[1])
        count = 0
        for params in params_list:
            for i in range(len(params)):
                prediction = f_model(branch_input, trunk_in, params[i])
                pred_tensor[count] = prediction.squeeze(dim=1)
                count += 1

        mean_pred = torch.mean(pred_tensor)
        std_pred = torch.std(pred_tensor, dim=0)
        err_pred = torch.abs((mean_pred - truth))
        plt.figure(figsize=(8, 5))
        plt.scatter(err_pred, std_pred)
        plt.xlabel('Absolute error')
        plt.ylabel('Standard deviation $\sigma$')
        plt.show()
        corr_coeff = np.corrcoef(err_pred.flatten(), std_pred.flatten())
        print('Correlation Coefficient: ', corr_coeff[0][1])
        return torch.mean(err_pred), torch.mean(std_pred), corr_coeff[0][1]

    corr_list = []
    err_list = []
    std_list = []
    for t in range(101):
        err, std, corr = eval_corr(t=t)
        err_list.append(err)
        std_list.append(std)
        corr_list.append(corr)

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, 101), corr_list)
    plt.xlabel('Time')
    plt.ylabel('Correlation coefficient')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, 101), err_list)
    plt.xlabel('Time')
    plt.ylabel('Mean absolute error')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, 101), std_list)
    plt.xlabel('Time')
    plt.ylabel('Mean standard deviation')
    plt.show()


def plot_predictions(f_model, params_list):
    """
    Function to plot the predictions and uncertainties
    Parameters
    ----------
    f_model : function
        Functional DeepONet model
    params_list : list
        List of samples of the parameters

    Returns
    -------
    None
    """
    test_ind = 79  # np.random.choice(range(cfg.N_valid)) 886, 79
    branch_input, trunk_input, soln = get_data()
    branch_in = branch_input[test_ind].unsqueeze(dim=0)

    pred_list = []
    for params in params_list:
        for i in range(len(params)):
            prediction = f_model(branch_in, trunk_input, params[i])
            prediction = prediction.squeeze(dim=0).squeeze(dim=0).detach().numpy()
            pred_list.append(prediction[-101:])

    truth = soln[test_ind]

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(0, 1, 101), np.array(pred_list[:]).T, 'C0', alpha=0.1)
    plt.plot(np.linspace(0, 1, 101), np.mean(pred_list[:], axis=0).T, 'k', alpha=0.9, linewidth=3,
             label='Mean prediction')
    plt.plot(np.linspace(0, 1, 101), truth[-101:], label='Ground Truth', color='red',
             linestyle='--', linewidth=3)
    plt.plot(np.linspace(0, 1, 101), branch_in.view(101, ), 'C1', alpha=0.6, linewidth=3)
    np.save(f'{cfg.out_dir}/prediction_std_{test_ind}.npy', np.std(pred_list[:], axis=0))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.out_dir}/vihmc_prediction_{test_ind}.pdf', dpi=300)
    plt.show()


def get_list_fnames():
    """
    Function to get the list of all unique ids of saved samples
    Returns
    -------
    list
        List of unique ids of saved samples
    """
    # Open the file in read mode
    with open(f'{cfg.out_dir}/fnames.txt', 'r') as f:
        # Read the lines of the file into a list
        lines = f.readlines()

    # Create an empty list to store the imported list
    my_list = []

    # Iterate over each line in the file
    for line in lines:
        # Strip the newline character and convert the line to the appropriate data type (e.g., int, float, or str)
        item = line.strip()
        my_list.append(item)
    return my_list


unique_id = get_list_fnames()
sens_uid = cfg.prior_uid
samples = []
for uid in unique_id:
    samples.append(torch.from_numpy(np.load(f'{cfg.out_dir}/hmc_params_{uid}.npy', allow_pickle=True)[cfg.burn:]))

net = DeepONet(cfg.width_branch, cfg.width_trunk, cfg.in_branch, cfg.in_trunk,
               cfg.branch_depth, cfg.trunk_depth, cfg.activation, cfg.output_neurons)
mean_params = torch.load(f'{cfg.prior_file}/means_flattened_{sens_uid}')
std_params = torch.load(f'{cfg.prior_file}/stds_flattened_{sens_uid}')
grad_ind = np.load(f'{cfg.prior_file}/gradient_indices_{sens_uid}.npy', allow_pickle=True)
f_deeponet = Functional_DeepONet(depth_branch=cfg.branch_depth, depth_trunk=cfg.trunk_depth,
                                 mus=mean_params.clone().detach(),
                                 sigmas=std_params.clone().detach(),
                                 activation=cfg.activation, sensitive_ind=grad_ind, model=net)
fmodel = f_deeponet.functional_model
# print_error(fmodel, samples)
# plot_correlation(fmodel, samples)
plot_predictions(fmodel, samples)
# animate_soln(fmodel,params_hmc[cfg.burn:])
