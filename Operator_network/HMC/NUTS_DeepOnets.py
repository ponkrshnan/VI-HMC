#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:17:46 2024

This is a script to run step size adaptation method for DeepONets. Adaptive step size is used to compute the required
step size for 80% acceptance rate.

@author: ponkrshnan
"""
import numpy as np
import torch
import torch.nn as nn
import util
from hamiltorch import samplers
from time import time
from datetime import datetime
import os
import scipy
from my_make_func import Functional_DeepONet
from model import DeepONet
import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_data(feat):
    """
    Function to normalize the data for Cone
    Parameters
    ----------
    feat : numpy.typing.NDArray
        Dataset for the Cone problem

    Returns
    -------
    numpy.typing.NDArray
        Normalized dataset for the Cone problem
    """
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


def get_data():
    """
    Function to get the Cone and Burgers dataset
    Returns
    -------
    tuple
        Training data and validation data
    """
    if cfg.dataset == 'Cone':
        raise NotImplementedError('Cone dataset is not available. Dataset should be Burgers')

    elif cfg.dataset == 'Burgers':
        data_mat = scipy.io.loadmat('../Data/DeepOnet_data.mat')
        branch_in = torch.tensor(np.expand_dims(data_mat['branch_in'][0:cfg.N_train].astype(np.float32), axis=1))
        trunk_in = torch.tensor(np.expand_dims(data_mat['trunk_in'].astype(np.float32), axis=0))
        y = torch.tensor(data_mat['solution'][0:cfg.N_train].astype(np.float32))
        train_data = (branch_in, trunk_in, y)

        branch_in = torch.tensor(
            np.expand_dims(data_mat['branch_in'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32), axis=1))
        trunk_in = torch.tensor(np.expand_dims(data_mat['trunk_in'].astype(np.float32), axis=0))
        y = torch.tensor(data_mat['solution'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32))
        valid_data = (branch_in, trunk_in, y)
        return train_data, valid_data
    else:
        raise NotImplementedError('Dataset should be Burgers')


def define_model_log_prob(model, model_loss, tr_data, params_flattened_list, params_shape_list, tau_list, tau_out,
                          predict=False, prior_scale=1.0, device='cpu'):
    """
    This function is taken from the Hamiltorch library and modified for DeepONets as necessary.
    This function defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
    function for any work with Bayesian neural networks.

    Parameters
    ----------
    model : torch.nn.Module
        This is the torch neural network model, which will be used when performing inference.
    model_loss : {'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output', 'regression'} or function
        This determines the likelihood to be used for the model. The options correspond to:
        * 'binary_class_linear_output': model has linear output and using binary cross entropy,
        * 'multi_class_linear_output': model has linear output and using cross entropy,
        * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
        * 'regression': model has linear output and using Gaussian likelihood,
        * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number of data points.
    tr_data : Training data
    params_flattened_list : list
        A list containing the total number of parameters (weights/biases) per layer in order of the model.
        E.g. `[weights.nelement() for weights in model.parameters()]`.
    params_shape_list : list
        A list describing the shape of each set of parameters in the model.
        E.g. `[weights.shape for weights in model.parameters()]`.
    tau_list : torch.tensor
        A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.
    tau_out : float
        Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
    predict : bool
        Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects
        to return.
    prior_scale : float
        Most relevant for splitting (otherwise leave as 1.0). The prior is divided by this value.
    device : name of device, or {'gpu', 'cpu'}
        The device to run on.

    Returns
    -------
    function
        Returns a `log_prob_func`, which takes a 1-D torch.tensor of a length equal to the parameter dimension and returns a single value.

    """

    f_deeponet = Functional_DeepONet(depth_branch=cfg.branch_depth, depth_trunk=cfg.trunk_depth,
                                     activation=cfg.activation, model=model)
    fmodel = f_deeponet.functional_model
    dist_list = []
    if cfg.load_prior:
        dist_list.append(torch.distributions.Normal(tau_list[0], tau_list[1]))
    else:
        for tau in tau_list:
            dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau * 0.5))

    if model_loss == 'NLL':
        nll_loss = torch.nn.GaussianNLLLoss(reduction='sum')

    def log_prob_func(params):

        l_prior = torch.zeros_like(params[0], requires_grad=True)  # Set l2_reg to be on the same device as params

        if cfg.load_prior:
            l_prior = dist_list[0].log_prob(params).sum() + l_prior
        else:
            i_prev = 0
            for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list,
                                                   dist_list):
                w = params[i_prev:index + i_prev]
                l_prior = dist.log_prob(w).sum() + l_prior
                i_prev += index

        branch_in, trunk_in, Y = tr_data
        output = fmodel(branch_in.to(device), trunk_in.to(device), parameters=params)
        y_device = Y.to(device)
        output = output.reshape(y_device.shape)

        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device))
        elif model_loss == 'multi_class_linear_output':
            crit = nn.CrossEntropyLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device.long().view(-1)))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out * (torch.nn.functional.nll_loss(output, y_device.long().view(-1)))

        elif model_loss == 'regression':
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)  #sum(0)

        elif model_loss == 'NLL':
            ll = - nll_loss(output, y_device, tau_out * torch.ones_like(output))

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if predict:
            return (ll + l_prior / prior_scale), output
        else:
            return (ll + l_prior / prior_scale)

    return log_prob_func


def predict_model(model, samples, test_loader=None, model_loss='multi_class_linear_output', tau_out=1., tau_list=None):
    """This function is taken from the Hamiltorch library and modified for DeepONets as necessary.
    Function used to make predictions given model samples.

    Parameters
    ----------
    model : torch.nn.Module
        This is the torch neural network model, which will be used when performing inference.
    samples : list of torch.Tensors
        A list, where each element is a torch.Tensor of shape (D,), where D is the number of parameters of the model.
        The length of the list is given by the number of samples, S.
    test_loader : torch.utils.data.Dataloader, optional
        Data loader to be used for evaluating the samples. This can be set to `None` if `x` and `y` are defined.
    model_loss : {'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output', 'regression'} or function
        This determines the likelihood to be used for the model. The options correspond to:
        * 'binary_class_linear_output': model has linear output and using binary cross entropy,
        * 'multi_class_linear_output': model has linear output and using cross entropy,
        * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
        * 'regression': model has linear output and using Gaussian likelihood,
        * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number of data points.
    tau_out : float
        Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
    tau_list : torch.Tensor
        A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.

    Returns
    -------
    predictions : torch.Tensor
        Output of the model of shape (S,N,O), where S is the number of samples, N is the number of data points, and O is the output shape of the model.
    pred_log_prob_list : list
        List of log probability values for each sample. The length of the list is S.

    """
    with torch.no_grad():
        params_shape_list = []
        params_flattened_list = []
        build_tau = False
        if tau_list is None:
            tau_list = []
            build_tau = True
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau:
                tau_list.append(torch.tensor(1.))

        log_prob_func = define_model_log_prob(model, model_loss, test_loader, params_flattened_list, params_shape_list,
                                              tau_list, tau_out, predict=True, device=samples[0].device)

        pred_log_prob_list = []
        pred_list = []
        for s in samples:
            lp, pred = log_prob_func(s)
            pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
            pred_list.append(pred.detach())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.stack(pred_list), pred_log_prob_list


def run_HMC():
    """
    Function to run Hamiltonian Monte Carlo with all the parameters
    Returns
    -------
    None
    """
    dt_string = datetime.now().strftime("%d%m%y%H%M%S")
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir, exist_ok=True)
    os.system(f'cp config.py  {cfg.out_dir}/config_{dt_string}.txt');

    net = DeepONet(cfg.width_branch, cfg.width_trunk, cfg.in_branch, cfg.in_trunk,
                   cfg.branch_depth, cfg.trunk_depth, cfg.activation, cfg.output_neurons,
                   impose_bc=False if cfg.dataset == 'Cone' else True)
    params_shape_list = []
    params_flattened_list = []
    tau_list = []
    if cfg.load_prior:
        build_tau = False
        mean_params = torch.load(f'{cfg.prior_file}/means_flattened', map_location=device)
        std_params = torch.load(f'{cfg.prior_file}/stds_flattened', map_location=device)
        tau_list = [mean_params, std_params]
    else:
        build_tau = True
    for weights in net.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(torch.tensor(cfg.prior_var))

    tr_data, vld_data = get_data()

    log_prob_func = define_model_log_prob(net, cfg.loss, tr_data, params_flattened_list, params_shape_list, tau_list,
                                          cfg.tau_out, device=device)

    params_init = mean_params.to(device) if cfg.init_prior else util.flatten(net).to(device).clone()
    print("Number of parameters: ", params_init.shape[0])
    start = time()
    params_hmc = samplers.sample(log_prob_func, params_init, num_samples=cfg.num_samples, num_steps_per_sample=cfg.L,
                                 step_size=cfg.step_size, sampler=samplers.Sampler.HMC_NUTS, burn=cfg.burn, debug=False)
    print("Time taken: ", time() - start)

    pred_list, log_prob_list = predict_model(net, params_hmc[:], vld_data, model_loss=cfg.loss, tau_out=cfg.tau_out,
                                             tau_list=tau_list)

    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    y_val = vld_data[2]
    sample_mse = []
    for pred in pred_list:
        sample_mse.append(((pred - y_val) ** 2).mean())
    print('\nExpected MSE: {:.2f}'.format((np.mean(sample_mse))))
    print('\nFinal MSE: {:.2f}'.format(sample_mse[-1]))

    np.save(f'{cfg.out_dir}hmc_params_{dt_string}.npy', params_hmc)


if __name__ == '__main__':
    run_HMC()
