#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:51:46 2024

This script implements a split Hamiltonian formulation to perform HMC for DeepONets.

@author: ponkrshnan
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.io
from random import sample
import util
from hamiltorch import samplers
from time import time
from datetime import datetime
import os
from model import DeepONet
import config_splitting as cfg
from my_make_func import Functional_DeepONet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_split_data():
    """
    Function to split the Data into 'n' splits. 'n' should divide the data equally between splits.
    Returns
    -------
    tuple
        Training and validation datasets.
    """
    if cfg.dataset == 'Burgers':
        data_mat = scipy.io.loadmat('../Data/DeepOnet_data.mat')
        branch_in = torch.tensor(np.expand_dims(data_mat['branch_in'][0:cfg.N_train].astype(np.float32), axis=1))
        trunk_in = torch.tensor(np.expand_dims(data_mat['trunk_in'].astype(np.float32), axis=0))
        y = torch.tensor(data_mat['solution'][0:cfg.N_train].astype(np.float32))
        if branch_in.shape[0] % cfg.num_splits != 0:
            raise ValueError('Number of splits does not split the data equally')
        n_per_batch = branch_in.shape[0] // cfg.num_splits
        train_data = []
        for i in range(cfg.num_splits):
            train_data.append((branch_in[i * n_per_batch:(i + 1) * n_per_batch:], trunk_in,
                               y[i * n_per_batch:(i + 1) * n_per_batch:]))

        branch_in = torch.tensor(
            np.expand_dims(data_mat['branch_in'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32), axis=1))
        trunk_in = torch.tensor(np.expand_dims(data_mat['trunk_in'].astype(np.float32), axis=0))
        y = torch.tensor(data_mat['solution'][cfg.N_train:cfg.N_train + cfg.N_valid].astype(np.float32))
        valid_data = (branch_in, trunk_in, y)
        return train_data, valid_data

    elif cfg.dataset == 'Cone':
        tr_data, vld_data = util.get_cone_data()
        branch_in = tr_data['Xf']
        trunk_in = tr_data['Xp']
        y = tr_data['Y']

        if branch_in.shape[0] % cfg.num_splits != 0:
            raise ValueError('Number of splits does not split the data equally')
        n_per_batch = branch_in.shape[0] // cfg.num_splits
        train_data = []
        for i in range(n_per_batch):
            train_data.append((branch_in[i * n_per_batch:(i + 1) * n_per_batch:],
                               trunk_in[i * n_per_batch:(i + 1) * n_per_batch:],
                               y[i * n_per_batch:(i + 1) * n_per_batch:]))

        branch_in = vld_data['Xf']
        trunk_in = vld_data['Xp']
        y = vld_data['Y']
        valid_data = (branch_in, trunk_in, y)

        return train_data, valid_data


def define_model_log_prob(model, model_loss, tr_data, tau_list, tau_out, predict=False, prior_scale=1.0, device='cpu'):
    """This function defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
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
    tr_data : torch.tensor
        Input training data to define the log probability. Should be a shape that can be passed into the model. First dimension is N, where N is the number of data points.
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

    # fmodel = util.make_functional(model)
    f_deeponet = Functional_DeepONet(depth_branch=cfg.branch_depth, depth_trunk=cfg.trunk_depth,
                                     activation=cfg.activation, model=model,
                                     impose_bc=False if cfg.dataset == 'Cone' else True)
    fmodel = f_deeponet.functional_model
    fsample = f_deeponet.sample_weights

    dist_list = []
    if cfg.load_prior:
        # for ind in range(tau_list[0].shape[0]):
        #     dist_list.append(torch.distributions.Normal(tau_list[0][ind], tau_list[1][ind]))
        dist_list.append(torch.distributions.Normal(tau_list[0], tau_list[1]))
    else:
        for tau in tau_list:
            dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau ** 0.5))

    if model_loss == 'NLL':
        nll_loss = torch.nn.GaussianNLLLoss(reduction='sum')

    def log_prob_func(params, *args):
        # model.zero_grad()
        # params is flat
        # Below we update the network weights to be params
        # params_unflattened = util.unflatten(model, params)
        if len(args) != 0:
            fsample()
            print('Sampled from learned parameter distributions')
            return

        l_prior = torch.zeros_like(params[0], requires_grad=True)  # Set l2_reg to be on the same device as params

        if cfg.load_prior:
            # for param, dist in zip(params,dist_list):
            l_prior = dist_list[0].log_prob(params).sum() + l_prior
        else:
            l_prior = dist_list[0].log_prob(params).sum() + l_prior

        x1, x2, y = tr_data
        if predict or not cfg.sample_data:
            # ind = list(range(x2.shape[1]))
            output = fmodel(x1.to(device), x2.to(device), parameters=params)
            y_device = y.to(device)
        else:
            ind = sample(range(x2.shape[1]), cfg.p)
            # ind = list(range(100))
            # ind = list(range(x2.shape[1]))
            output = fmodel(x1.to(device), x2[:, ind].to(device), parameters=params)
            # output.append(fmodel(tr_data['Xf'].to(device), tr_data['Xp'].to(device), params)) 
            y_device = y[:, ind].to(device)

        # output = torch.cat(output)
        output = output.reshape(y_device.shape)
        # y_device = torch.cat(y_device)
        # output = fmodel(x_device, params=params_unflattened)
        # model.load_state_dict(temp_dict)
        assert output.shape == y_device.shape
        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device))
        elif model_loss == 'multi_class_linear_output':
            #         crit = nn.MSELoss(reduction='mean')
            crit = nn.CrossEntropyLoss(reduction='sum')
            #         crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device.long().view(-1)))
            # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out * (torch.nn.functional.nll_loss(output, y_device.long().view(-1)))

        elif model_loss == 'regression':
            #crit = nn.MSELoss(reduction='mean')
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)  #sum(0)
            #print(crit(output,y_device))

        elif model_loss == 'NLL':
            ll = - nll_loss(output, y_device, tau_out * torch.ones_like(output))

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            #del x_device, y_device
            torch.cuda.empty_cache()

        if predict:
            return (ll + l_prior / prior_scale), output
        else:
            return (ll + l_prior / prior_scale)

    return log_prob_func


def define_split_model_log_prob(model, model_loss, train_loader, num_splits, tau_list, tau_out,
                                predict=False, device='cpu', verbose=True):
    """ This function is taken from the Hamiltorch library and modified for DeepONets as necessary. This function defines the list of log_prob_func's to be used for splitting. It follows the same formulation as `define_model_log_prob`, except
    it will split the log_prob_func according to data subsets.

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
    train_loader : torch.utils.data.Dataloader
        Data loader to be used for dividing data into subsets. The batch size corresponds to the subset size.
    num_splits : int
        Determines the number of splits to use. For maximum use of data set, set to the total number of batches. Be careful to ensure each batch
        is the same length.
    tau_list : torch.tensor
        A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.
    tau_out : float
        Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
    predict : bool
        Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects
        to return.
    device : name of device, or {'gpu', 'cpu'}
        The device to run on.
    verbose : bool
        If set to true then do not display loading bar.

    Returns
    -------
    list
        Returns a list of functions, where each function corresponds to the data subset split.

    """

    log_prob_list = []
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > num_splits - 1:
            break
        log_prob_func = define_model_log_prob(model, model_loss, data, tau_list, tau_out,
                                              prior_scale=num_splits, predict=predict, device=device)
        log_prob_list.append(log_prob_func)
    if verbose:
        print('Number of splits: ', len(log_prob_list), ' , each of batch size ', train_loader[0][0].shape[0], '\n')
    return log_prob_list


def predict_model(model, samples, test_loader=None, model_loss='multi_class_linear_output', tau_out=1., tau_list=None):
    """ This function is taken from the Hamiltorch library and modified for DeepONets as necessary. Function used to make predictions given model samples. Note that either a data loader can be passed in, or two tensors (x,y) but make sure
    not to pass in both.

    Parameters
    ----------
    model : torch.nn.Module
        This is the torch neural network model, which will be used when performing inference.
    samples : list of torch.tensors
        A list, where each element is a torch.tensor of shape (D,), where D is the number of parameters of the model.
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
    tau_list : torch.tensor
        A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.

    Returns
    -------
    predictions : torch.tensor
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

        log_prob_func = define_model_log_prob(model, model_loss, test_loader, tau_list, tau_out, predict=True,
                                              device=samples[0].device)

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
    Function to run Hamiltonian Monte Carlo with a split Hamiltonian formulation
    Returns
    -------
    None
    """
    dt_string = datetime.now().strftime("%d%m%y%H%M%S")
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir, exist_ok=True)
    os.system(f'cp config.py  {cfg.out_dir}/config_{dt_string}.txt');

    net = DeepONet(cfg.width_branch, cfg.width_trunk, cfg.in_branch, cfg.in_trunk,
                   cfg.branch_depth, cfg.trunk_depth, cfg.activation, cfg.output_neurons)
    params_shape_list = []
    params_flattened_list = []
    tau_list = []

    if cfg.load_prior:
        # build_tau = False
        mean_params = torch.load(f'{cfg.prior_file}/means_flattened', map_location=device)
        std_params = torch.load(f'{cfg.prior_file}/stds_flattened', map_location=device)
        tau_list = [mean_params, std_params]
    else:
        # build_tau = True
        tau_list.append(torch.tensor(cfg.prior_var))
    for weights in net.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())

    tr_data, vld_data = get_split_data()
    log_prob_func = define_split_model_log_prob(net, cfg.loss, tr_data, cfg.num_splits, tau_list, cfg.tau_out,
                                                device=device)

    params_trained = mean_params.to(device) if cfg.init_prior else util.flatten(net).to(device).clone()
    params_init = params_trained.clone()
    print("Number of parameters: ", params_init.shape[0])
    start = time()
    if cfg.is_nuts:
        params_hmc = samplers.sample(log_prob_func, params_init, num_samples=cfg.num_samples,
                                     num_steps_per_sample=cfg.L,
                                     step_size=cfg.step_size, sampler=samplers.Sampler.HMC_NUTS,
                                     integrator=samplers.Integrator.SPLITTING, burn=cfg.burn, debug=True)
    else:
        params_hmc = samplers.sample(log_prob_func, params_init, num_samples=cfg.num_samples,
                                     num_steps_per_sample=cfg.L,
                                     step_size=cfg.step_size, integrator=samplers.Integrator.SPLITTING, debug=False)

    print("Time taken: ", time() - start)

    pred_list, log_prob_list = predict_model(net, params_hmc, vld_data, model_loss=cfg.loss,
                                             tau_out=cfg.tau_out, tau_list=tau_list)

    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    *x, y_val = vld_data

    sample_mse = []
    for pred in pred_list:
        sample_mse.append(((pred.detach().cpu() - y_val) ** 2).mean())
    print('\nExpected MSE: {:.4f}'.format((np.mean(sample_mse))))
    print('\nFinal MSE: {:.4f}'.format(sample_mse[-1]))


def validate_HMC(eval_id):
    """
    Function to validate the saved samples from HMC
    Parameters
    ----------
    eval_id : str
        Unique id of the saved samples

    Returns
    -------
    None
    """
    net = DeepONet(cfg.width_branch, cfg.width_trunk, cfg.in_branch, cfg.in_trunk,
                   cfg.branch_depth, cfg.trunk_depth, cfg.activation, cfg.output_neurons)
    params_shape_list = []
    params_flattened_list = []
    tau_list = []

    if cfg.load_prior:
        mean_params = torch.load(f'{cfg.prior_file}/means_flattened', map_location=device)
        std_params = torch.load(f'{cfg.prior_file}/stds_flattened', map_location=device)
        tau_list = [mean_params, std_params]
    else:
        tau_list.append(torch.tensor(cfg.prior_var))
    for weights in net.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())

    tr_data, vld_data = get_split_data()
    params_hmc = torch.tensor(np.load(f'{cfg.out_dir}/hmc_params_{eval_id}.npy'))
    pred_list, log_prob_list = predict_model(net, params_hmc, vld_data, model_loss=cfg.loss,
                                             tau_out=cfg.tau_out, tau_list=tau_list)

    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    *x, y_val = vld_data

    sample_mse = []
    for pred in pred_list:
        sample_mse.append(((pred.detach().cpu() - y_val) ** 2).mean())
    print('\nExpected MSE: {:.6f}'.format((np.mean(sample_mse))))
    print('\nFinal MSE: {:.6f}'.format(sample_mse[-1]))


if __name__ == '__main__':
    if cfg.evaluate:
        validate_HMC(cfg.eval_uid)
    else:
        run_HMC()
