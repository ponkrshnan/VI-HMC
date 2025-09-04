#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:33:47 2024

Script to implement the hybrid VI-HMC framework for neural networks

@author: ponkrshnan
"""

import torch
import util
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from datetime import datetime
from my_make_func import Functional_Net
import config as cfg
import os
import matplotlib
import hamiltorch.samplers as samplers

matplotlib.rcParams['text.usetex'] = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, prior_list, tau_out,
                          predict=False, prior_scale=1.0, device='cpu', dt_string=None, grad_ind=None):
    """ This function is taken from the Hamiltorch library and modified for VI-HMC as necessary. This function defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
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
    x : torch.tensor
        Input training data to define the log probability. Should be a shape that can be passed into the model. First dimension is N, where N is the number of data points.
    y : torch.tensor
        Output training data to define the log probability. Should be a shape that suits the likelihood (or - loss) of the model.
        First dimension is N, where N is the number of data points.
    params_flattened_list : list
        A list containing the total number of parameters (weights/biases) per layer in order of the model.
        E.g. `[weights.nelement() for weights in model.parameters()]`.
    params_shape_list : list
        A list describing the shape of each set of parameters in the model.
        E.g. `[weights.shape for weights in model.parameters()]`.
    prior_list : list
        A list containing the  prior parameters for each set of per layer parameters. This is assuming a Gaussian prior.
    tau_out : float
        Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
    predict : bool
        Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects
        to return.
    prior_scale : float
        Most relevant for splitting (otherwise leave as 1.0). The prior is divided by this value.
    device : name of device, or {'gpu', 'cpu'}
        The device to run on.
    dt_string : str
        Unique id of the samples
    grad_ind : list
        Indices of the sensitive parameters
    Returns
    -------
    function
        Returns a `log_prob_func`, which takes a 1-D torch.tensor of a length equal to the parameter dimension and returns a single value.

    """
    params_mu = torch.load(f'{cfg.prior_file}/means_flattened_{cfg.prior_uid}')
    params_std = torch.load(f'{cfg.prior_file}/stds_flattened_{cfg.prior_uid}')
    if grad_ind is None:
        grad_ind = np.load(f'{cfg.prior_file}/gradient_indices_{cfg.prior_uid}.npy', allow_pickle=True)
    f_net = Functional_Net(depth=cfg.depth, act=cfg.act, bias=cfg.bias, mus=params_mu.clone().detach().to(device),
                           sigmas=params_std.clone().detach().to(device), sensitive_ind=grad_ind,
                           model=model)
    fmodel = f_net.functional_model
    fsample = f_net.sample_weights

    dist_list = []
    if cfg.load_prior:
        dist_list.append(torch.distributions.Normal(prior_list[0], prior_list[1]))
    else:
        for tau in prior_list:
            dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau ** 0.5))

    if model_loss == 'NLL':
        nll_loss = torch.nn.GaussianNLLLoss(reduction='sum')

    def log_prob_func(params, *args):
        if len(args) != 0:
            fsample(dt_string)
            return

        i_prev = 0
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

        # Sample prior if no data
        if x is None:
            return l_prior / prior_scale

        x_device = x.to(device)
        y_device = y.to(device)

        output = fmodel(x_device, params)

        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device))
        elif model_loss == 'multi_class_linear_output':
            crit = nn.CrossEntropyLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device.long().view(-1)))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out * (torch.nn.functional.nll_loss(output, y_device.long().view(-1)))

        elif model_loss == 'regression':
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)

        elif model_loss == 'NLL':
            ll = - nll_loss(output, y_device, tau_out * torch.ones_like(output))

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            del x_device, y_device
            torch.cuda.empty_cache()

        if predict:
            return (ll + l_prior / prior_scale), output
        else:
            return (ll + l_prior / prior_scale)

    return log_prob_func


def predict_model(model, samples, x=None, y=None, test_loader=None, model_loss='multi_class_linear_output', tau_out=1.,
                  prior_list=None, verbose=False, dt_string=None, grad_ind=None):
    """ This function is taken from the Hamiltorch library and modified for VI-HMC as necessary. Function used to make predictions given model samples. Note that either a data loader can be passed in, or two tensors (x,y) but make sure
    not to pass in both.

    Parameters
    ----------
    model : torch.nn.Module
        This is the torch neural network model, which will be used when performing inference.
    samples : list of torch.tensors
        A list, where each element is a torch.tensor of shape (D,), where D is the number of parameters of the model.
        The length of the list is given by the number of samples, S.
    x : torch.tensor, optional
        Input data to be evaluated over. Set this to `None` if using `test_loader`.
    y : torch.tensor, optional
        Output labels to be evaluated with. Set this to `None` if using `test_loader`.
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
    prior_list : list
        A list containing the prior parameters for each set of per layer parameters. This is assuming a Gaussian prior.
    verbose : bool
        If set to true then do not display loading bar.
    dt_string : str
        Unique id of the samples
    grad_ind : list
        Indices of the sensitive parameters

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
        if prior_list is None:
            prior_list = []
            build_tau = True
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau:
                prior_list.append(torch.tensor(1.))

        if test_loader.__class__ is torch.utils.data.dataloader.DataLoader:
            # Calc number of batches
            if len(test_loader.dataset) % test_loader.batch_size == 0.0:
                num_batches = len(test_loader.dataset) / test_loader.batch_size
            else:
                num_batches = int(round(len(test_loader.dataset) / test_loader.batch_size) + 1)

            log_prob_list = samplers.define_split_model_log_prob(model, model_loss, test_loader, num_batches,
                                                                 params_flattened_list, params_shape_list, prior_list,
                                                                 tau_out, normalizing_const=1., predict=True,
                                                                 device=samples[0].device, verbose=verbose)

            pred_log_prob_list = []
            pred_list = []
            for s in samples:
                lp_l = 0.
                pred_l = []
                for log_prob_func in log_prob_list:
                    lp, pred = log_prob_func(s)
                    lp_l += lp.cpu()
                    pred_l.append(pred)
                lp = lp_l
                pred = torch.cat(pred_l)
                pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
                pred_list.append(pred.detach())
        elif x is not None and y is not None:

            if x.device != samples[0].device:
                raise RuntimeError('x on device: {} and samples on device: {}'.format(x.device, samples[0].device))

            log_prob_func = define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list,
                                                  prior_list, tau_out, predict=True, device=samples[0].device,
                                                  dt_string=dt_string, grad_ind=grad_ind)

            pred_log_prob_list = []
            pred_list = []
            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
                pred_list.append(pred.detach())
        else:
            raise RuntimeError('Val data not defined (i.e. arguments x, y, val_loader are all not defined)')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.stack(pred_list), pred_log_prob_list


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
        # torch.manual_seed(0)
        x_val = torch.linspace(-1.2, 1.2, cfg.N_val).view(-1, 1)
        # y_val = (0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(3 * x_val)).view(-1, 1)
        y_val = (4 * torch.sin(4 * x_val) + 5 * torch.cos(12 * x_val)).view(-1, 1)

        x_train = torch.cat((torch.linspace(-1, -0.2, cfg.N_tr // 2), torch.linspace(0.2, 1, cfg.N_tr // 2))).view(-1,
                                                                                                                   1)
        # y_train = (0.4 * torch.sin(4 * x_train) + 0.5 * torch.cos(3 * x_train)) + torch.randn_like(x_train) * (
        #         1 / cfg.tau_out ** 0.5)
        y_train = (4 * torch.sin(4 * x_train) + 5 * torch.cos(12 * x_train)) + torch.randn_like(x_train) * (
                    cfg.tau_out)

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


def draw_hmc_samples(unique_id):
    """
    Function to draw HMC samples in the reduced space
    Parameters
    ----------
    unique_id : str
        Unique id to save the parameters

    Returns
    -------
    None
    """
    x_train, y_train, x_val, y_val = get_data()
    net = get_model(cfg.bias)

    params_shape_list = []
    params_flattened_list = []
    prior_list = []
    grad_ind = np.load(f'{cfg.prior_file}/gradient_indices_{cfg.prior_uid}.npy', allow_pickle=True)

    if cfg.load_prior:
        print("Loading priors from file")
        build_tau = False
        mean_params = torch.load(f'{cfg.prior_file}/means_flattened_{cfg.prior_uid}', map_location=device)
        std_params = torch.load(f'{cfg.prior_file}/stds_flattened_{cfg.prior_uid}', map_location=device
                                ) if cfg.load_std else cfg.prior_var * torch.ones_like(mean_params)
        prior_list = [mean_params[grad_ind], std_params[grad_ind]]
    else:
        build_tau = True

    for weights in net.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            prior_list.append(torch.tensor(cfg.prior_var))

    log_prob_func = define_model_log_prob(net, cfg.loss, x_train, y_train, params_flattened_list, params_shape_list,
                                          prior_list, cfg.tau_out, device=device, dt_string=unique_id)
    params_trained = mean_params.to(device) if cfg.init_prior else util.flatten(net).to(device).clone()
    params_init = params_trained[grad_ind].clone()
    print('Parameter size: ', params_init.shape[0])

    params_hmc = samplers.sample(log_prob_func, params_init, num_samples=cfg.num_samples, num_steps_per_sample=cfg.L,
                                 step_size=cfg.step_size, debug=False)
    np.save(f'{cfg.out_dir}hmc_params_{unique_id}.npy', params_hmc)


def validate(unique_id):
    """
    Function to evaluate the sampled network parameters
    Parameters
    ----------
    unique_id : str
        Unique id of the saved parameters

    Returns
    -------
    None
    """
    grad_ind = np.load(f'{cfg.prior_file}/gradient_indices_{cfg.prior_uid}.npy', allow_pickle=True)
    x_train, y_train, x_val, y_val = get_data()
    net = get_model(cfg.bias)

    params_shape_list = []
    params_flattened_list = []
    prior_list = []
    if cfg.load_prior:
        print("Loading priors from file")
        build_tau = False
        mean_params = torch.load(f'{cfg.prior_file}/means_flattened_{cfg.prior_uid}', map_location=device)
        std_params = torch.load(f'{cfg.prior_file}/stds_flattened_{cfg.prior_uid}', map_location=device
                                ) if cfg.load_std else cfg.prior_var * torch.ones_like(mean_params)
        prior_list = [mean_params[grad_ind], std_params[grad_ind]]
    else:
        build_tau = True

    for weights in net.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            prior_list.append(torch.tensor(cfg.prior_var))
    params_hmc = torch.tensor(np.load(f'{cfg.out_dir}hmc_params_{unique_id}.npy'))

    pred_list, log_prob_list = predict_model(net, params_hmc[cfg.burn:], x_val, y_val, model_loss=cfg.loss,
                                             tau_out=cfg.tau_out, prior_list=prior_list, dt_string=unique_id,
                                             grad_ind=None)

    print('\nExpected validation log probability: {:.2f}'.format(torch.stack(log_prob_list).mean()))
    sample_mse = []
    for pred in pred_list:
        sample_mse.append(((pred - y_val) ** 2).mean())
    print('\nExpected MSE: {:.2f}'.format(((pred_list.mean(0) - y_val) ** 2).mean()))
    print('\nFinal MSE: {:.2f}'.format(sample_mse[-1]))

    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })

    plt.figure(figsize=(8, 5))
    plt.plot(x_val.cpu().numpy(), pred_list[:].cpu().numpy().squeeze().T, 'C0', alpha=0.051)
    plt.plot(x_val.cpu().numpy(), y_val.cpu().numpy(), 'r', linewidth=3, label='True function')
    plt.plot(x_val.cpu().numpy(), pred_list.mean(0).cpu().numpy().squeeze().T, 'k', alpha=0.9, linewidth=3,
             label='Mean prediction')
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), '.C3', markersize=30, label='x train', alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.out_dir}VI_HMC_prediction.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt_string = datetime.now().strftime("%d%m%y%H%M%S")
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir, exist_ok=True)

    if not cfg.test:
        os.system(f'cp config.py  {cfg.out_dir}/config_{dt_string}.txt');
        for run_num in range(cfg.num_chains):
            draw_hmc_samples(f'{dt_string}_{run_num}')
            validate(f'{dt_string}_{run_num}')
    else:
        validate(cfg.test_dtstring)
