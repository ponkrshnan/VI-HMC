# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:54:28 2024

This is a script to train and validate a Bayesian DeepONet with VI

@author: Ponkrshnan
"""

import os
from datetime import datetime
import torch
from torch.optim import Adam, lr_scheduler
import config as cfg
import metrics
import utils
from bayesian_model import Bayesian_DeepONet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(train_loader, model, loss, optimizer, train_size, num_batches, num_ens=1, beta_type=0.1, epoch=None,
                num_epochs=None, noise_param=None):
    """
    This is a function to train a Bayesian neural network model
    Parameters
    ----------
    train_loader : torch.DataLoader
        A data loader containing the training data
    model : torch.nn.Module
        Bayesian neural network model to train
    loss : function
        Loss function to be optimized
    optimizer : torch.Optimizer
        A PyTorch optimizer
    train_size : int
        Length of the training data (to evaluate the loss function)
    num_batches : int
        Number of batches in the training data
    num_ens : int
        Number of ensembles (MC samples) to compute the loss function
    beta_type : {int, float, str}
        Modifier to the KL loss
    epoch : int
        Current epoch
    num_epochs : int
        Number of epochs
    noise_param : torch.Tensor
        Parameter describing variance of Gaussian NLL. log(variance) if trainable else variance

    Returns
    -------
    torch.Tensor
        Training loss (mean of all batches)
    """
    l_total = 0
    for i, batch_data in enumerate(train_loader):
        if cfg.dataset == 'Cone':
            x = [batch_data['Xf'].to(device), batch_data['Xp'].to(device)]
            y = batch_data['Y'].to(device)
        else:
            x = [batch_data[0].to(device), batch_data[1].to(device)]
            y = batch_data[2].to(device)
        model.train()
        optimizer.zero_grad()
        beta = metrics.get_beta(i, num_batches, beta_type, epoch, num_epochs)
        l = 0.0
        for j in range(num_ens):
            if cfg.noise_type:
                pred, noise_param, _kl = model(*x)
            else:
                pred, _kl = model(*x)
            l += loss(pred, y, _kl, beta, train_size, noise_param)
        l = l / num_ens

        l_total += l.item()
        l.backward()
        optimizer.step()
    l_total = l_total / (i + 1)
    return l_total


def validate_model(valid_loader, model, loss, valid_size, beta_type, num_batches, noise_param=None):
    """
    This function evaluates the loss function on the validation data
    Parameters
    ----------
    valid_loader : torch.DataLoader
        A dataloader containing the validation dataset
    model : torch.nn.Module
        Bayesian neural network model to validate
    loss : function
        Loss function
    valid_size : int
        Length of the validation data
    beta_type : {int, float, str}
        Modifier to the KL loss
    num_batches : int
        Number of batches
    noise_param : torch.Tensor
        Parameter describing variance of Gaussian NLL. log(variance) if trainable else variance

    Returns
    -------
    torch.Tensor
        Validation loss (mean of all batches)
    """
    l_total = 0
    for i, batch_data in enumerate(valid_loader):
        if cfg.dataset == 'Cone':
            x = [batch_data['Xf'].to(device), batch_data['Xp'].to(device)]
            y = batch_data['Y'].to(device)
        else:
            x = [batch_data[0].to(device), batch_data[1].to(device)]
            y = batch_data[2].to(device)

        model.eval()
        if cfg.noise_type:
            y_pred, noise_param, kl = model(*x)
        else:
            y_pred, kl = model(*x)
        beta = metrics.get_beta(i, num_batches, beta_type, None, None)
        l = loss(y_pred, y, kl, beta, valid_size, noise_param)
        l_total += l.item()
    l_total = l_total / (i + 1)
    return l_total


def run():
    """
    This function performs the preprocessing, trains a neural network and performs the postprocessing steps
    Returns
    -------
    None
    """
    dt_string = datetime.now().strftime("%d%m%y%H%M%S")
    ckpt_name = f'{cfg.ckpt_dir}/max_model_{dt_string}.pt'
    ckpt_last = f'{cfg.ckpt_dir}/last_model_{dt_string}'
    op_file = f'{cfg.ckpt_dir}/{dt_string}_output.txt'

    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.system(f'cp config.py  {cfg.ckpt_dir}/config_{dt_string}.txt');

    train_loader, valid_loader, tr_size, vld_size = utils.get_data(cfg)
    num_tr_batches = utils.get_numbatches(train_loader)
    num_val_batches = utils.get_numbatches(valid_loader)
    model = Bayesian_DeepONet(cfg.priors, cfg.width_branch, cfg.width_trunk, cfg.in_branch,
                              cfg.in_trunk, cfg.branch_depth, cfg.trunk_depth,
                              cfg.output_neurons + cfg.noise_neuron, cfg.activation,
                              cfg.noise_type, cfg.noise_neuron,
                              impose_bc=True if cfg.dataset == 'Burgers' else False)
    if cfg.learn_noise and cfg.noise_type == 0:
        noise_param = torch.nn.Parameter(torch.randn((1), device=device))
        optimizer = Adam(list(model.parameters()) + [noise_param], lr=cfg.lr_start)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.lr_start)
        noise_param = torch.tensor(cfg.noise_param)

    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.lr_patience, min_lr=1e-5)
    loss = metrics.ELBO(cfg.learn_noise, cfg.noise_type).to(device)
    train_metrics = []
    valid_loss_min = float("Inf")
    for epoch in range(cfg.epochs):
        train_loss = train_model(train_loader, model, loss, optimizer, tr_size, num_tr_batches, cfg.num_ens,
                                 cfg.beta_type, noise_param=noise_param)
        valid_loss = validate_model(valid_loader, model, loss, vld_size, cfg.beta_type, num_val_batches,
                                    noise_param=noise_param)
        lr_sched.step(valid_loss)
        train_mse = metrics.mse(train_loader, model, cfg.noise_type, cfg.dataset)
        valid_mse = metrics.mse(valid_loader, model, cfg.noise_type, cfg.dataset)
        if cfg.learn_noise and cfg.noise_type == 0:
            alea_unc = torch.exp(noise_param).detach().numpy().item()
            train_metrics.append([train_loss, valid_loss, train_mse, valid_mse, alea_unc])
        else:
            alea_unc = 0
            train_metrics.append([train_loss, valid_loss, train_mse, valid_mse])
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining MSE: {:.6f} \tValidation MSE: {'
            ':.6f} \tNoise: {:.6f}'.format(
                epoch, train_loss, valid_loss, train_mse, valid_mse, alea_unc), file=open(op_file, 'a'))
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining MSE: {:.6f} \tValidation MSE: {'
            ':.6f} \tNoise: {:.6f}'.format(
                epoch, train_loss, valid_loss, train_mse, valid_mse, alea_unc))
        checkpt = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'lr_sched': lr_sched,
                   'metrics': train_metrics,
                   'net': model
                   }

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss), file=open(op_file, 'a'))
            torch.save(checkpt, ckpt_name)
            valid_loss_min = valid_loss
        # save model every n epoch
        if epoch % cfg.n_save == 0:
            torch.save(checkpt, f'{ckpt_last}_{epoch}.pt')
    torch.save(checkpt, f'{ckpt_last}.pt')


if __name__ == '__main__':
    run()
