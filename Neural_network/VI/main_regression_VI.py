# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:54:28 2024

This scripts trains a Bayesian neural network using VI on the regression problem

@author: Ponkrshnan
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, lr_scheduler
import os
from datetime import datetime
from bayesian_model import Bayesian_Net
import config as cfg
import metrics
import matplotlib

matplotlib.rcParams['text.usetex'] = True

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


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
    tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        Training and validation data.
    """

    if cfg.load_data_from_file:
        print('Loading data from file')
        x_train = torch.load('../Data/x_train')
        y_train = torch.load('../Data/y_train')
        x_val = torch.load('../Data/x_val')
        y_val = torch.load('../Data/y_val')
    else:

        x_val = torch.linspace(-1.2, 1.2, N_val).view(-1, 1)
        # y_val = (0.4 * torch.sin(4 * x_val) + 0.5 * torch.cos(3 * x_val)).view(-1, 1)
        y_val = 4*torch.sin(4*x_val) + 5*torch.cos(12*x_val)

        x_train = torch.cat((torch.linspace(-1, -0.2, N_tr // 2), torch.linspace(0.2, 1, N_tr // 2))).view(-1, 1)
        # y_train = (0.4 * torch.sin(4 * x_train) + 0.5 * torch.cos(3 * x_train)) + torch.randn_like(x_train) * noise
        y_train = 4*torch.sin(4*x_train) + 5*torch.cos(12*x_train) + torch.randn_like(x_train)*noise

        # torch.save(x_val, 'Data/x_val')
        # torch.save(x_train,'Data/x_train')
        # torch.save(y_val,'Data/y_val')
        # torch.save(y_train,'Data/y_train')
        torch.manual_seed(0)

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)
    return (x_train, y_train), (x_val, y_val)


def train_model(train_loader, model, loss, optimizer, train_size, num_batches, num_ens=1, beta_type=0.1, epoch=None,
                num_epochs=None, noise_param=None):
    """
    Function to train the regression model
    Parameters
    ----------
    train_loader : tuple
        Training data loader
    model : torch.nn.module
        Bayesian neural network model to train
    loss : function
        Loss function to optimize
    optimizer : torch.Optimizer
        PyTorch optimizer
    train_size : int
        Length of the training data
    num_batches : int
        Number of batches
    num_ens : int
        Number of ensembles (MC samples) to compute the loss function
    beta_type : {float, int, str}
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
        Training loss
    """
    l_total = 0
    model.train()
    x, y = train_loader
    optimizer.zero_grad()
    beta = metrics.get_beta(0, num_batches, beta_type, epoch, num_epochs)
    l = 0.0
    for j in range(num_ens):
        pred, _kl = model(x)
        l += loss(pred, y, _kl, beta, train_size, noise_param)
    l = l / num_ens

    l_total += l.item()
    l.backward()
    optimizer.step()
    return l_total


def validate_model(valid_loader, model, loss, valid_size, beta_type, num_batches, epoch=None, num_epochs=None,
                   noise_param=None):
    """
    This function evaluates the loss function on the validation data
    Parameters
    ----------
    valid_loader : tuple
        A dataloader containing the validation dataset
    model : torch.nn.module
        Bayesian neural network model to validate
    loss : function
        Loss function
    valid_size : int
        Length of the validation data
    beta_type :  {int, float, str}
        Modifier to the KL loss
    num_batches : int
        Number of batches
    epoch : int
        Current epoch
    num_epochs : int
        Number of epochs
    noise_param : torch.Tensor
        Parameter describing variance of Gaussian NLL. log(variance) if trainable else variance

    Returns
    -------
    torch.Tensor
        Validation loss
    """
    l_total = 0
    model.eval()
    x, y = valid_loader
    y_pred, kl = model(x)
    beta = metrics.get_beta(0, num_batches, beta_type, epoch, num_epochs)
    l = loss(y_pred, y, kl, beta, valid_size, noise_param)
    l_total += l.item()
    return l_total


def do_uq(valid_loader, model, num_uq_samps):
    """
    Function to quantify uncertainties in the prediction
    Parameters
    ----------
    valid_loader : tuple
        Data loader
    model : torch.nn.module
        Bayesian neural network model
    num_uq_samps : int
        Number of samples to perform UQ

    Returns
    -------
    torch.Tensor
        Prediction using all the parameter samples
    """
    model.train()
    x, y = valid_loader
    y_pred = []
    for i in range(num_uq_samps):
        pred, _ = model(x)
        y_pred.append(pred.detach().numpy())
    return torch.tensor(y_pred)


def plot_uq(train_loader, valid_loader, model):
    """
    Plot the predictions and uncertainties
    Parameters
    ----------
    train_loader : tuple
        Training data
    valid_loader : tuple
        Validation data
    model : torch.nn.module
        Bayesian neural network

    Returns
    -------
    None
    """
    x_val, y_val = valid_loader
    x_train, y_train = train_loader
    pred_list = do_uq(valid_loader, model, cfg.num_uq_samps)
    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })
    plt.figure(figsize=(8, 5))
    plt.plot(x_val.cpu().numpy(), pred_list[:].cpu().numpy().squeeze().T, 'C0', alpha=0.051)
    plt.plot(x_val.cpu().numpy(), y_val.cpu().numpy(), 'r', linewidth=3, label='True function')
    plt.plot(x_val.cpu().numpy(), pred_list.mean(0).cpu().numpy().squeeze().T, 'k', alpha=0.9, linewidth=3,
             label='Mean prediction')
    plt.plot(x_train.cpu().numpy(), y_train.cpu().numpy(), '.C3', markersize=30, label='x train', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/VI_prediction.pdf', dpi=600)
    plt.show()


def plot_metrics(train_metrics):
    """
    Function to plot and save the training and validation losses
    Parameters
    ----------
    train_metrics : list
        Metrics saved during VI training

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))
    train_metrics = np.array(train_metrics)
    plt.plot(train_metrics[:, 0], label='Train loss')
    plt.plot(train_metrics[:, 1], label='Validation loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/VI_loss.pdf', dpi=600)

    plt.figure(figsize=(8, 5))
    plt.plot(train_metrics[:, 2], label='Train mse')
    plt.plot(train_metrics[:, 3], label='Validation mse')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/VI_mse.pdf', dpi=600)
    plt.show()


def evaluate():
    """
    Function to evaluate a pre trained model
    Returns
    -------
    None
    """
    train_loader, valid_loader = get_data(cfg.train_size, cfg.valid_size, cfg.noise)
    trained_model = torch.load(cfg.model_file)
    model = trained_model['net']
    train_metrics = trained_model['metrics']
    plot_uq(train_loader, valid_loader, model)
    plot_metrics(train_metrics)


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

    tr_size = cfg.train_size
    vld_size = cfg.valid_size
    noise = cfg.noise
    train_loader, valid_loader = get_data(tr_size, vld_size, noise)
    num_tr_batches = 1
    num_val_batches = 1
    if cfg.restart:
        trained_model = torch.load(cfg.model_file)
        model = trained_model['net']
    else:
        model = Bayesian_Net(cfg.priors, cfg.layer_width, cfg.input_size, cfg.output_size, cfg.activation, cfg.bias_on)
    optimizer = Adam(model.parameters(), lr=cfg.lr_start)
    noise_param = torch.tensor(noise ** 2)

    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.lr_patience, min_lr=1e-5)
    loss = metrics.ELBO().to(device)
    train_metrics = []
    valid_loss_min = float("Inf")
    for epoch in range(cfg.epochs):
        train_loss = train_model(train_loader, model, loss, optimizer, tr_size, num_tr_batches, cfg.num_ens,
                                 cfg.beta_type, epoch=epoch, num_epochs=cfg.beta_epochs, noise_param=noise_param)
        valid_loss = validate_model(valid_loader, model, loss, vld_size, cfg.beta_type, num_val_batches, epoch=epoch,
                                    num_epochs=cfg.beta_epochs, noise_param=noise_param)
        lr_sched.step(valid_loss)
        train_mse = metrics.mse(train_loader, model)
        valid_mse = metrics.mse(valid_loader, model)
        train_metrics.append([train_loss, valid_loss, train_mse, valid_mse])
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining MSE: {:.6f} \tValidation MSE: {:.6f} \tNoise: {:.6f}'.format(
                epoch, train_loss, valid_loss, train_mse, valid_mse, noise), file=open(op_file, 'a'))
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining MSE: {:.6f} \tValidation MSE: {:.6f} \tNoise: {:.6f}'.format(
                epoch, train_loss, valid_loss, train_mse, valid_mse, noise))
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
    plot_uq(train_loader, valid_loader, model)
    plot_metrics(train_metrics)


if __name__ == '__main__':
    if cfg.test:
        evaluate()
    else:
        run()
