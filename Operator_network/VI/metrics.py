"""
Script containing metrics to train a Bayesian neural network
"""

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ELBO(nn.Module):
    def __init__(self, learn_noise=False, noise_type=0):
        super(ELBO, self).__init__()
        self.learn_noise = learn_noise
        self.noise_type = noise_type

    def forward(self, prediction, target, kl, beta, train_size, noise_param=None):
        assert not target.requires_grad
        if self.learn_noise:
            assert not (noise_param is None)
            if self.noise_type == 0:
                return F.gaussian_nll_loss(prediction, target, torch.exp(noise_param) * torch.ones_like(target),
                                           reduction='mean') * train_size + beta * kl
            else:
                return F.gaussian_nll_loss(prediction, target, torch.exp(noise_param),
                                           reduction='mean') * train_size + beta * kl
        else:
            return F.gaussian_nll_loss(prediction.reshape(target.shape), target, noise_param * torch.ones_like(target),
                                       reduction='mean') * train_size + beta * kl

def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def mse(data_loader, model, noise_type=0, dataset='Burgers'):
    loss = nn.MSELoss()
    l_total = 0
    for i, batch_data in enumerate(data_loader):
        if dataset == 'Cone':
            x = [batch_data['Xf'].to(device), batch_data['Xp'].to(device)]
            y = batch_data['Y'].to(device)
        else:
            x = [batch_data[0].to(device), batch_data[1].to(device)]
            y = batch_data[2].to(device)
        model.eval()
        if noise_type:
            y_pred, _, _ = model(*x)
        else:
            y_pred, _ = model(*x)
        l = loss(y_pred.reshape(y.shape), y)
        l_total += l.item()
    l_total = l_total / (i + 1)
    return l_total


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
