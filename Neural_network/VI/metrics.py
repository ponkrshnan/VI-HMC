"""
Script containing metrics to train a Bayesian neural network
"""

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class ELBO(nn.Module):
    def __init__(self, ):
        super(ELBO, self).__init__()

    def forward(self, prediction, target, kl, beta, train_size, noise_param=None):
        assert not target.requires_grad
        return F.gaussian_nll_loss(prediction, target, noise_param * torch.ones_like(target),
                                   reduction='sum') + beta * kl

def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def mse(data_loader, model, noise_type=0):
    loss = nn.MSELoss()
    l_total = 0
    x, y = data_loader
    # for x,y in enumerate(data_loader):
    model.eval()
    if noise_type:
        y_pred, _, _ = model(x)
    else:
        y_pred, _ = model(x)
    l = loss(y_pred, y)
    l_total += l.item()
    # l_total = l_total/(i+1)
    return l_total


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "linear":
        beta = min(1, (1 - 1e-4) / num_epochs * epoch + 1e-4)
    elif beta_type == "step":
        beta = min(1, 1e-4 * 10 ** ((epoch + 1) // num_epochs))
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
