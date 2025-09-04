#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:48:47 2024

Functional network to implement VI-HMC

@author: ponkrshnan
"""
import torch
from torch import nn
import torch.nn.functional as F
import util
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Functional_Net:
    def __init__(self, depth, act, bias=True, mus=None, sigmas=None, sensitive_ind=None, model=None):
        self.depth = depth
        # self.learned_weights = learned_weights
        self.sensitive_ind = sensitive_ind
        self.model = model
        self.learned_mus = mus
        self.learned_sigmas = sigmas
        self.sampled_weights = mus
        self.vi_params = []
        self.bias = bias
        # print('Bias off in functional net')
        if act == 'relu':
            self.activation = F.relu
        elif act == 'tanh':
            self.activation = F.tanh
        elif act == 'sine':
            self.activation = Sin()
        else:
            raise ValueError('Activation should be relu, sine or tanh')

    def sample_weights(self, uid):
        self.sampled_weights = torch.normal(self.learned_mus, self.learned_sigmas)
        # print('Not sampling')
        # self.sampled_weights = self.learned_mus.clone()
        self.vi_params.append(self.sampled_weights)
        np.save(f'vi_params_{uid}.npy', torch.stack(self.vi_params).clone().detach().cpu())

    def functional_model(self, X, parameters):
        if self.learned_mus is None:
            weight = parameters
        else:
            flatten_weight = self.sampled_weights.clone()
            flatten_weight[self.sensitive_ind] = parameters
            weight = util.unflatten(self.model, flatten_weight)

        count = 0
        x = F.linear(X, weight[count], weight[count + 1])
        count += 2
        x = self.activation(x)
        for i in range(self.depth):
            x = F.linear(x, weight[count], weight[count + 1])
            count += 2
            x = self.activation(x)
        if self.bias:
            x = F.linear(x, weight[count], weight[count + 1])
        else:
            x = F.linear(x, weight[count])

        return x
