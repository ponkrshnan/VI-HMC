#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:48:47 2024

This is a class that implements a functional version of the DeepONet architecture. The functional model method takes
the inputs to the DeepONets and the parameters to evaluate the outputs.

@author: ponkrshnan
"""
import torch
import numpy as np
import torch.nn.functional as F
import util


class Functional_DeepONet:
    def __init__(self, depth_branch=9, depth_trunk=9, mus=None, sigmas=None,
                 activation='relu', sensitive_ind=None, model=None, impose_bc=True):
        self.depth_branch = depth_branch
        self.depth_trunk = depth_trunk
        self.learned_mus = mus
        self.learned_sigmas = sigmas
        self.sampled_weights = mus
        self.sensitive_ind = sensitive_ind
        self.model = model
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'tanh':
            self.act = F.tanh
        else:
            raise ValueError('activation should be relu or tanh')
        self.vi_params = []
        self.impose_bc = impose_bc

    def lambda_layer(self, x):
        return torch.stack([torch.sin(2 * np.pi * x), torch.sin(4 * np.pi * x),
                            torch.cos(2 * np.pi * x), torch.cos(4 * np.pi * x),
                            ], dim=2)

    def sample_weights(self, ):
        self.sampled_weights = torch.normal(self.learned_mus, self.learned_sigmas)
        # self.sampled_weights = self.learned_mus.clone()
        self.vi_params.append(self.sampled_weights)
        np.save('vi_params.npy', self.vi_params)

    def functional_model(self, X1, X2, parameters, vi_params=None):
        if self.learned_mus is None:
            weight = util.unflatten(self.model, parameters)
        else:
            flatten_weight = self.sampled_weights.clone() if vi_params is None else vi_params
            flatten_weight[self.sensitive_ind] = parameters
            weight = util.unflatten(self.model, flatten_weight)
        count = 1  # 0 th index is for the final bias
        xb = F.linear(X1, weight[count], weight[count + 1])
        count += 2
        xb = self.act(xb)
        for i in range(self.depth_branch - 2):
            xb = F.linear(xb, weight[count], weight[count + 1])
            count += 2
            xb = self.act(xb)
        xb = F.linear(xb, weight[count], weight[count + 1])
        count += 2

        if self.impose_bc:
            x_bc = self.lambda_layer(X2[:, :, 1])
            x_bc = torch.cat([X2[:, :, 0].unsqueeze(dim=2), x_bc], dim=2)
        else:
            x_bc = X2

        xtr = F.linear(x_bc, weight[count], weight[count + 1])
        count += 2
        xtr = self.act(xtr)
        for i in range(self.depth_trunk - 2):
            xtr = F.linear(xtr, weight[count], weight[count + 1])
            count += 2
            xtr = self.act(xtr)
        xtr = F.linear(xtr, weight[count], weight[count + 1])
        count += 2

        x = torch.einsum("...i,...i->...", xb, xtr)

        x = torch.unsqueeze(x, 1)
        x = x + weight[0]
        return x
