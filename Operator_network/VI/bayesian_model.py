# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:58:29 2024

This is a class implementing the Bayesian DeepONet architecture

@author: Ponkrshnan
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from layers import BBB_Linear, ModuleWrapper
from metrics import calculate_kl as KL_DIV


class Bayesian_DeepONet(ModuleWrapper):
    def __init__(self, priors, neurons_branch=40, neurons_trunk=40, in_branch=1,
                 in_trunk=1, depth_branch=1, depth_trunk=4, output_neurons=20,
                 activation='relu', noise_type=0, noise_neurons=0, impose_bc=True):
        super(Bayesian_DeepONet, self).__init__()
        self.neurons_branch = neurons_branch
        self.neurons_trunk = neurons_trunk
        self.in_branch = in_branch
        self.in_trunk = in_trunk
        self.depth_branch = depth_branch
        self.depth_trunk = depth_trunk
        self.output_neurons = output_neurons
        self.noise_type = noise_type
        self.noise_neurons = noise_neurons
        # self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.b_mu = Parameter(torch.empty((1), device=self.device))
        self.b_rho = Parameter(torch.empty((1), device=self.device))
        self.priors = priors
        self.impose_bc = impose_bc
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('activation should be relu or tanh')
        self.b1 = self._branch()
        self.b2 = self._trunk()

        self.b_mu.data.normal_(*self.priors['posterior_mu_initial'])
        self.b_rho.data.normal_(*self.priors['posterior_rho_initial'])

    def lambda_layer(self, x):
        return torch.stack([torch.sin(2 * np.pi * x), torch.sin(4 * np.pi * x),
                            torch.cos(2 * np.pi * x), torch.cos(4 * np.pi * x),
                            ], dim=2)

    def _branch(self):
        mod_list = []
        mod_list.append(BBB_Linear(self.in_branch, self.neurons_branch, bias=True, priors=self.priors))
        mod_list.append(self.act)
        for i in range(self.depth_branch - 2):
            mod_list.append(BBB_Linear(self.neurons_branch, self.neurons_branch, bias=True, priors=self.priors))
            mod_list.append(self.act)
        mod_list.append(BBB_Linear(self.neurons_branch, self.output_neurons, bias=True, priors=self.priors))
        b1 = nn.Sequential(*mod_list)
        return b1

    def _trunk(self):
        mod_list = []
        mod_list.append(BBB_Linear(self.in_trunk, self.neurons_trunk, bias=True, priors=self.priors))
        mod_list.append(self.act)
        for i in range(self.depth_trunk - 2):
            mod_list.append(BBB_Linear(self.neurons_trunk, self.neurons_trunk, bias=True, priors=self.priors))
            mod_list.append(self.act)
        mod_list.append(BBB_Linear(self.neurons_trunk, self.output_neurons, bias=True, priors=self.priors))
        b2 = nn.Sequential(*mod_list)
        return b2

    def forward(self, x_branch, x_trunk):
        x1 = self.b1(x_branch)
        if self.impose_bc:
            x_bc = self.lambda_layer(x_trunk[:, :, 1])
            x_bc = torch.cat([x_trunk[:, :, 0].unsqueeze(dim=2), x_bc], dim=2)
            x2 = self.b2(x_bc)
        else:
            x2 = self.b2(x_trunk)

        try:
            tmp = self.noise_type
        except:
            self.noise_type = 0

        if self.noise_type:
            x = torch.einsum("bi,bi->b", x1[:, :self.noise_neurons], x2[:, :self.noise_neurons])
            noise_param = torch.einsum("bi,bi->b", x1[:, self.noise_neurons:], x2[:, self.noise_neurons:])
        else:
            x = torch.einsum("...i,...i->...", x1, x2)

        # x = torch.unsqueeze(x, 0)

        self.b_sigma = torch.log1p(torch.exp(self.b_rho))
        if self.training:
            b_eps = torch.empty(self.b_mu.size()).normal_(0, 1).to(self.device)
            b = self.b_mu + b_eps * self.b_sigma
        else:
            b = self.b_mu
        x += b

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        kl += KL_DIV(self.priors['prior_mu'], self.priors['prior_sigma'], self.b_mu, self.b_sigma)
        if self.noise_type:
            return x, noise_param, kl
        else:
            return x, kl
