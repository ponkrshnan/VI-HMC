# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:58:29 2024

This class implements a deterministic DeepONet architecture

@author: Ponkrshnan
"""
import torch
import numpy as np
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, neurons=40, in_branch=1, in_trunk=1, depth_branch=1,
                 depth_trunk=4, output_neurons=100, activation='relu', impose_bc=True):
        super(DeepONet, self).__init__()
        self.neurons = neurons
        self.in_branch = in_branch
        self.in_trunk = in_trunk
        self.depth_branch = depth_branch
        self.depth_trunk = depth_trunk
        self.output_neurons = output_neurons
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('activation should be relu or tanh')
        self.b1 = self._branch()
        self.b2 = self._trunk()
        self.impose_bc = impose_bc

    def _branch(self):
        modules = []
        modules.append(nn.Linear(self.in_branch, self.neurons))
        modules.append(self.act)
        for i in range(self.depth_branch - 2):
            modules.append(nn.Linear(self.neurons, self.neurons))
            modules.append(self.act)
        modules.append(nn.Linear(self.neurons, self.output_neurons))
        b1 = nn.Sequential(*modules)
        return b1

    def _trunk(self):
        modules = []
        modules.append(nn.Linear(self.in_trunk, self.neurons))
        modules.append(self.act)
        for i in range(self.depth_trunk - 2):
            modules.append(nn.Linear(self.neurons, self.neurons))
            modules.append(self.act)
        modules.append(nn.Linear(self.neurons, self.output_neurons))
        b2 = nn.Sequential(*modules)
        return b2

    def lambda_layer(self, x):
        return torch.stack([torch.sin(2 * np.pi * x), torch.sin(4 * np.pi * x),
                            torch.cos(2 * np.pi * x), torch.cos(4 * np.pi * x),
                            ], dim=2)

    def forward(self, x_branch, x_trunk):
        x1 = self.b1(x_branch)
        if self.impose_bc:
            x_bc = self.lambda_layer(x_trunk[:, :, 1])
            x_bc = torch.cat([x_trunk[:, :, 0].unsqueeze(dim=2), x_bc], dim=2)
            x2 = self.b2(x_bc)
        else:
            x2 = self.b2(x_trunk)
        x = torch.einsum("...i,...i->...", x1, x2)
        x = torch.unsqueeze(x, 1)
        x += self.b
        return x
