# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:58:29 2024

@author: Ponkrshnan
"""
import torch
import torch.nn as nn
import numpy as np

class DeepONet(nn.Module):
    def __init__(self, width_branch=40, width_trunk=20, in_branch=1, in_trunk=1,
                 depth_branch=1, depth_trunk=4, activation='relu', output_neurons=None,
                 impose_bc = True):
        super(DeepONet, self).__init__()
        self.width_branch = width_branch
        self.width_trunk = width_trunk
        self.in_branch = in_branch
        self.in_trunk = in_trunk
        self.depth_branch = depth_branch
        self.depth_trunk = depth_trunk
        if output_neurons is None:
            self.output_neurons = width_branch
        else:    
            self.output_neurons = output_neurons
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('Activation can be tanh or relu')
        self.b1 = self._branch()
        self.b2 = self._trunk()
        self.impose_bc = impose_bc
        
    def lambda_layer(self,x):
        return torch.stack([torch.sin(2*np.pi*x), torch.sin(4*np.pi*x), 
                            torch.cos(2*np.pi*x), torch.cos(4*np.pi*x), 
                            ],dim=2)
    
    def _branch(self):
        modules = []
        modules.append(nn.Linear(self.in_branch, self.width_branch))
        modules.append(self.act)
        for i in range(self.depth_branch-2):
            modules.append(nn.Linear(self.width_branch, self.width_branch))
            modules.append(self.act)
        modules.append(nn.Linear(self.width_branch, self.output_neurons))
        b1 = nn.Sequential(*modules)
        return b1
    
    def _trunk(self):
        modules = []
        modules.append(nn.Linear(self.in_trunk, self.width_trunk))
        modules.append(self.act)
        for i in range(self.depth_trunk-2):
            modules.append(nn.Linear(self.width_trunk, self.width_trunk))
            modules.append(self.act)
        modules.append(nn.Linear(self.width_trunk, self.output_neurons))
        b2 = nn.Sequential(*modules)
        return b2
        
    def forward(self, x1, x2):
        x1_out = self.b1(x1)
        if self.impose_bc:
            x_bc = self.lambda_layer(x2[:,:,1])
            x_bc = torch.cat([x2[:,:,0].unsqueeze(dim=2),x_bc],dim=2)
            x2_out = self.b2(x_bc)
        else:
            x2_out = self.b2(x2)
        y = torch.einsum("...i,...i->...", x1_out, x2_out)
        # x = torch.unsqueeze(x, 1)
        y+=self.b
        return y
