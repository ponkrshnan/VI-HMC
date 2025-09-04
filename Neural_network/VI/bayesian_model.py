# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:58:29 2024

This is a class implementing the Bayesian DeepONet architecture

@author: Ponkrshnan
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from layers import BBB_Linear, ModuleWrapper
from metrics import calculate_kl as KL_DIV


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Bayesian_Net(ModuleWrapper):
    def __init__(self, priors, layer_width, input_size, output_size, act='relu', bias_on = True ):
        super(Bayesian_Net, self).__init__()
        self.layer_width = layer_width
        self.layer_depth = len(layer_width) - 1
        self.input_size = input_size
        self.output_size = output_size
        self.bias_on = bias_on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priors = priors
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sine':
            self.act = Sin()
        else:
            raise ValueError("only relu. sine or tanh accepted in activation")

        self.net = self._model()

    def _model(self):
        mod_list = [BBB_Linear(self.input_size, self.layer_width[0], bias=True, priors=self.priors), self.act]
        i = -1  # when 1 hidden layer is used
        for i in range(self.layer_depth):
            mod_list.append(BBB_Linear(self.layer_width[i], self.layer_width[i + 1], bias=True, priors=self.priors))
            mod_list.append(self.act)
        mod_list.append(BBB_Linear(self.layer_width[i + 1], self.output_size, bias=self.bias_on, priors=self.priors))
        model = nn.Sequential(*mod_list)
        return model

    def forward(self, x):
        x = self.net(x)
        # x = torch.unsqueeze(x, 1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        # kl+=KL_DIV(self.priors['prior_mu'], self.priors['prior_sigma'], self.b_mu, self.b_sigma)

        return x, kl
