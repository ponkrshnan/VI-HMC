#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:32:47 2024

Input file to main_regression_VI.py

@author: ponkrshnan
"""

train_size = 10
valid_size = 300
noise = 5e-2  # std of noise or parameter describing precision
load_data_from_file = True

priors = {
    'prior_mu': 0,
    'prior_sigma': 1.0,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-3, 0.1),  # (mean, std) normal_
}

layer_width = 2*[10]
input_size = 1
output_size = 1
activation = 'tanh'
bias_on = True  # last layer Bias off if False
lr_start = 1e-2
lr_patience = 5000
epochs = 10_000
num_ens = 10
beta_type = 1.0  # 'Blundell', 'Standard', etc. Use float for const value
beta_epochs = None
n_save = epochs // 10
num_uq_samps = 500

ckpt_dir = 'checkpoints/Regression'

test = False
restart = False
model_file = 'checkpoints/Regression/max_model_121124163740.pt'
