# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

Configuration file for main_bayesian_deeponet.py

@author: Ponkrshnan
"""

# Training params
batch_size = 128
epochs = 10
lr_start = 1e-3
lr_patience = 500
n_save = int(epochs / 10)

# Network params
width_branch = 100
width_trunk = 100
branch_depth = 9
trunk_depth = 9
in_branch = 101
in_trunk = 5  # if enforcing bc: sin 2pi x to sin 4pi x, cos 2pi x to cos 4pi x,  t
output_neurons = 100
activation = 'tanh'

# Data params
dataset = 'Burgers'  # Cone or Burgers
if dataset == 'Burgers':
    p = 10201
    N_train = 1000
    N_valid = 1000

# Learning params
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}
num_ens = 5
beta_type = 1.0  # 'Blundell', 'Standard', etc. Use float for const value

# Noise params
learn_noise = False  # Learn Aleatoric uncertainty?
noise_type = 0  # 0: Homoscedastic, 1: Hetroscedastic
noise_neuron = 0

if not learn_noise:
    noise_type = 0
    noise_param = 1.0 ** 2  # noise variance in the NLL loss

ckpt_dir = f'checkpoints/{dataset}'  # checkpoint directory to save results
