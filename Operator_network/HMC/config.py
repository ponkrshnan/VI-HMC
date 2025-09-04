# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

This is the input file for NUTS_DeepONets.py

@author: Ponkrshnan
"""
import numpy as np

width_branch = 100
width_trunk = 100
branch_depth = 9
trunk_depth = 9
in_branch = 101
in_trunk = 5  # sin 2pi x to sin 4pi x, cos 2pi x to cos 4pi x,  t
output_neurons = 100
activation = 'tanh'

# HMC Params
step_size = 1e-4
num_samples = 10
burn = num_samples // 10
load_prior = False
prior_file = 'Saved_models/'
init_prior = False
prior_var = 0.1 ** 2
post_var = 0.0214 ** 2
L = int(np.pi * post_var / (2 * step_size))
print("step size: ", L)

# Data params
dataset = 'Burgers'  # Cone or Burgers
sample_data = False  # subsample the input to trunk; Always false for cone
if dataset == 'Burgers':
    p = 10201
    N_train = 10
    N_valid = 10

loss = 'NLL'
tau_out = 1.0 ** 2  # if loss is regression, this is the precision parameter (1/var) if loss is NLL this is the
# variance

out_dir = 'Experiments/'
