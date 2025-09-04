# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

This is the input file for main_HMC_splitting.py

@author: Ponkrshnan
"""
import numpy as np

# Net Params
width_branch = 100
width_trunk = 100
branch_depth = 9
trunk_depth = 9
in_branch = 101
in_trunk = 5  # sin 2pi x to sin 4pi x, cos 2pi x to cos 4pi x,  t
output_neurons = 100
activation = 'tanh'

# Data params
dataset = 'Burgers'  # Cone or Burgers
sample_data = False  #subsample the input to trunk; Always false for cone
if dataset == 'Burgers':
    p = 10201
    N_train = 1000
    N_valid = 1000
    # batch_size = 1000

# HMC Params
is_nuts = False
step_size = 1e-4  # 2.24e-09, 4e-08, 9.55e-05; 6.83e-09, 2.09e-7, 3.45e-4; 3.9e-09, 4.12e-06, 8.07e-05
num_samples = 1001
burn = num_samples // 2
load_prior = False
load_std = False
prior_file = f'Saved_models/{dataset}'
prior_uid = '020125162111'
init_prior = False
prior_var = 0.1 ** 2
post_var = 0.0214 ** 2  #std_max = 0.4055 [0.0799,0.0647,0.1027], std_min = 0.0016[2e-4,6e-4], std_median = 0.0108 [0.0082,0.0076,0.0214]
L = int(np.pi * post_var / (2 * step_size))
# L = 10
print("Number of steps: ", L)

split = True  # This is to keep tab if the split is on or not. Does nothing in the code
if split:
    num_splits = 2  # Num splits should split the data equally

loss = 'NLL'
tau_out = 1.0 ** 2  # if loss is regression, this is the precision parameter (1/var)
# if loss is NLL this is the variance

out_dir = f'samples/{dataset}'

evaluate = False
eval_uid = '01'
