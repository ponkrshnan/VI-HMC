# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

This is the input file for both main_VI_HMC_burgers.py and VI_HMC_splitting.py

@author: Ponkrshnan
"""
import numpy as np

#  Net Params
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
sample_data = False  # subsample the input to trunk; Always false for cone
if dataset == 'Burgers':
    p = 10201
    N_train = 1000
    N_valid = 1000

# HMC Params
step_size = 1e-4
num_samples = 1000
burn = 100
load_prior = False
load_std = False
prior_file = f'../VI/Saved_models/{dataset}'  # Location of saved sensitivity scores
prior_uid = '040925151056'  # Unique id (time stamp) of VI model
init_prior = False
if init_prior:
    sample_prior = True
prior_var = 0.1 ** 2
post_var = 0.0214 ** 2
L = int(np.pi * post_var / (2 * step_size))
print("Number of steps: ", L)

split = False  # This is to keep tab if the split is on or not. Does nothing in the code
if split:
    num_splits = 100  # Num splits should split the data equally

loss = 'NLL'
tau_out = 1.0 ** 2  # if loss is regression, this is the precision parameter (1/var)
# if loss is NLL this is the variance

evaluate = False
eval_dt_string = '140125164316'

out_dir = f'samples/{dataset}'
