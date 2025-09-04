# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

This is the input file for sensitivity.py

@author: Ponkrshnan
"""

# Training params
batch_size = 1

# Network params
layer_width = 100
branch_depth = 9
trunk_depth = 9
in_branch = 101
in_trunk = 5  # if enforcing bc: sin 2pi x to sin 4pi x, cos 2pi x to cos 4pi x,  t
output_neurons = 100
activation = 'tanh'

# Data params
dataset = 'Burgers'  # Cone or Burgers
if dataset == 'Burgers':
    p = 100
    N_train = 1000
    N_valid = 1000

# File locations
save_loc = f'Saved_models/{dataset}'  # location of saved sensitivity scores if loading from file
model_loc = f'checkpoints/{dataset}'  # location of trained DeepONet model

importance_threshold = 0.90  # sensitivity threshold
uid = '040925151056'  # unique id (time stamp) of the VI model to compute sensitivity for

load_saved_sens = False
