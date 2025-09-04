# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:13:03 2024

Input file for VI_HMC.py

@author: Ponkrshnan
"""
import numpy as np

# Net Params
N_tr = 20
N_val = 300
width = 2 * [10]
act = 'tanh'
depth = len(width) - 1
bias = True  # bias in the last layer

# HMC Params
step_size = 5e-4
num_samples = 100
burn = num_samples // 5
prior_var = 1.0
post_var = 0.2501 ** 2
L = int(np.pi * post_var / (2 * step_size))
print("step size: ", L)
loss = 'NLL'  # regression or NLL
tau_out = 5e-2 ** 2  # Measure of precision:1/variance if Regression or variance if NLL.Affects training data generation
num_chains = 10  # number of HMC chains

# Directories and other options
out_dir = 'samples_large_network/try/'
load_prior = False
load_std = False
prior_file = '../VI/checkpoints/Sensitivity'  # location of sensitivity results
prior_uid = '040925121902'  # Unique id (time stamp) of saved VI model
init_prior = False
test = False
test_dtstring = '080725142421_0'
