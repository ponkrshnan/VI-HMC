#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:40:23 2024

Input file to regeression_hmc.py

@author: ponkrshnan
"""
import numpy as np

# Network parameters
N_tr = 20
N_val = 300
width = 2*[10]
act = 'tanh'
depth = len(width) - 1
bias = True

# Hmc parameters
tau = 1.  # 1/prior variance
step_size = 1e-4
num_samples = 1000
post_var = 0.2024 ** 2
L = int(np.pi * post_var / (2 * step_size))
print('Step size:', L)
tau_out = 1 / 5e-2 ** 2  # 1/noise_variance
burn = num_samples // 5
num_chains = 1

out_dir = 'samples'
test = False
test_dtstring = '291024130550_0'
