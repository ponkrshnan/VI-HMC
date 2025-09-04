"""
Input file for sensitivity.py
"""
width = 2*[10]
act = 'tanh'
bias = True
N_tr, N_val, noise = (20, 300, 5e-2)
load_data = False

ckpt_dir = 'checkpoints/Sensitivity'
unique_id = '040925121902'  # Unique id (Time stamp) of the saved VI model

importance_threshold = 0.90  # captured variance if Taylor else threshold
