"""
Additional function used by the scripts
"""

import torch
from matplotlib import pyplot as plt
import numpy as np


def unflatten(model, flattened_params):
    if flattened_params.dim() != 1:
        raise ValueError('Expecting a 1d flattened_params')
    params_list = []
    i = 0
    for val in list(model.parameters()):
        length = val.nelement()
        param = flattened_params[i:i + length].view_as(val)
        params_list.append(param)
        i += length

    return params_list


def plot_hists(mean_grads, cfg):
    # Create the histogram
    plt.rcParams.update({
        'font.size': 22,  # Base font size
    })
    plt.figure(figsize=(8, 5))
    plt.hist(mean_grads, bins=np.linspace(0, 1, 100), edgecolor='black', color='blue', alpha=0.7)

    # Customize the plot
    # plt.title('Histogram of gradients' if cfg.importance == 'dydw' else 'Histogram of Sensitivity')
    plt.xlabel('Sensitivity ($S_i^2$)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Increase the tick parameters
    plt.xticks()
    plt.yticks()

    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/hist1.pdf', dpi=600)
    # Show the plot
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(mean_grads, bins=np.linspace(0, np.percentile(mean_grads, 99), 100), edgecolor='black', color='blue',
             alpha=0.7)

    # Customize the plot
    # plt.title('Histogram of gradients' if cfg.importance == 'dydw' else 'Histogram of Sensitivity')
    plt.xlabel('Sensitivity ($S_i^2$)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Increase the tick parameters
    plt.xticks()
    plt.yticks()

    plt.tight_layout()
    plt.savefig(f'{cfg.ckpt_dir}/hist2.pdf', dpi=600)
    # Show the plot
    plt.show()


def plot_grads(grads_unflattened):
    for mat in grads_unflattened:
        if mat.dim() == 1:
            mat = mat.unsqueeze(1)
        elif mat.dim() == 0:
            mat = mat.unsqueeze(0)
            mat = mat.unsqueeze(1)

        plt.imshow(mat, cmap='coolwarm', vmin=0, vmax=1e-8)
        # plt.colorbar(pad=0.1)
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.show()


def flatten_mean_std(unique_id):
    saved_model = torch.load(f'checkpoints/Regression/max_model_{unique_id}.pt')
    mean_params = []
    std_params = []
    for name, param in saved_model['model'].items():
        if 'mu' in name:
            mean_params.append(param.flatten())
        elif 'rho' in name:
            std_params.append(torch.log1p(torch.exp(param)).flatten())
    return torch.cat(mean_params), torch.cat(std_params)
