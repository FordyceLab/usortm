import os
import time
import glob

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import lognorm

import bionumpy as bnp


# Plotting function for distribution of library counts
def plot_model_library_counts(model_dist):
    """
    Plot the distribution of library counts for a given model distribution.
    """
    # Plot distribution of counts
    size = 100000        # number of samples

    samples = model_dist.rvs(size)
    samples = [ np.log10(int(s)) for s in samples]

    q_10, q_90 = np.quantile(samples, [0.1, 0.9])
    plt.fill_betweenx([0, 1.75], q_10, q_90, color='green', alpha=0.2, zorder=-10, linewidth=0)

    plt.hist(samples, bins=100, density=True, alpha=0.6, zorder=10)

    # Label over 10th and 90th percentiles
    plt.vlines(q_10, color='green', linestyle='-', linewidth=1, ymin=0, ymax=1.75)
    plt.vlines(q_90, color='green', linestyle='-', linewidth=1, ymin=0, ymax=1.75)
    plt.text(q_10, 1.77, '10th', horizontalalignment='center', color='green')
    plt.text(q_90, 1.77, '90th', horizontalalignment='center', color='green')

    # Add fold-variation
    fold_change_10_90 = 10 ** (q_90 - q_10)
    plt.text((q_10 + q_90) / 2, 1.85, f'Fold-Variation:\n{fold_change_10_90:.2f}', horizontalalignment='center', color='black')

    plt.xlabel('Log$_{10}$(Variant Count)')
    plt.ylabel('Density')
    plt.ylim(0, 2.2)

    plt.show()


# Plotting function for recovery rates
def plot_recovery_curve(recovery_summary, depths, 
                        library_size, max_samples, 
                        figsize=(8, 6), dpi=100):
    """
    Plot the recovery curve.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Prepare data for scatter plot
    x_pos = depths
    means = [np.mean(recovery_summary[d]) for d in depths]
    stds = [np.std(recovery_summary[d]) for d in depths]
    
    # Create scatter plot with error bars
    ax.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=3, alpha=0.8, 
                markersize=1, elinewidth=1, capthick=1)
    ax.plot(x_pos, means, alpha=0.3, linestyle='--', color='blue', zorder=-10)  # Add connecting line
    
    # Formatting
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Simulated\nRecovery Rate')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(0, max_samples+1, 1000))
    ax.set_yticks([0, 0.5, 0.8, 0.9, 1])
    ax.grid(axis='y', alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    coverages = [d / library_size for d in depths]
    ax2.set_xticks([library_size*2, library_size*4, library_size*6, library_size*8])
    ax2.set_xticklabels([f'{c/library_size:.0f}x' for c in ax2.get_xticks()])
    ax2.set_xlabel('Coverage')
    
    plt.tight_layout()
    plt.show()