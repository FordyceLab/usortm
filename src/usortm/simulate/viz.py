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

from . import utils


def _simulate_simple_recovery(a, total_draws, n_reps, seed=None):
    """Simulate recovery curve for simple sampling without resynthesis."""
    rng = np.random.default_rng(seed)
    probs = a / a.sum()
    n_variants = len(a)
    curves = np.empty((n_reps, total_draws), dtype=float)

    for rep in range(n_reps):
        draws = rng.choice(n_variants, size=total_draws, replace=True, p=probs)
        seen = np.zeros(n_variants, dtype=bool)
        unique_count = 0
        curve = curves[rep]

        for idx, draw in enumerate(draws):
            if not seen[draw]:
                seen[draw] = True
                unique_count += 1
            curve[idx] = unique_count / n_variants

    xs = np.arange(1, total_draws + 1)
    ys = curves.mean(axis=0)
    return xs, ys


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


# Plotting function for recovery rates using utils.recovery_curve output
def plot_recovery_curve(xs, ys, library_size, t1=None, figsize=(8, 6), dpi=100):
    """Plot recovery curve given outputs from utils.recovery_curve.

    Parameters
    ----------
    xs : np.ndarray
        Sample sizes (1..T) returned by utils.recovery_curve.
    ys : np.ndarray
        Mean recovery fractions corresponding to `xs`.
    library_size : int
        Number of unique variants in the library (for coverage axis).
    t1 : int, optional
        If provided, marks the resynthesis point on the curve.
    figsize, dpi : tuple, int
        Matplotlib figure size and DPI.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Main curve
    ax.plot(xs, ys, color='blue', lw=2)

    # Optional resynthesis marker
    if t1 is not None and len(xs) > 0:
        idx = np.searchsorted(xs, t1, side='left')
        if idx < len(xs):
            ax.scatter([xs[idx]], [ys[idx]], color='red', s=40, zorder=5)

    # Formatting
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Recovery Fraction')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Coverage axis on top (in multiples of library_size)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    if library_size > 0 and len(xs) > 0:
        kmax = int(xs[-1] // library_size)
        if kmax >= 1:
            ticks = np.arange(1, kmax + 1) * library_size
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([f'{k}x' for k in range(1, kmax + 1)])
    ax2.set_xlabel('Coverage')

    plt.tight_layout()
    plt.show()

def plot_recovery_with_resynthesis(a, t1, t2, n_reps=100,
                                   figsize=(8,6), dpi=100,
                                   simple_sampling=False, seed=None):
    """
    Plot recovery curve with optional resynthesis or simple sampling.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    effective_seed = 0 if seed is None else seed

    if simple_sampling:
        total_draws = t1 + t2
        xs_simple, ys_simple = _simulate_simple_recovery(a, total_draws, n_reps, seed=seed)
        xs_res, ys_res = utils.recovery_curve_with_resynthesis(
            a, t1, t2, n_reps=n_reps, seed=effective_seed
        )
        ax.plot(xs_res, ys_res, color="blue", lw=2, label="With resynthesis")
        ax.plot(xs_simple, ys_simple, color="orange", lw=2, linestyle="--",
                label="Simple sampling")
        res_idx = np.searchsorted(xs_res, t1)
        res_x = xs_res[res_idx]
        res_y = ys_res[res_idx]
        ax.scatter([res_x], [res_y], color="red", s=50, zorder=5,
                   label=f"Resynthesis at {t1}")
        ax.text(res_x, res_y, f"  resynthesize", color="red",
                verticalalignment="bottom", fontsize=9)
    else:
        xs_res, ys_res = utils.recovery_curve_with_resynthesis(
            a, t1, t2, n_reps=n_reps, seed=effective_seed
        )
        ax.plot(xs_res, ys_res, color="blue", lw=2, label="With resynthesis")
        res_idx = np.searchsorted(xs_res, t1)
        res_x = xs_res[res_idx]
        res_y = ys_res[res_idx]
        ax.scatter([res_x], [res_y], color="red", s=50, zorder=5,
                   label=f"Resynthesis at {t1}")
        ax.text(res_x, res_y, f"  resynthesize", color="red",
                verticalalignment="bottom", fontsize=9)
    
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Recovery Fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
