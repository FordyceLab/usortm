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
from scipy.stats import norm

import bionumpy as bnp


def model_distribution(N, skew=5, seed=0):
    """
    Simulate a log-normal abundance distribution with a specified skew.

    Parameters
    ----------
    N : int
        Number of unique items (species, variants, etc.) in the pool.
    skew : float
        Fold difference between the 90th and 10th percentiles (Q90/Q10).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    a : np.ndarray of shape (N,)
        Simulated abundance values for each item. Mean abundance ≈ 1.
    """
    rng = np.random.default_rng(seed)

    # infer sigma from skew = Q90/Q10
    z90, z10 = norm.ppf(0.9), norm.ppf(0.1)
    sigma = np.log(skew) / (z90 - z10)
    mu = -0.5 * sigma**2   # ensures mean ~1

    a = rng.lognormal(mean=mu, sigma=sigma, size=N)
    return a

def take_sample(a, t, recovery=False, seed=None):
    """
    Perform an explicit random sampling experiment.

    Parameters
    ----------
    a : np.ndarray
        Abundance distribution (length N).
    t : int
        Number of draws with replacement.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    frac_recovered : float
        Fraction of unique items recovered in this sample.
    """
    rng = np.random.default_rng(seed)
    probs = a / a.sum()
    draws = rng.choice(len(a), size=t, replace=True, p=probs)
    unique = len(np.unique(draws))
    if recovery:
        return draws, unique / len(a)
    else:
        return draws

# def recovery_curve(a, n_samples=3000):
#     """
#     Compute the expected recovery curve given an abundance distribution.

#     Uses Poisson approximation: probability an item is unseen after t draws
#     ≈ exp(-t * a_i / (N * mean(a))).

#     Parameters
#     ----------
#     a : np.ndarray
#         Abundance distribution (length N).
#     n_samples : int
#         Maximum number of samples to evaluate.

#     Returns
#     -------
#     xs : np.ndarray
#         Sample sizes, from 0 to n_samples.
#     ys : np.ndarray
#         Expected recovery fraction for each sample size.
#     """
#     N = len(a)
#     m = a.mean()
#     xs = np.arange(n_samples+1)
#     ys = [1 - np.mean(np.exp(-t * a / (N*m))) for t in xs]
#     return xs, np.array(ys)

def recovery_curve_with_resynthesis(a, resynthesis=False, t1=None, t2=None, n_reps=100, seed=0):
    """
    Simulate expected recovery fractions with optional resynthesis of unseen variants.

    Parameters
    ----------
    a : np.ndarray
        Abundance distribution.
    resynthesis : bool, optional
        Whether to perform a second sampling stage that uniformly samples only
        the currently unseen variants. If False, all draws use the original
        abundance distribution.
    t1 : int
        Number of draws in the first stage.
    t2 : int
        Number of draws in the second stage (resynthesis or continued sampling).
    n_reps : int
        Number of replicate simulations to average.
    seed : int
        Random seed.

    Returns
    -------
    xs : np.ndarray
        Sample counts from 1 to the total number of draws.
    ys : np.ndarray
        Mean recovery fractions at each step.
    """
    if t1 is None:
        raise ValueError("t1 must be provided")

    t1 = int(t1)
    if t1 < 0:
        raise ValueError("t1 must be non-negative")

    if t2 is None:
        t2 = 0
    else:
        t2 = int(t2)
        if t2 < 0:
            raise ValueError("t2 must be non-negative")

    if resynthesis and t2 == 0:
        raise ValueError("t2 must be provided when resynthesis is enabled")

    total_draws = t1 + t2
    if total_draws == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    probs = a / a.sum()
    n_variants = len(a)
    curves = np.empty((n_reps, total_draws), dtype=float)

    for rep in range(n_reps):
        seen = np.zeros(n_variants, dtype=bool)
        unique_count = 0
        curve = curves[rep]

        if t1:
            draws1 = rng.choice(n_variants, size=t1, replace=True, p=probs)
            for idx, draw in enumerate(draws1):
                if not seen[draw]:
                    seen[draw] = True
                    unique_count += 1
                curve[idx] = unique_count / n_variants

        if t2:
            if resynthesis:
                unseen = np.flatnonzero(~seen)
                if unseen.size == 0:
                    curve[t1:] = unique_count / n_variants
                    continue
                draws2 = rng.choice(unseen, size=t2, replace=True)
            else:
                draws2 = rng.choice(n_variants, size=t2, replace=True, p=probs)

            for offset, draw in enumerate(draws2, start=t1):
                if not seen[draw]:
                    seen[draw] = True
                    unique_count += 1
                curve[offset] = unique_count / n_variants

    xs = np.arange(1, total_draws + 1)
    ys = curves.mean(axis=0)
    return xs, ys
