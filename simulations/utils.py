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


# Model synthetic library distribution
def model_library_distribution():
    # Parameters
    mean_val = 1000        # arbitrary mean center
    fold_change = 4      # 90th / 10th percentile ratio

    # Solve for sigma given 90th/10th ratio
    sigma = np.log(fold_change) / (2 * 1.21816)  # 1.2816 ~ z-score for 90th
    mu = np.log(mean_val) - 0.5 * sigma**2  # shift to match mean

    # Model distribution
    model_dist = lognorm(sigma, scale=np.exp(mu))
    return model_dist