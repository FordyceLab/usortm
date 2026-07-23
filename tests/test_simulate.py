"""Tests for the Monte-Carlo coverage simulation.

Guards the regression where a stray `from pysam import samples` at module import
broke `usortm.simulate.sortm`, so `plan` silently fell back to the analytic
coverage approximation instead of running the simulation.
"""
import numpy as np

# Importing here (module scope) is itself part of the test: it must succeed with
# only the core deps (numpy, pandas, numba/scipy), no pysam or tqdm required.
from usortm.simulate.sortm import sortm


def test_sortm_returns_valid_coverage_array():
    result = sortm(n_sims=25, lib_size=50, fold_sampling=8, skew=4, seed=1)
    assert len(result) == 25
    assert result.min() >= 0
    assert result.max() <= 50


def test_higher_fold_sampling_increases_coverage():
    low = sortm(n_sims=50, lib_size=100, fold_sampling=1, skew=4, seed=1)
    high = sortm(n_sims=50, lib_size=100, fold_sampling=12, skew=4, seed=1)
    assert np.mean(high) > np.mean(low)


def test_sortm_is_seed_reproducible():
    a = sortm(n_sims=30, lib_size=80, fold_sampling=6, skew=4, seed=42)
    b = sortm(n_sims=30, lib_size=80, fold_sampling=6, skew=4, seed=42)
    assert np.array_equal(a, b)


def test_simulation_does_not_import_pysam():
    # The stray `from pysam import samples` regression: the simulation module
    # must not import pysam (it is a demux-only, optional dependency).
    import inspect
    import usortm.simulate.sortm as s
    src = inspect.getsource(s)
    assert "pysam" not in src
