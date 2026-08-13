"""Tests for the Monte-Carlo coverage simulation.

Guards the regression where a stray `from pysam import samples` at module import
broke `usortm.simulate.sortm`, so `plan` silently fell back to the analytic
coverage approximation instead of running the simulation.
"""
import numpy as np

# Importing here (module scope) is itself part of the test: it must succeed with
# only the core deps (numpy, pandas, numba/scipy), no pysam or tqdm required.
from usortm.simulate.sortm import expected_coverage, sortm


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


def test_expected_coverage_matches_the_sortm_mean():
    # p_grow is passed explicitly: expected_coverage follows find_fold_sampling's
    # 0.67 sorting efficiency, while sortm itself defaults to 0.9.
    kwargs = dict(lib_size=80, fold_sampling=4, skew=2, p_grow=0.67,
                  n_sims=30, seed=7)
    prediction = expected_coverage(**kwargs)
    counts = sortm(**kwargs)

    assert prediction["coverage"] == np.mean(counts) / 80
    assert prediction["recovered"] == np.mean(counts)
    assert prediction["wells"] == 320
    assert 0 <= prediction["coverage"] <= 1


def test_expected_coverage_brackets_the_mean_with_percentiles():
    prediction = expected_coverage(lib_size=80, fold_sampling=4, skew=2,
                                   n_sims=30, seed=7)
    assert prediction["coverage_p10"] <= prediction["coverage"] <= prediction["coverage_p90"]
    assert prediction["coverage_sd"] >= 0


def test_expected_coverage_rises_with_fold_sampling():
    low = expected_coverage(lib_size=100, fold_sampling=2, skew=4, n_sims=30, seed=3)
    high = expected_coverage(lib_size=100, fold_sampling=10, skew=4, n_sims=30, seed=3)
    assert high["coverage"] > low["coverage"]


def test_expected_coverage_falls_with_skew():
    flat = expected_coverage(lib_size=100, fold_sampling=4, skew=1.5, n_sims=30, seed=3)
    skewed = expected_coverage(lib_size=100, fold_sampling=4, skew=20, n_sims=30, seed=3)
    assert skewed["coverage"] < flat["coverage"]


def test_expected_coverage_is_seed_reproducible():
    kwargs = dict(lib_size=80, fold_sampling=5, skew=3, n_sims=20, seed=11)
    assert expected_coverage(**kwargs) == expected_coverage(**kwargs)


def test_simulation_does_not_import_pysam():
    # The stray `from pysam import samples` regression: the simulation module
    # must not import pysam (it is a demux-only, optional dependency).
    import inspect
    import usortm.simulate.sortm as s
    src = inspect.getsource(s)
    assert "pysam" not in src
