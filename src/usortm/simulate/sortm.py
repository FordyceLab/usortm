import numpy as np
import pandas as pd

from .sample import (
    generate_pool,
    assemble,
    transform,
    sort,
    run_PCR,
)

def sortm(
    n_sims=10000,
    lib_size=1000,
    skew=4,
    p_incorrect=0.3,
    transformation_scale=50,
    fold_sampling=10,
    p_grow=0.9,
    p_fail=0.03,
    return_correct=True,
    seed=None,
):
    """Sort them!

    Runs all steps of sampling during a usort-m run.
    See the sample module for more details.

    Parameters:
    -----------
    n_sims : int
        Number of simulations to perform with the selected parameters.
    lib_size : int
        Number of unique items (species, variants, etc.) in the pool.
    skew : int or float, [1, inf)
        Fold difference between the 90th and 10th percentiles (Q90/Q10)
        of most/least abundant library members.
    p_incorrect : float, between [0, 1]
        What fraction of the input library is incorrect variants.
    transformation_scale : int or float
        Oversampling of the library. The total number of transformants is
        scale*(lib_size).
    fold_sampling : int or float
        Equal to number of wells sorted divided by the library size
        (lib_size). 1-fold sampling of a 100-member library would be
        100 wells sorted.
    p_grow :  float between [0, 1]
        Probability that a well is successfully grown up to a culture.
    p_fail :  float between [0, 1]
        Probability that a well yields a PCR product.
    return_correct : bool, default True
        Whether to return only the correct samples or also include the
        final index of inccorect variant abundance.
    seed : int or None
        Random seed for reproducibility. 

    Returns:
    --------
    samples : np.array
        Array of length `lib_size` if return_correct is True, or `lib_size`+1
        if return_correct is False, with the final (-1) index being the
        abundance of incorrect variants. Each index corresponds to a specific
        library member and the value corresponds to its abundance, or the
        number of sequenced wells containing that variant.
    
    """
    samples = np.arange(n_sims)

    # Initialize list of seeds for all simulations
    if seed is not None:
        seeds = np.arange(seed, seed + n_sims)

    for i in range(n_sims):
        if seed is not None:
            seed = seeds[i]

        pool = generate_pool(lib_size, skew, seed)
        assembled_pool = assemble(pool, p_incorrect)
        clones = transform(assembled_pool, transformation_scale, seed)
        wells = sort(clones, fold_sampling, p_grow, seed)
        barcoded = run_PCR(wells, p_fail, seed)

        if return_correct:
            barcoded = len(barcoded[:-1][barcoded[:-1]>0])
    
        samples[i] = barcoded    

    return samples


def find_fold_sampling(
    target_coverage=0.90,
    lib_size=1000,
    skew=4,
    p_grow=0.67,
    p_fail=0.03,
    p_incorrect=0.3,
    transformation_scale=50,
    n_sims=100,
    seed=42,
    tol=0.01,
    max_iter=15,
    progress_callback=None,
):
    """Find the minimum fold-sampling to achieve a target coverage.

    Uses binary search over fold-sampling values, running the sortm
    simulation at each candidate to evaluate expected coverage.

    Parameters
    ----------
    target_coverage : float
        Target fraction of library recovered (e.g. 0.90 for 90%).
    lib_size : int
        Number of unique variants in the library.
    skew : float
        Library skew (Q90/Q10 abundance ratio).
    p_grow : float
        Sorting efficiency (fraction of wells that grow).
    p_fail : float
        PCR failure rate.
    p_incorrect : float
        Fraction of incorrect variants in the pool.
    transformation_scale : int or float
        Transformant oversampling factor.
    n_sims : int
        Number of simulations per evaluation.
    seed : int or None
        Random seed for reproducibility.
    tol : float
        Acceptable deviation from target coverage.
    max_iter : int
        Maximum binary search iterations.
    progress_callback : callable or None
        Called with (iteration, fold_sampling, coverage) after each evaluation.

    Returns
    -------
    fold_sampling : float
        Minimum fold-sampling (rounded up to 0.5) to achieve target coverage.
    coverage : float
        Expected coverage at the returned fold-sampling.
    """
    low, high = 1.0, 20.0

    # Check if high bound is sufficient
    result = sortm(
        n_sims=n_sims, lib_size=lib_size, fold_sampling=high,
        skew=skew, p_grow=p_grow, p_fail=p_fail,
        p_incorrect=p_incorrect, transformation_scale=transformation_scale,
        seed=seed,
    )
    high_cov = np.mean(result) / lib_size
    iteration = 1
    if progress_callback:
        progress_callback(iteration, high, high_cov)
    while high_cov < target_coverage and high < 100:
        high *= 2
        iteration += 1
        result = sortm(
            n_sims=n_sims, lib_size=lib_size, fold_sampling=high,
            skew=skew, p_grow=p_grow, p_fail=p_fail,
            p_incorrect=p_incorrect, transformation_scale=transformation_scale,
            seed=seed,
        )
        high_cov = np.mean(result) / lib_size
        if progress_callback:
            progress_callback(iteration, high, high_cov)

    best_fold, best_cov = high, high_cov

    for _ in range(max_iter):
        mid = (low + high) / 2
        iteration += 1
        result = sortm(
            n_sims=n_sims, lib_size=lib_size, fold_sampling=mid,
            skew=skew, p_grow=p_grow, p_fail=p_fail,
            p_incorrect=p_incorrect, transformation_scale=transformation_scale,
            seed=seed,
        )
        mid_cov = np.mean(result) / lib_size

        if progress_callback:
            progress_callback(iteration, mid, mid_cov)

        if mid_cov >= target_coverage:
            best_fold, best_cov = mid, mid_cov
            high = mid
        else:
            low = mid

        if abs(mid_cov - target_coverage) < tol:
            break

    # Round up to nearest 0.5 for practical use
    import math
    best_fold = math.ceil(best_fold * 2) / 2

    # Re-evaluate at the rounded value
    result = sortm(
        n_sims=n_sims, lib_size=lib_size, fold_sampling=best_fold,
        skew=skew, p_grow=p_grow, p_fail=p_fail,
        p_incorrect=p_incorrect, transformation_scale=transformation_scale,
        seed=seed,
    )
    best_cov = np.mean(result) / lib_size

    return best_fold, best_cov


def simulate_resynthesis_strategy(
    target_coverage=0.90,
    lib_size=1000,
    skew=4,
    round1_fold=3.0,
    p_grow=0.67,
    p_fail=0.03,
    p_incorrect=0.3,
    transformation_scale=50,
    n_sims=100,
    seed=42,
    progress_callback=None,
):
    """Simulate a two-round resynthesis strategy.

    Round 1 sorts the original (skewed) library at round1_fold. Unrecovered
    variants are resynthesized as a new pool (skew=1) and sorted in round 2
    to hit target_coverage.

    Parameters
    ----------
    target_coverage : float
        Target overall coverage of the original library.
    lib_size : int
        Number of unique variants in the original library.
    skew : float
        Library skew (Q90/Q10) for round 1.
    round1_fold : float
        Fold-sampling for round 1.
    p_grow, p_fail, p_incorrect, transformation_scale, n_sims, seed :
        See sortm() for descriptions.
    progress_callback : callable or None
        Called with (step_name, detail_string) for progress updates.

    Returns
    -------
    dict with keys:
        round1_fold, round1_coverage, round1_recovered, round1_wells,
        dropout_count, round2_fold, round2_coverage, round2_wells,
        total_wells, total_coverage
    """
    sim_kwargs = dict(
        p_grow=p_grow, p_fail=p_fail, p_incorrect=p_incorrect,
        transformation_scale=transformation_scale, n_sims=n_sims, seed=seed,
    )

    # Round 1: sort original skewed library
    if progress_callback:
        progress_callback("round1", f"Simulating round 1 at {round1_fold}×...")
    r1 = sortm(lib_size=lib_size, fold_sampling=round1_fold, skew=skew,
               return_correct=True, **sim_kwargs)
    r1_recovered = int(np.mean(r1))
    r1_coverage = r1_recovered / lib_size
    r1_wells = int(lib_size * round1_fold)

    dropout_count = lib_size - r1_recovered

    if dropout_count <= 0:
        # Round 1 already hit full coverage
        return {
            "round1_fold": round1_fold,
            "round1_coverage": r1_coverage,
            "round1_recovered": r1_recovered,
            "round1_wells": r1_wells,
            "dropout_count": 0,
            "round2_fold": 0,
            "round2_coverage": 1.0,
            "round2_wells": 0,
            "total_wells": r1_wells,
            "total_coverage": r1_coverage,
        }

    # Determine what coverage we need in round 2 to hit overall target
    # We need: r1_recovered + r2_recovered >= target_coverage * lib_size
    needed_from_r2 = max(1, int(target_coverage * lib_size) - r1_recovered)
    r2_target = min(0.99, needed_from_r2 / dropout_count)

    # Round 2: sort dropout library with skew=1 (uniform resynthesis)
    if progress_callback:
        progress_callback("round2", f"Simulating round 2 for {dropout_count} dropouts...")

    def _r2_progress(iteration, fs, cov):
        if progress_callback:
            progress_callback("round2", f"Round 2: {fs:.1f}× → {cov:.1%} of dropouts")

    r2_fold, r2_cov = find_fold_sampling(
        target_coverage=r2_target,
        lib_size=dropout_count,
        skew=1,  # uniform — resynthesized individually
        progress_callback=_r2_progress,
        **sim_kwargs,
    )
    r2_wells = int(dropout_count * r2_fold)
    r2_recovered = int(r2_cov * dropout_count)

    total_coverage = (r1_recovered + r2_recovered) / lib_size

    return {
        "round1_fold": round1_fold,
        "round1_coverage": r1_coverage,
        "round1_recovered": r1_recovered,
        "round1_wells": r1_wells,
        "dropout_count": dropout_count,
        "round2_fold": r2_fold,
        "round2_coverage": r2_cov,
        "round2_recovered": r2_recovered,
        "round2_wells": r2_wells,
        "total_wells": r1_wells + r2_wells,
        "total_coverage": total_coverage,
    }


def simulate_coverage_curve(
    fold_samplings=np.linspace(1, 10, 20),
    lib_size=328,
    n_sims=100,
    skew=4,
    transformation_scale=30,
    p_incorrect=0.1,
    p_grow=0.67,
    p_fail=0.03,
    seed=None,
    pbar=True,
):
    """Run the sortm function for many values of sorted wells and for many
    simulations per each value.

    Parameters:
    -----------
    fold_samplings : array-like
        An array of different fold_sampling values to sample.

    See `sortm` function for other parameter descriptions.

    Returns:
    --------
    df : pd.DataFrame
        A DataFrame containing all sampling information.
    """
    from tqdm import tqdm  # lazy: only the coverage-curve helpers need it

    # Instantiate dictionary
    all_samples = {}

    # For each value of # wells sampled
    for fold_sampling in tqdm(fold_samplings, disable=not pbar):

        # Sort them
        samples = sortm(
            lib_size=lib_size,
            fold_sampling=fold_sampling,
            skew=skew,
            n_sims=n_sims,
            transformation_scale=transformation_scale,
            p_incorrect=p_incorrect,
            p_grow=p_grow,
            p_fail=p_fail,
            return_correct=True,
            seed=seed,
        )
        
        # Add samples to dictionary
        all_samples[float(fold_sampling)] = samples
        
        # Add zero
        all_samples[0] = np.array([0]*n_sims)

    # Convert to df
    df = pd.DataFrame(all_samples).melt(
        var_name='fold-sampling',
        value_name='unique variants'
    )

    # Add all data
    df['library size'] = lib_size
    df['library skew'] = skew
    df['transformation scale'] = transformation_scale
    df['fraction library incorrect'] = p_incorrect
    df['sorting efficiency'] = p_grow
    df['PCR failure rate'] = p_fail
    df = df.set_index([
        'library size',
        'library skew',
        'transformation scale',
        'fraction library incorrect',
        'sorting efficiency',
        'PCR failure rate',
    ])

    df.insert(0, 'wells sampled', df['fold-sampling']*lib_size)
    df.insert(1, 'coverage', df['unique variants']/lib_size)
    df.insert(0, 'transformants', int(lib_size*transformation_scale))

    return df

def simulate_coverage_curve_with_resampling(
    n_wells=np.linspace(1, 3000, 20),
    lib_size=328,
    n_sims=100,
    skew=4,
    transformation_scale=30,
    p_incorrect=0.1,
    p_grow=0.67,
    p_fail=0.03,
    seed=None,
    pbar=True,
    resampling_well_count=None
):
    """
    Simulate coverage curves with and without resampling from the remaining library.
    Adds a 'Resampled' column for comparison.
    """
    from tqdm import tqdm  # lazy: only the coverage-curve helpers need it

    rng = np.random.default_rng(seed)
    starting_lib_size = lib_size
    base_args = dict(
        skew=skew,
        transformation_scale=transformation_scale,
        p_incorrect=p_incorrect,
        p_grow=p_grow,
        p_fail=p_fail,
        n_sims=n_sims,
        return_correct=True,
        seed=seed,
    )

    def simulate_curve(resampled=False):
        all_samples = {}
        total_recovered, curr_lib_size = 0, starting_lib_size
        samples_prev = np.zeros(n_sims)

        for wells in tqdm(n_wells, disable=not pbar):
            # Resynthesize at the resampling point
            if resampled and resampling_well_count and wells == resampling_well_count:
                recovered = int(np.mean(samples_prev))
                total_recovered += recovered
                curr_lib_size = starting_lib_size - recovered
                samples_prev = np.zeros(n_sims)

            # Adjust wells if past resampling point
            curr_wells = wells - resampling_well_count if (
                resampled and resampling_well_count and wells >= resampling_well_count
            ) else wells

            samples = sortm(lib_size=curr_lib_size,
                            fold_sampling=curr_wells / curr_lib_size,
                            **base_args)
            samples = np.array(samples) + total_recovered
            all_samples[wells] = samples
            samples_prev = samples

        df = pd.DataFrame(all_samples).melt(var_name='wells sampled', value_name='unique variants')
        df['coverage'] = df['unique variants'] / starting_lib_size
        df['transformants'] = int(starting_lib_size * transformation_scale)
        df['Resampled'] = resampled
        return df

    # Combine both curves
    df_noresamp = simulate_curve(resampled=False)
    df_resamp = simulate_curve(resampled=True) if resampling_well_count else pd.DataFrame()
    df_all = pd.concat([df_noresamp, df_resamp], ignore_index=True)
    return df_all