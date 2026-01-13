"""Not finalized."""


import numpy as np


# def model_distribution(N, skew=5, seed=0):
#     """
#     Simulate a log-normal abundance distribution with a specified skew.

#     Parameters
#     ----------
#     N : int
#         Number of unique items (species, variants, etc.) in the pool.
#     skew : float
#         Fold difference between the 90th and 10th percentiles (Q90/Q10).
#     seed : int, optional
#         Random seed for reproducibility.

#     Returns
#     -------
#     a : np.ndarray of shape (N,)
#         Simulated abundance values for each item. Mean abundance ≈ 1.
#     """
#     rng = np.random.default_rng(seed)

#     # infer sigma from skew = Q90/Q10
#     z90, z10 = norm.ppf(0.9), norm.ppf(0.1)
#     sigma = np.log(skew) / (z90 - z10)
#     mu = -0.5 * sigma**2   # ensures mean ~1

#     a = rng.lognormal(mean=mu, sigma=sigma, size=N)
#     return a

def gen_pool(n, skew, n_mols=1e8, seed=None):
    """Simulate a log-normal abundance distribution with a specified skew.

    Parameters
    ----------
    n : int
        Number of unique items (species, variants, etc.) in the pool.
    skew : float
        Fold difference between the 90th and 10th percentiles (Q90/Q10).
    n_mols : float
        Number of molecules. Defaults to 1e8 (0.2 fmol).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    a : np.ndarray of shape (n,)
        Simulated abundance values for each item, normalized to number of molecules.
    """
    rng = np.random.default_rng(seed)

    # infer sigma from skew = Q90/Q10
    z90, z10 = norm.ppf(0.9), norm.ppf(0.1)
    sigma = np.log(skew) / (z90 - z10)
    mu = -0.5 * sigma**2   # ensures mean ~1

    a = rng.lognormal(mean=mu, sigma=sigma, size=n)
    a = (a/a.sum())*n_mols
    a = a.astype(int)

    return pd.Series(a)

def assemble(pool, frac_pool=0.1, p_incorrect=0.1):
    """Simulate the assembly of a library with a uniform probability of
    failure to assemble correctly, creating new incorrect samples."""
    # Use a portion of pool
    used_pool = pool*frac_pool

    # Make some incorrect samples
    n_incorrect = int(sum(used_pool)*p_incorrect)

    # Remove these from pool
    new_pool = np.floor(used_pool*(1-p_incorrect))

    # Dedicate '-1' index to incorrect things
    new_pool[-1] = n_incorrect

    return new_pool.astype(int).sort_index()

def transform(assembled_pool, n_transformed=10000, seed=None):
    """Simulate a transformation yielding `n_transformed` cells."""
    rng = np.random.default_rng(seed)
    
    probs = assembled_pool / assembled_pool.sum()
    draws = rng.choice(
        len(assembled_pool),
        size=n_transformed,
        replace=True,
        p=probs,
    )

    # Get counts of each clone
    clones = pd.Series(draws).value_counts()
    # '0' was actually the -1 value; correct
    clones.index = clones.index-1

    # Set any missing to 0
    clones = clones + pd.Series(0, index=range(-1, len(assembled_pool)-1))
    clones = clones.fillna(0).astype(int)

    return clones

def sort(clones, n_wells=3072, p_clone=0.9, seed=None):
    """Sort transformed cells into `n_wells` wells with a probability
    `p_clone` of the cell culturing successfully.
    
    NEED TO IMPLEMENT DOUBLE-MUTANT BEHAVIOR
    """
    rng = np.random.default_rng(seed)
    
    probs = clones / clones.sum()
    draws = rng.choice(
        len(clones),
        size=int(n_wells*p_clone),
        replace=True,
        p=probs,
    )

    # TODO: Set doubles?

    # Get value counts
    wells = pd.Series(draws).value_counts()
    # '0' was actually the -1 value; correct
    wells.index = wells.index-1

    wells = wells + pd.Series(0, index=range(-1, len(clones)-1))
    wells = wells.fillna(0).astype(int)

    return wells

def PCR(wells, p_fail=0.03, seed=None):
    """Randomly fail `p_fail` percent of PCRs across the wells."""
    rng = np.random.default_rng(seed)
    
    # Flatten the array, then sample directly, then reconvert
    flattened = [[i]*wells[i] for i in wells.index]
    flattened = [item for sublist in flattened for item in sublist]

    draws = rng.choice(
        flattened,
        size=int(sum(wells)*(1-p_fail)),
        replace=False,
    )

    # Get value counts
    counts = pd.Series(draws).value_counts()
    counts = counts + pd.Series(0, index=range(-1, len(wells)-1))
    counts = counts.fillna(0).astype(int)

    return counts

def sortm_pipe(
    lib_size,
    n_wells=3072,
    skew=4,
    n_transformed=10000,
    p_incorrect=0.1,
    p_clone=0.9,
    p_fail=0.03,
    n_mols=1e8,
    seed=None,
):
    """Sort them!"""
    pool = gen_pool(lib_size, skew, n_mols, seed)
    assembled_pool = assemble(pool, p_incorrect)
    clones = transform(assembled_pool, n_transformed, seed)
    good_wells = sort(clones, n_wells, p_clone, seed)
    sequenced_wells = PCR(good_wells, p_fail, seed)
    #TODO: sequencing depth

    return sequenced_wells

def simulate_coverage_per_well(
    lib_size=328,
    n_sims=100,
    n_wells=np.linspace(1, 5000, 25),
    skew=4,
    n_transformed=5250,
    p_incorrect=0.1,
    p_clone=0.67,
    p_fail=0.03,
    n_mols=1e8,
    seed=None,
):
    """Run the sortm function for many values of sorted wells and for many
    simulations per each value."""
    # Instantiate dictionary
    all_samples = {}

    # For each value of # wells sampled
    for n_wells_sampled in n_wells:

        # Instantiate sample container
        samples = np.array([_ for _ in range(n_sims)])

        # For each simulation
        for i in range(n_sims):

            # Sort them
            wells = sortm_pipe(
                lib_size,
                n_wells_sampled,
                skew,
                n_transformed,
                p_incorrect,
                p_clone,
                p_fail,
                n_mols,
                seed,
            )

            # Store number of non-zero correct variants
            n = len(wells[1:][wells[1:]>0])
            samples[i] = n
        
        # Add samples to dictionary
        all_samples[n_wells_sampled] = samples
        
        # Add zero
        all_samples[0] = np.array([0]*n_sims)

    # Convert to df
    df = pd.DataFrame(all_samples).melt(
        var_name='wells sampled',
        value_name='unique variants'
    )

    # Add metadata
    df['library size'] = lib_size
    df['library skew'] = skew
    df['transformants'] = n_transformed
    df['desired mut. freq.'] = 1-p_incorrect
    df['sorting efficiency'] = p_clone
    df['PCR efficiency'] = 1-p_fail
    df = df.set_index([
        'library size',
        'library skew',
        'transformants',
        'desired mut. freq.',
        'sorting efficiency',
        'PCR efficiency',
    ])

    return df

def run_simulation(combos):
    lib_size, skew, n_transform, mut_freq, sort_eff, PCR_fail = combos
    df = simulate_coverage_per_well(
            lib_size=lib_size,
            n_sims=10,
            n_wells=np.linspace(1, 5000, 25),
            skew=skew,
            n_transformed=n_transform,
            p_incorrect=mut_freq,
            p_clone=sort_eff,
            p_fail=PCR_fail,
            n_mols=1e8,
            seed=None,
        )
    return df

# def take_sample(a, t, recovery=False, seed=None):
#     """
#     Perform an explicit random sampling experiment.

#     Parameters
#     ----------
#     a : np.ndarray
#         Abundance distribution (length N).
#     t : int
#         Number of draws with replacement.
#     seed : int, optional
#         Random seed for reproducibility.

#     Returns
#     -------
#     frac_recovered : float
#         Fraction of unique items recovered in this sample.
#     """
#     rng = np.random.default_rng(seed)
#     probs = a / a.sum()
#     draws = rng.choice(len(a), size=t, replace=True, p=probs)
#     unique = len(np.unique(draws))
#     if recovery:
#         return draws, unique / len(a)
#     else:
#         return draws

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
