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

    for i in range(n_sims):
        pool = generate_pool(lib_size, skew, seed)
        assembled_pool = assemble(pool, p_incorrect)
        clones = transform(assembled_pool, transformation_scale, seed)
        wells = sort(clones, fold_sampling, p_grow, seed)
        barcoded = run_PCR(wells, p_fail, seed)

        if return_correct:
            barcoded = len(barcoded[:-1][barcoded[:-1]>0])
    
        samples[i] = barcoded    

    return samples


def simulate_coverage_curve(
    fold_samplings=np.linspace(1, 5000, 25)/328,
    lib_size=328,
    n_sims=100,
    skew=4,
    transformation_scale=30,
    p_incorrect=0.1,
    p_grow=0.67,
    p_fail=0.03,
    seed=None,
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
    # Instantiate dictionary
    all_samples = {}

    # For each value of # wells sampled
    for fold_sampling in fold_samplings:

        # Sort them
        samples = sortm(
            lib_size=lib_size,
            fold_sampling=fold_sampling,
            skew=skew,
            transformation_scale=transformation_scale,
            p_incorrect=p_incorrect,
            p_grow=p_grow,
            p_fail=p_fail,
            return_correct=True,
            seed=seed,
        )
        
        # Add samples to dictionary
        all_samples[fold_sampling] = samples
        
        # Add zero
        all_samples[0] = np.array([0]*n_sims)

    # Convert to df
    df = pd.DataFrame(all_samples).melt(
        var_name='wells sampled',
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

    df.insert(0, 'fold-sampling', df['wells sampled']/lib_size)
    df.insert(1, 'coverage', df['unique variants']/lib_size)
    df.insert(0, 'transformants', int(lib_size*transformation_scale))

    return df