import numpy as np
import numba
from scipy.stats import norm


@numba.njit
def sample(arr, size, seed=None):
    """Fast numba-compatible probability-based sampler.

    Array should be full of some value that maps the index to its abundance
    or probability. The sample `i` would be drawn with a probability of
    arr[i]/sum(arr).

    Parameters:
    -----------
    arr : array-like
        An array of abundances/probabilities.
    size : int
        Number of samples to draw.
    seed : int
        Seed for reproducibility.

    Returns:
    --------
    draws : np.array
        An array of length `size` with values ranging [0, len(arr)).
    """
    # Make the seed not always identical, but reproducible.
    if seed is not None:
        seed = int(np.median(arr)*size)+seed
        np.random.seed(seed)

    # All library members start with zero counts
    draws = [0 for _ in arr]

    # Get probabilities and convert to cumulative sum
    probs = arr / arr.sum()
    cumsum = np.cumsum(probs)
    
    # Draw a random float from [0, 1) and then insert this at idx within
    # the cumulative sum that preseves the order (equivalent to a weighted
    # random sample) 'size' times and keep track of all idx counts
    for _ in range(size):
        idx = np.searchsorted(cumsum, np.random.random(), side='right')
        
        # Add to counter
        draws[idx] += 1

    return np.array(draws)

def generate_pool(lib_size, skew, seed=None):
    """Simulate a log-normal abundance distribution with a specified skew.

    Parameters:
    -----------
    lib_size : int
        Number of unique items (species, variants, etc.) in the pool.
    skew : int or float, [1, inf)
        Fold difference between the 90th and 10th percentiles (Q90/Q10)
        of most/least abundant library members.
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    lib : np.ndarray of shape (lib_size,)
        Simulated probabilities for each item. The final (-1) position
        is "incorrect"/undesired variants.
    """
    # Instantiate rng
    rng = np.random.default_rng(seed)

    # Determine sigma from 90/10 skew.
    z90, z10 = norm.ppf(0.9), norm.ppf(0.1)
    sigma = np.log(skew) / (z90 - z10)

    # Mean is centered around 1
    mu = -0.5 * sigma**2

    # Draw abundances for each variant
    lib = rng.lognormal(mean=mu, sigma=sigma, size=lib_size)

    # Convert to probabilities
    lib = (lib/lib.sum())

    return lib

@numba.njit
def assemble(pool, p_incorrect=0.3):
    """Simulate assembly of the library.
    
    Add a uniform probability of incorrect variants, arising from errors
    in the input library and during preparation and assembly. Incorrect
    variants are stored as the final entry (`new_pool[-1]`).
    
    Parameters:
    -----------
    pool : array-like
        An array containing abundances/probabilities for each library member.

    p_incorrect : float, between [0, 1]
        What fraction of the library is incorrect variants.

    Returns:
    --------
    new_pool : np.array
        Array of shape (pool+1,), containing abundances/probabilities
        for each library member plus the total abundance/probability of
        incorrect members at the final index.
    """
    # Adjust to probs
    total = np.sum(pool)
    probs = np.array(pool/total)

    # Remove these from pool
    new_probs = probs*(1-p_incorrect)

    # Dedicate '-1' index to incorrect things
    new_probs = np.append(new_probs, [p_incorrect])

    # Adjust back
    new_pool = new_probs*total

    return new_pool

@numba.njit
def transform(assembled_pool, scale=30, seed=None):
    """Simulate a transformation yielding len(assembled_pool)*scale total
    colonies.

    Parameters:
    -----------
    assembled_pool : array-like
        An array containing abundances/probabilities for each library member.
        The final index (assembled_pool[-1]) should correspond to incorrect
        variants.

    scale : int or float
        Oversampling of the library. The total number of transformants is
        scale*(len(assembled_pool)-1).

    seed : int or None
        Random seed for reproducibility. 

    Returns:
    --------
    clones : np.array
        Array of shape (assembled_pool,), containing total number of clones
        for each library member plus the total number of cells that
        correspond to incorrect members at the final index.

    """
    size = int((len(assembled_pool)-1)*scale)
    clones = sample(assembled_pool, size, seed=seed)

    return clones

@numba.njit
def sort(clones, fold_sampling=8, p_grow=0.9, seed=None):
    """Sort transformed cells into wells with a probability `p_grow` of
    the cell culturing successfully. The number of wells is equal to the
    library size (len(clones)-1) times fold_sampling.
    
    Not implemented yet: double sort behavior.

    Parameters:
    -----------
    clones : array-like
        An array containing the total number of clones for each library
        member. The final index is the number of clones with incorrect
        variants.
    
    fold_sampling : int or float
        Equal to number of wells sorted divided by the library size
        (len(clones)-1). 1-fold sampling of a 100-member library would be
        100 wells sorted.

    p_grow :  float between [0, 1]
        Probability that a well is successfully grown up to a culture.

    seed : int or None
        Random seed for reproducibility. 

    Returns:
    --------
    wells : np.array
        Array of shape (clones,), containing total number of grown wells for
        each library member plus the total number of wells that correspond
        to incorrect members at the final index.

    """
    # Determine number of grown up wells
    variants = np.sum(clones)-1
    sorted_wells = fold_sampling*variants
    grown_wells = p_grow*sorted_wells
    size = int(grown_wells)

    # Sample
    wells = sample(clones, size, seed=seed)
    # TODO: doubles

    return wells

@numba.njit
def run_PCR(wells, p_fail=0.03, seed=None):
    """Randomly fail `p_fail` percent of PCRs across the wells.
    
    Parameters:
    -----------
    wells : array-like
        An array containing the total number of well for each library
        member. The final index is the number of well with incorrect
        variants.

    p_fail :  float between [0, 1]
        Probability that a well yields a PCR product.

    seed : int or None
        Random seed for reproducibility. 

    Returns:
    --------
    barcoded : np.array
        Array of shape (wells,), containing total number of barcoded
        wells that will be sequenced for each library member, plus the
        total number of wells that correspond to incorrect members at 
        the final index.
    """
    # How many wells fail
    size = int(np.sum(wells)*p_fail)

    # Assign failed wells to variants
    fails = sample(wells, size=size, seed=seed)

    # Subtract failures from total
    barcoded = wells - fails

    return barcoded