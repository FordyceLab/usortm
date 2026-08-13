"""Library skew measurement and sorting-depth recommendation.

Given per-variant read counts from a shallow sequencing run of the
amplified library (e.g. Plasmidsaurus premium PCR, 12-20k reads), estimate
how unevenly the library is distributed and how deeply it must be sorted to
recover a target fraction of it.

The central difficulty is that at 12-20k reads over a few hundred to a few
thousand variants, each variant is seen only ~8-30 times.  Poisson counting
noise alone makes a *perfectly uniform* library look skewed, so the raw
Q90/Q10 ratio of read counts overstates the true skew and would lead to
over-sorting.

This module fits a zero-inflated Poisson-log-normal model::

    absent_i ~ Bernoulli(delta)                    # synthesis dropout
    lambda_i ~ LogNormal(mu, sigma^2)              # true abundance
    c_i      ~ Poisson(lambda_i)                   # observed reads

and reports the deconvolved sigma, from which skew is recovered as
``exp(sigma * (z90 - z10))`` — the exact inverse of the parameterization
:func:`usortm.simulate.sample.generate_pool` uses, so a simulated pool
round-trips through the estimator.

Separating ``delta`` from ``sigma`` matters: synthesis dropouts inflate the
variance of the count vector, and a model without a dropout term would
attribute that variance to skew and recommend deeper sorting to chase
variants that are not in the tube at all.

Validated range
---------------
Against synthetic libraries of known skew, the corrected estimate is
unbiased to within a few percent up to about 8x Q90/Q10, across library
sizes of 300-2000 and depths of 7-50 reads per variant.  Above roughly 10x
it reads *low* — around 0.85x of truth at 16x — because so much of the
library falls below one expected read that the likelihood has little
information about how deep the tail goes, and the interval under-covers.
A low estimate under-recommends sorting depth, so
:attr:`SkewStats.beyond_validated_range` flags it and the estimate should be
treated as a lower bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp, roots_hermite
from scipy.stats import norm

# Q90/Q10 of a log-normal is exp(sigma * (z90 - z10)); generate_pool()
# inverts this to pick sigma from a requested skew.
_Z_SPREAD = norm.ppf(0.9) - norm.ppf(0.1)

# Gauss-Hermite nodes for integrating the log-normal mixing distribution.
# 40 nodes is far more than needed for the smooth integrands here.
_GH_NODES = 40

# Below this many reads per variant, counting noise dominates and the
# deconvolution has little signal to work with.
_MIN_DEPTH_FOR_FIT = 10.0

# Search range for the log-normal sigma, spanning uniform to extremely
# skewed libraries (sigma = 3 is a Q90/Q10 ratio of ~2000).
_SIGMA_BOUNDS = (1e-3, 3.0)

# Above this Q90/Q10 the fit is biased low (see "Validated range" above), so
# the result is a lower bound rather than an estimate.
_VALIDATED_SKEW_MAX = 10.0

__all__ = [
    "VariantCounts",
    "SkewStats",
    "SamplingRecommendation",
    "ci_to_json",
    "sigma_to_skew",
    "skew_to_sigma",
    "measure_skew",
    "recommend_sampling",
    "predicted_count_distribution",
    "underlying_log10_density",
    "log10_histogram",
]


def ci_to_json(ci) -> Optional[list]:
    """Render a confidence interval for JSON.

    The interval is NaN when the likelihood fit did not converge; JSON has
    no NaN literal, so that becomes null rather than a token strict
    parsers reject.
    """
    if any(not np.isfinite(v) for v in ci):
        return None
    return [round(float(v), 3) for v in ci]


def sigma_to_skew(sigma: float) -> float:
    """Convert log-normal sigma to a Q90/Q10 abundance ratio."""
    return float(np.exp(sigma * _Z_SPREAD))


def skew_to_sigma(skew: float) -> float:
    """Convert a Q90/Q10 abundance ratio to log-normal sigma."""
    return float(np.log(skew) / _Z_SPREAD)


@dataclass
class VariantCounts:
    """Per-variant read counts from a library sequencing run.

    Attributes:
        counts: Variant name -> uniquely assigned reads. Contains an entry
            for every variant in the input library, including zeros.
        ambiguous: Reads whose best alignment did not beat the runner-up
            on a different reference by the required margin.
        unmapped: Reads with no alignment to any library member.
        low_cov: Reads whose alignment spanned too little of the reference.
        total_reads: Reads in the input FASTQ.
        duplicate_groups: Groups of variant names sharing an identical
            sequence; reads cannot be attributed among them.
    """

    counts: dict
    ambiguous: int = 0
    unmapped: int = 0
    low_cov: int = 0
    total_reads: int = 0
    duplicate_groups: list = field(default_factory=list)

    @property
    def library_size(self) -> int:
        return len(self.counts)

    @property
    def assigned_reads(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def names(self) -> list:
        return list(self.counts.keys())

    def as_array(self) -> np.ndarray:
        """Counts as a float array, ordered as `names`."""
        return np.asarray(list(self.counts.values()), dtype=np.float64)


@dataclass
class SkewStats:
    """Abundance statistics for a measured library.

    Attributes:
        q90_q10_observed: Raw Q90/Q10 of the read counts. Inflated by
            counting noise; None when the 10th percentile is zero.
        q90_q10_corrected: Q90/Q10 after deconvolving Poisson noise. This
            is the number to feed downstream planning.
        q90_q10_ci: 95% profile-likelihood interval on the corrected
            skew. Wide intervals are common for small libraries and mean
            the recommendation should be treated as a lower bound.
        sigma_log: Fitted log-normal sigma — the width of the abundance
            distribution in natural-log units. A tight distribution is a
            small sigma.
        mu_log: Fitted log-normal mu, on the scale of expected reads per
            present variant. Together with sigma_log this is the fitted
            Gaussian in log space.
        dropout_fraction: Estimated fraction of the library absent from
            the tube, distinct from variants merely missed by sequencing.
        gini: Gini coefficient of the shrunk abundance estimates.
        effective_library_size: Inverse Simpson index — the number of
            equally abundant variants that would sample as unevenly as
            this library does.
        n_detected: Variants with at least one read.
        n_undetected: Variants with zero reads. Not the same as absent:
            at low depth a present variant is often missed by chance.
        undetected_names: Names of the zero-count variants.
        median_depth: Median reads per detected variant.
        mean_depth: Assigned reads divided by library size.
        depth_sufficient: Whether depth supports a reliable deconvolution.
        shrunk_abundance: Empirical-Bayes posterior mean relative
            abundance per variant, ordered as `VariantCounts.names`.
        fit_method: "mle" or "moments".
        fit_converged: Whether the optimizer reported success.
    """

    q90_q10_observed: Optional[float]
    q90_q10_corrected: float
    q90_q10_ci: tuple
    sigma_log: float
    mu_log: float
    dropout_fraction: float
    gini: float
    effective_library_size: float
    n_detected: int
    n_undetected: int
    undetected_names: list
    median_depth: float
    mean_depth: float
    depth_sufficient: bool
    shrunk_abundance: np.ndarray
    fit_method: str
    fit_converged: bool

    @property
    def coverage_ceiling(self) -> float:
        """Largest library fraction any sorting depth could recover."""
        return 1.0 - self.dropout_fraction

    @property
    def beyond_validated_range(self) -> bool:
        """Whether the fit is in the regime where it reads low.

        True means treat the skew as a lower bound and the recommended
        sorting depth as a floor. See "Validated range" in the module
        docstring.
        """
        return self.q90_q10_corrected > _VALIDATED_SKEW_MAX

    def to_dict(self) -> dict:
        """JSON-serializable summary, omitting the per-variant array."""
        return {
            "q90_q10_observed": self.q90_q10_observed,
            "q90_q10_corrected": round(self.q90_q10_corrected, 3),
            "q90_q10_ci": ci_to_json(self.q90_q10_ci),
            "sigma_log": round(self.sigma_log, 4),
            "sigma_log10": round(self.sigma_log / math.log(10), 4),
            "dropout_fraction": round(self.dropout_fraction, 4),
            "gini": round(self.gini, 4),
            "effective_library_size": round(self.effective_library_size, 1),
            "n_detected": self.n_detected,
            "n_undetected": self.n_undetected,
            "median_depth": round(self.median_depth, 1),
            "mean_depth": round(self.mean_depth, 2),
            "depth_sufficient": self.depth_sufficient,
            "beyond_validated_range": self.beyond_validated_range,
            "coverage_ceiling": round(self.coverage_ceiling, 4),
            "fit_method": self.fit_method,
            "fit_converged": self.fit_converged,
        }


@dataclass
class SamplingRecommendation:
    """Recommended sorting depth for a measured library.

    Attributes:
        fold_sampling: Wells to sort per library member.
        n_wells: Total wells to sort.
        n_plates: 384-well plates that implies.
        expected_coverage: Predicted fraction of the *full* library
            recovered at this depth.
        expected_coverage_of_present: Predicted fraction of the variants
            actually present in the tube.
        target_coverage: Coverage that was requested.
        coverage_ceiling: Ceiling imposed by synthesis dropouts.
        target_reachable: False when the target exceeds the ceiling.
        basis: "empirical" (measured abundances) or "lognormal" (fitted).
    """

    fold_sampling: float
    n_wells: int
    n_plates: int
    expected_coverage: float
    expected_coverage_of_present: float
    target_coverage: float
    coverage_ceiling: float
    target_reachable: bool
    basis: str

    def to_dict(self) -> dict:
        return {
            "fold_sampling": self.fold_sampling,
            "n_wells": self.n_wells,
            "n_plates": self.n_plates,
            "expected_coverage": round(self.expected_coverage, 4),
            "expected_coverage_of_present": round(self.expected_coverage_of_present, 4),
            "target_coverage": self.target_coverage,
            "coverage_ceiling": round(self.coverage_ceiling, 4),
            "target_reachable": self.target_reachable,
            "basis": self.basis,
        }


# ---------------------------------------------------------------------------
# Poisson-log-normal machinery
# ---------------------------------------------------------------------------

def _pln_log_pmf(k_values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """log P(c = k) under a Poisson-log-normal, for an array of k.

    Integrates Poisson(k; e^z) against Normal(z; mu, sigma^2) by
    Gauss-Hermite quadrature, in log space for stability.
    """
    nodes, weights = roots_hermite(_GH_NODES)
    # Substitution z = mu + sqrt(2) * sigma * t maps the Gauss-Hermite
    # weight function to the normal density (up to the 1/sqrt(pi) factor).
    z = mu + np.sqrt(2.0) * sigma * nodes            # (Q,)
    log_w = np.log(weights) - 0.5 * np.log(np.pi)    # (Q,)

    k = np.asarray(k_values, dtype=np.float64)[:, None]   # (K, 1)
    lam = np.exp(z)[None, :]                              # (1, Q)

    # log Poisson pmf: k*z - e^z - log(k!)
    log_pois = k * z[None, :] - lam - gammaln(k + 1.0)
    return logsumexp(log_pois + log_w[None, :], axis=1)


def _moment_sigma(counts: np.ndarray) -> float:
    """Method-of-moments sigma, subtracting the Poisson variance.

    For c ~ Poisson(lambda) with lambda log-normal:
        Var[c] = E[c] + E[c]^2 * (exp(sigma^2) - 1)
    so sigma^2 = ln(1 + (Var[c] - E[c]) / E[c]^2).  Under-dispersed data
    (no skew resolvable above counting noise) yields sigma = 0.
    """
    mean = counts.mean()
    if mean <= 0:
        return 0.0
    excess = counts.var() - mean
    if excess <= 0:
        return 0.0
    return float(np.sqrt(np.log1p(excess / mean**2)))


def _make_neg_log_lik(counts: np.ndarray):
    """Build the zero-inflated Poisson-log-normal negative log-likelihood.

    Reads are shared only among present variants, so the log-normal is
    centred on ``N / (L * (1 - delta))`` rather than ``N / L``.

    Returns (neg_log_lik(sigma, delta), delta_hi, delta0).
    """
    n_reads = counts.sum()
    lib_size = len(counts)

    # Collapse to a count histogram: the likelihood only depends on how
    # many variants had each count, and unique counts are few.
    uniq, multiplicity = np.unique(counts.astype(np.int64), return_counts=True)
    uniq_f = uniq.astype(np.float64)
    has_zero = bool((uniq == 0).any())
    n_zero = int(multiplicity[uniq == 0].sum()) if has_zero else 0
    zero_idx = int(np.flatnonzero(uniq == 0)[0]) if has_zero else -1

    def neg_log_lik(sigma, delta):
        mean_present = n_reads / (lib_size * (1.0 - delta))
        mu = np.log(mean_present) - 0.5 * sigma**2
        log_pmf = _pln_log_pmf(uniq_f, mu, sigma)

        # Zero inflation applies only to the k = 0 term.
        log_lik_terms = np.log1p(-delta) + log_pmf
        if has_zero:
            log_lik_terms = log_lik_terms.copy()
            log_lik_terms[zero_idx] = np.logaddexp(
                np.log(delta) if delta > 0 else -np.inf,
                np.log1p(-delta) + log_pmf[zero_idx],
            )
        total = float(np.sum(multiplicity * log_lik_terms))
        return -total if np.isfinite(total) else 1e12

    # Without zero counts there is no dropout signal, so pin delta at 0.
    delta_hi = 0.9 if n_zero else 0.0
    delta0 = min(0.5 * n_zero / lib_size, 0.45) if n_zero else 0.0
    return neg_log_lik, delta_hi, delta0


def _fit_zip_lognormal(counts: np.ndarray, sigma0: float) -> tuple:
    """Jointly fit (sigma, dropout_fraction) by maximum likelihood.

    Returns (sigma, delta, converged, neg_log_lik_at_optimum).
    """
    if counts.sum() <= 0:
        return 0.0, 0.0, False, float("inf")

    neg_log_lik, delta_hi, delta0 = _make_neg_log_lik(counts)

    result = minimize(
        lambda x: neg_log_lik(x[0], x[1]),
        x0=[max(sigma0, 0.05), delta0],
        bounds=[(1e-3, 3.0), (0.0, delta_hi)],
        method="L-BFGS-B",
    )
    sigma, delta = float(result.x[0]), float(result.x[1])
    return sigma, delta, bool(result.success), float(result.fun)


def _profile_ci_sigma(counts, sigma_hat, nll_min, drop=1.9207, n_bisect=24):
    """Profile-likelihood confidence interval for sigma.

    Walks outward from the optimum until the profiled negative
    log-likelihood rises by `drop` (half a chi-squared(1) 95% quantile),
    then bisects.  Dropout is re-optimized at each sigma so the interval
    reflects uncertainty in skew alone.

    Returns (sigma_low, sigma_high).
    """
    neg_log_lik, delta_hi, delta0 = _make_neg_log_lik(counts)

    def profiled(sigma):
        if delta_hi <= 0:
            return neg_log_lik(sigma, 0.0)
        res = minimize_scalar(
            lambda d: neg_log_lik(sigma, d),
            bounds=(0.0, delta_hi),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return float(res.fun)

    def search(direction):
        """Find where the profile crosses the threshold, going up or down."""
        lo, hi = _SIGMA_BOUNDS
        edge = hi if direction > 0 else lo
        inside, outside = sigma_hat, None

        step = 0.05
        probe = sigma_hat
        for _ in range(40):
            probe = probe + direction * step
            if not (lo < probe < hi):
                probe = edge
            if profiled(probe) - nll_min > drop:
                outside = probe
                break
            inside = probe
            if probe == edge:
                return edge
            step *= 1.5

        if outside is None:
            return edge

        for _ in range(n_bisect):
            mid = 0.5 * (inside + outside)
            if profiled(mid) - nll_min > drop:
                outside = mid
            else:
                inside = mid
        return 0.5 * (inside + outside)

    return search(-1), search(+1)


def _posterior_mean_lambda(
    counts: np.ndarray, mu: float, sigma: float, delta: float
) -> np.ndarray:
    """Empirical-Bayes posterior mean abundance for each variant.

    Uses the identity ``lambda * Poisson(k; lambda) = (k+1) * Poisson(k+1; lambda)``,
    so ``E[lambda | k] = (k+1) * P(k+1) / P(k)`` needs only the PLN pmf.

    Zero-count variants are additionally down-weighted by the posterior
    probability that they are present at all, which is what keeps likely
    synthesis dropouts from being handed to the simulator as recoverable.
    """
    uniq = np.unique(counts.astype(np.int64))
    uniq_f = uniq.astype(np.float64)

    log_p_k = _pln_log_pmf(uniq_f, mu, sigma)
    log_p_k1 = _pln_log_pmf(uniq_f + 1.0, mu, sigma)
    post_mean = (uniq_f + 1.0) * np.exp(log_p_k1 - log_p_k)

    if delta > 0 and (uniq == 0).any():
        zero_idx = int(np.flatnonzero(uniq == 0)[0])
        log_absent = np.log(delta)
        log_present = np.log1p(-delta) + log_p_k[zero_idx]
        p_present = float(np.exp(log_present - np.logaddexp(log_absent, log_present)))
        post_mean[zero_idx] *= p_present

    lookup = dict(zip(uniq, post_mean))
    return np.array([lookup[c] for c in counts.astype(np.int64)], dtype=np.float64)


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative abundance vector."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    total = x.sum()
    if total <= 0:
        return 0.0
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_skew(counts: VariantCounts, fit: str = "mle") -> SkewStats:
    """Estimate library skew from per-variant read counts.

    Args:
        counts: Per-variant read counts covering the whole library.
        fit: "mle" for the joint zero-inflated fit (default), or
            "moments" for the closed-form estimator, which is faster but
            attributes synthesis dropouts to skew.

    Returns:
        SkewStats with both the raw and noise-corrected skew.

    Raises:
        ValueError: If the library is empty or no reads were assigned.
    """
    if fit not in ("mle", "moments"):
        raise ValueError(f"fit must be 'mle' or 'moments', got {fit!r}")

    c = counts.as_array()
    lib_size = len(c)
    if lib_size == 0:
        raise ValueError("no variants in library")
    n_assigned = c.sum()
    if n_assigned <= 0:
        raise ValueError(
            "no reads were assigned to any variant — check that the FASTQ "
            "and the variant list describe the same library"
        )

    mean_depth = n_assigned / lib_size
    detected = c > 0
    n_detected = int(detected.sum())
    median_depth = float(np.median(c[detected])) if n_detected else 0.0

    # Raw observed skew, undefined when the bottom decile saw no reads.
    q90, q10 = np.percentile(c, 90), np.percentile(c, 10)
    q90_q10_observed = float(q90 / q10) if q10 > 0 else None

    sigma_mom = _moment_sigma(c)
    ci = (float("nan"), float("nan"))
    if fit == "mle":
        sigma, delta, converged, nll_min = _fit_zip_lognormal(c, sigma_mom)
        if converged:
            lo, hi = _profile_ci_sigma(c, sigma, nll_min)
            ci = (sigma_to_skew(lo), sigma_to_skew(hi))
        else:
            # Fall back to the closed form rather than report a bad optimum.
            sigma, delta = sigma_mom, 0.0
    else:
        sigma, delta, converged = sigma_mom, 0.0, True

    mean_present = n_assigned / (lib_size * (1.0 - delta))
    mu = math.log(mean_present) - 0.5 * sigma**2
    lam_hat = _posterior_mean_lambda(c, mu, sigma, delta)
    total = lam_hat.sum()
    shrunk = lam_hat / total if total > 0 else np.full(lib_size, 1.0 / lib_size)

    undetected_names = [n for n, v in counts.counts.items() if v == 0]

    return SkewStats(
        q90_q10_observed=q90_q10_observed,
        q90_q10_corrected=sigma_to_skew(sigma),
        q90_q10_ci=ci,
        sigma_log=sigma,
        mu_log=mu,
        dropout_fraction=delta,
        gini=_gini(shrunk),
        effective_library_size=float(1.0 / np.sum(shrunk**2)),
        n_detected=n_detected,
        n_undetected=lib_size - n_detected,
        undetected_names=undetected_names,
        median_depth=median_depth,
        mean_depth=float(mean_depth),
        depth_sufficient=bool(mean_depth >= _MIN_DEPTH_FOR_FIT),
        shrunk_abundance=shrunk,
        fit_method=fit,
        fit_converged=converged,
    )


def predicted_count_distribution(stats: SkewStats, library_size: int, max_count: int):
    """Expected number of variants at each read count, under the fitted model.

    This is the *observable* distribution: the fitted log-normal convolved
    with Poisson sampling, plus the dropout spike at zero.  Comparing it
    against the observed histogram is the goodness-of-fit check — it is
    broader than the underlying log-normal by exactly the amount counting
    noise accounts for.

    Args:
        stats: Output of :func:`measure_skew`.
        library_size: Number of variants the counts covered.
        max_count: Largest read count to evaluate.

    Returns:
        (counts, expected) arrays, with `counts` spanning 0..max_count and
        `expected` the number of variants predicted at each.
    """
    ks = np.arange(0, max(1, int(max_count)) + 1, dtype=np.float64)
    present = library_size * (1.0 - stats.dropout_fraction)
    expected = present * np.exp(
        _pln_log_pmf(ks, stats.mu_log, stats.sigma_log)
    )
    # Absent variants contribute only to the zero bin.
    expected[0] += library_size * stats.dropout_fraction
    return ks, expected


def underlying_log10_density(stats: SkewStats, library_size: int, edges):
    """Variants expected per bin from the *underlying* log-normal alone.

    Integrates the fitted Gaussian over bin edges given in log10(reads),
    with no Poisson broadening, so plotting it beside
    :func:`predicted_count_distribution` separates true abundance spread
    from counting noise.

    Args:
        stats: Output of :func:`measure_skew`.
        library_size: Number of variants the counts covered.
        edges: Bin edges in log10 reads.

    Returns:
        Array of expected variant counts, one per bin.
    """
    ln10 = math.log(10.0)
    mean_log10 = stats.mu_log / ln10
    sd_log10 = stats.sigma_log / ln10
    if sd_log10 <= 0:
        return np.zeros(len(edges) - 1)
    present = library_size * (1.0 - stats.dropout_fraction)
    cdf = norm.cdf((np.asarray(edges, dtype=np.float64) - mean_log10) / sd_log10)
    return present * np.diff(cdf)


def log10_histogram(counts: VariantCounts, stats: SkewStats, n_bins: int = 24):
    """Bin log10 abundance, with the fitted curves on the same bins.

    Shared by the plot and the terminal summary so both show identical
    numbers.  Zero-count variants are excluded — they have no log — and
    are available as ``stats.n_undetected``.

    Args:
        counts: VariantCounts.
        stats: Output of :func:`measure_skew`.
        n_bins: Number of bins across the observed range.

    Returns:
        (edges, observed, predicted, underlying). `edges` has n_bins+1
        entries in log10 reads; the other three have n_bins entries and
        are variant counts — observed, fitted-including-counting-noise,
        and the underlying log-normal with that noise removed.
    """
    values = counts.as_array()
    detected = values[values > 0]
    if len(detected) < 2:
        raise ValueError("need at least 2 detected variants to build a histogram")

    log_values = np.log10(detected)
    lo, hi = float(log_values.min()), float(log_values.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)

    observed, _ = np.histogram(log_values, bins=edges)

    # Read counts are integers, so sum the per-count probabilities into the
    # log-spaced bins rather than evaluating a density that ignores that.
    max_count = int(detected.max() * 2 + 10)
    ks, expected_per_count = predicted_count_distribution(
        stats, counts.library_size, max_count
    )
    nonzero = ks >= 1
    bin_index = np.digitize(np.log10(ks[nonzero]), edges) - 1
    predicted = np.zeros(n_bins)
    valid = (bin_index >= 0) & (bin_index < n_bins)
    np.add.at(predicted, bin_index[valid], expected_per_count[nonzero][valid])

    underlying = underlying_log10_density(stats, counts.library_size, edges)
    return edges, observed, predicted, underlying


def recommend_sampling(
    stats: SkewStats,
    *,
    target_coverage: float = 0.90,
    p_grow: float = 0.67,
    p_fail: float = 0.03,
    p_incorrect: float = 0.3,
    transformation_scale: float = 50,
    basis: str = "empirical",
    n_sims: int = 100,
    seed: int = 42,
    plate_format: int = 384,
    progress_callback=None,
) -> SamplingRecommendation:
    """Recommend a sorting depth from measured library statistics.

    Args:
        stats: Output of :func:`measure_skew`.
        target_coverage: Fraction of the library to recover.
        p_grow: Sorting efficiency (fraction of wells that grow).
        p_fail: PCR failure rate.
        p_incorrect: Fraction of assembled clones that are incorrect.
        transformation_scale: Transformant oversampling factor.
        basis: "empirical" to search against the measured (shrunk)
            abundances, or "lognormal" to regenerate a pool from the
            fitted skew, matching what `usortm plan` does.
        n_sims: Simulations per fold-sampling evaluation.
        seed: Random seed.
        plate_format: Wells per plate, for the plate count.
        progress_callback: Passed through to `find_fold_sampling`.

    Returns:
        SamplingRecommendation.
    """
    if basis not in ("empirical", "lognormal"):
        raise ValueError(f"basis must be 'empirical' or 'lognormal', got {basis!r}")

    from usortm.simulate.sample import generate_pool
    from usortm.simulate.sortm import find_fold_sampling, sortm

    lib_size = len(stats.shrunk_abundance)
    if basis == "empirical":
        pool = stats.shrunk_abundance
    else:
        pool = generate_pool(lib_size, stats.q90_q10_corrected, seed)

    ceiling = stats.coverage_ceiling
    # Sorting cannot recover variants that were never synthesized, so a
    # target above the ceiling is capped rather than chased forever.
    reachable = target_coverage <= ceiling
    search_target = target_coverage if reachable else ceiling * 0.99

    sim_kwargs = dict(
        p_grow=p_grow,
        p_fail=p_fail,
        p_incorrect=p_incorrect,
        transformation_scale=transformation_scale,
        n_sims=n_sims,
        seed=seed,
    )

    fold, coverage = find_fold_sampling(
        target_coverage=search_target,
        pool=pool,
        progress_callback=progress_callback,
        **sim_kwargs,
    )

    n_wells = int(math.ceil(lib_size * fold))
    n_plates = int(math.ceil(n_wells / plate_format))

    return SamplingRecommendation(
        fold_sampling=float(fold),
        n_wells=n_wells,
        n_plates=n_plates,
        expected_coverage=float(coverage),
        expected_coverage_of_present=float(coverage / ceiling) if ceiling > 0 else 0.0,
        target_coverage=target_coverage,
        coverage_ceiling=ceiling,
        target_reachable=reachable,
        basis=basis,
    )
