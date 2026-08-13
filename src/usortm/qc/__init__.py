"""Pre-sort QC for uSort-M libraries.

Measures how evenly an amplified library is distributed, from a shallow
sequencing run of the library itself (e.g. Plasmidsaurus premium PCR), and
turns that measurement into a sorting-depth recommendation.

Typical use::

    from usortm.qc import profile_library

    profile = profile_library("library.fastq", "variants.csv", "skew_out/")
    print(profile.stats.q90_q10_corrected)
    print(profile.recommendation.fold_sampling)

`usortm skew` wraps the same code path.

To check the measurement against a known answer, :mod:`usortm.qc.synthetic`
generates a library CSV and FASTQ whose abundances and dropouts are known.
It is imported directly rather than re-exported here, so that pulling in the
measurement API does not also pull in the simulation stack::

    from usortm.qc.synthetic import make_synthetic_library
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from usortm.qc.counting import (
    collect_fastqs,
    count_variant_reads,
    write_reference_fasta,
)
from usortm.qc.resolve import (
    ResolvabilitySummary,
    check_resolvability,
    read_variant_sequences,
)
from usortm.qc.skew import (
    SamplingRecommendation,
    SkewStats,
    VariantCounts,
    ci_to_json,
    measure_skew,
    recommend_sampling,
    sigma_to_skew,
    skew_to_sigma,
)

__all__ = [
    "LibraryProfile",
    "profile_library",
    "VariantCounts",
    "SkewStats",
    "SamplingRecommendation",
    "ResolvabilitySummary",
    "collect_fastqs",
    "count_variant_reads",
    "write_reference_fasta",
    "check_resolvability",
    "read_variant_sequences",
    "ci_to_json",
    "measure_skew",
    "recommend_sampling",
    "sigma_to_skew",
    "skew_to_sigma",
]


@dataclass
class LibraryProfile:
    """Everything `profile_library` measured about one library."""

    counts: VariantCounts
    stats: SkewStats
    recommendation: SamplingRecommendation
    resolvability: Optional[ResolvabilitySummary] = None

    def to_dict(self) -> dict:
        """JSON-serializable summary of the whole profile."""
        return {
            "reads": {
                "total": self.counts.total_reads,
                "assigned": self.counts.assigned_reads,
                "ambiguous": self.counts.ambiguous,
                "low_coverage": self.counts.low_cov,
                "unmapped": self.counts.unmapped,
            },
            "library_size": self.counts.library_size,
            "skew": self.stats.to_dict(),
            "recommendation": self.recommendation.to_dict(),
            "resolvability": (
                self.resolvability.to_dict() if self.resolvability else None
            ),
        }


def profile_library(
    fastq,
    variants_csv,
    work_dir,
    *,
    target_coverage: float = 0.90,
    p_grow: float = 0.67,
    p_fail: float = 0.03,
    p_incorrect: float = 0.3,
    basis: str = "empirical",
    fit: str = "mle",
    min_ref_cov: float = 0.8,
    margin: float = 0.02,
    threads: int = 4,
    n_sims: int = 100,
    seed: int = 42,
    skip_resolvability: bool = False,
    count_progress=None,
    sim_progress=None,
) -> LibraryProfile:
    """Measure library skew from reads and recommend a sorting depth.

    Args:
        fastq: Library sequencing reads (plain or gzipped) — a FASTQ file, a
            directory searched recursively for FASTQs, or a list of either.
        variants_csv: CSV of the starting variants (Name, Sequence).
        work_dir: Directory for intermediate files.
        target_coverage: Fraction of the library to recover by sorting.
        p_grow: Sorting efficiency (fraction of wells that grow).
        p_fail: PCR failure rate.
        p_incorrect: Fraction of assembled clones that are incorrect.
        basis: "empirical" (measured abundances) or "lognormal" (fitted).
        fit: "mle" or "moments"; see `measure_skew`.
        min_ref_cov: Minimum reference coverage for a read to count.
        margin: Minimum relative score lead for unambiguous assignment.
        threads: minimap2 threads.
        n_sims: Simulations per fold-sampling evaluation.
        seed: Random seed.
        skip_resolvability: Skip the pre-flight separability check.
        count_progress: Callback ``(n_done, total)`` during counting.
        sim_progress: Callback ``(iteration, fold, coverage)`` during search.

    Returns:
        LibraryProfile.
    """
    resolvability = None
    if not skip_resolvability:
        resolvability = check_resolvability(variants_csv, threads=threads)

    counts = count_variant_reads(
        fastq,
        variants_csv,
        work_dir,
        min_ref_cov=min_ref_cov,
        margin=margin,
        threads=threads,
        progress_callback=count_progress,
    )
    stats = measure_skew(counts, fit=fit)
    recommendation = recommend_sampling(
        stats,
        target_coverage=target_coverage,
        p_grow=p_grow,
        p_fail=p_fail,
        p_incorrect=p_incorrect,
        basis=basis,
        n_sims=n_sims,
        seed=seed,
        progress_callback=sim_progress,
    )
    return LibraryProfile(
        counts=counts,
        stats=stats,
        recommendation=recommendation,
        resolvability=resolvability,
    )
