"""Synthetic libraries and reads with known ground truth.

Generates a variant CSV and a matching FASTQ from a library whose abundance
distribution, dropouts, and per-variant read counts are all known, so the
whole measurement chain — alignment, counting, deconvolution, sorting
recommendation — can be checked against the answer rather than against
another estimate.

Two library shapes, because they behave completely differently:

``diverse``
    Unrelated sequences, as a designed library of distinct genes would be.
    Reads are attributable and `usortm skew` measures it directly.

``codon_scan``
    One shared backbone with single-codon substitutions, like an amber or
    site-saturation scan. Variants sit ~3 bp apart, which is below what
    nanopore reads can resolve one at a time, so this shape exists to
    exercise the separability guard and any estimator built for it.

Reads carry ONT-like substitutions, insertions and deletions, optional
truncation, and optional unrelated junk, so the coverage filter and the
unmapped tally see real input. Everything is seeded and reproducible.

Nothing here is committed to the repo — a 15k-read FASTQ is several MB, and
regenerating from a seed is cheaper than storing it.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from usortm.simulate.sample import generate_pool

BASES = ("A", "C", "G", "T")

# Rough ONT error split: mostly substitutions, then deletions, then
# insertions. Not tuned to any chemistry — it exists to make reads
# imperfect in all three ways.
_SUB_SHARE = 0.55
_DEL_SHARE = 0.25

__all__ = ["SyntheticLibrary", "make_synthetic_library"]


@dataclass
class SyntheticLibrary:
    """A generated library, its reads, and the truth behind them.

    Attributes:
        variants_csv: Path to the Name/Sequence CSV, with lowercase
            flanking adaptors around an uppercase variable region, matching
            the format of a real Twist order.
        fastq: Path to the generated reads.
        truth_json: Path to a JSON dump of everything below.
        true_abundance: Variant name -> true relative abundance. Absent
            variants are present as 0.0.
        true_counts: Variant name -> reads actually written for it.
        absent: Names given zero abundance (synthesis dropouts).
        realized_skew: Q90/Q10 of the abundances that are actually
            present. This, not the requested `skew`, is what an estimator
            should recover — a finite draw's realized skew differs from the
            distribution it came from.
        n_reads: Reads written, including junk.
        n_junk: Unrelated reads included.
        params: The generation parameters.
    """

    variants_csv: Path
    fastq: Path
    truth_json: Path
    true_abundance: dict
    true_counts: dict
    absent: list
    realized_skew: float
    n_reads: int
    n_junk: int
    params: dict = field(default_factory=dict)

    @property
    def library_size(self) -> int:
        return len(self.true_abundance)

    @property
    def n_absent(self) -> int:
        return len(self.absent)


def _random_seq(rng, length: int) -> str:
    return "".join(rng.choice(BASES, length))


def _apply_errors(rng, seq: str, error_rate: float) -> str:
    """Apply substitutions, deletions and insertions at `error_rate`/base."""
    if error_rate <= 0:
        return seq
    out = []
    draws = rng.random(len(seq))
    for base, r in zip(seq, draws):
        if r >= error_rate:
            out.append(base)
            continue
        kind = rng.random()
        if kind < _SUB_SHARE:
            out.append(rng.choice([b for b in BASES if b != base]))
        elif kind < _SUB_SHARE + _DEL_SHARE:
            continue                      # deletion
        else:
            out.append(base)
            out.append(rng.choice(BASES))  # insertion
    return "".join(out)


def _diverse_variants(rng, library_size, seq_length):
    """Unrelated sequences: every variant differs from every other."""
    return {f"var{i:04d}": _random_seq(rng, seq_length) for i in range(library_size)}


def _codon_scan_variants(rng, library_size, seq_length, alts_per_position=4):
    """One backbone, single-codon substitutions, plus a WT entry.

    Mirrors an amber/site-saturation scan: variants differ from WT at one
    codon and from each other by 3 bp (same position) or 6 bp (different
    positions).
    """
    if seq_length % 3:
        raise ValueError(f"codon_scan needs seq_length divisible by 3, got {seq_length}")
    n_codons = seq_length // 3
    n_positions = max(1, (library_size - 1) // alts_per_position)
    if n_positions > n_codons:
        raise ValueError(
            f"library_size {library_size} needs {n_positions} varied codons but "
            f"seq_length {seq_length} only has {n_codons}"
        )

    backbone = _random_seq(rng, seq_length)
    variants = {"WT": backbone}
    # Spread the varied codons evenly along the backbone.
    positions = np.linspace(0, n_codons - 1, n_positions).astype(int)
    for pos in positions:
        wt_codon = backbone[pos * 3:pos * 3 + 3]
        seen = {wt_codon}
        for j in range(alts_per_position):
            for _ in range(50):
                codon = "".join(rng.choice(BASES, 3))
                if codon not in seen:
                    break
            else:
                continue
            seen.add(codon)
            seq = backbone[:pos * 3] + codon + backbone[pos * 3 + 3:]
            variants[f"c{int(pos):03d}_{j}"] = seq
    return variants


def make_synthetic_library(
    out_dir,
    *,
    library_size: int = 300,
    seq_length: int = 300,
    skew: float = 4.0,
    dropout: float = 0.0,
    n_reads: int = 15000,
    error_rate: float = 0.03,
    truncation_rate: float = 0.02,
    junk_fraction: float = 0.005,
    flank_length: int = 12,
    mode: str = "diverse",
    seed: int = 0,
) -> SyntheticLibrary:
    """Generate a library CSV and FASTQ with known ground truth.

    Args:
        out_dir: Directory for variants.csv, library.fastq and truth.json.
        library_size: Number of variants (approximate for codon_scan,
            which rounds to whole positions and adds a WT entry).
        seq_length: Length of the uppercase variable region. Must be
            divisible by 3 for codon_scan.
        skew: Requested Q90/Q10 of the abundance distribution. Check
            `realized_skew` for what was actually drawn.
        dropout: Fraction of variants given zero abundance, standing in
            for synthesis failures.
        n_reads: Reads to write, junk included.
        error_rate: Per-base probability of a substitution, insertion or
            deletion.
        truncation_rate: Fraction of reads trimmed at one end, to exercise
            the reference-coverage filter.
        junk_fraction: Fraction of reads that are unrelated sequence, to
            exercise the unmapped tally.
        flank_length: Lowercase adaptor length on each side of the
            variable region in the CSV. Reads include these, so they
            overhang the reference as real amplicon reads do.
        mode: "diverse" or "codon_scan".
        seed: Random seed.

    Returns:
        SyntheticLibrary.

    Raises:
        ValueError: On an unknown mode or an infeasible codon_scan request.
    """
    if mode not in ("diverse", "codon_scan"):
        raise ValueError(f"mode must be 'diverse' or 'codon_scan', got {mode!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if mode == "diverse":
        variants = _diverse_variants(rng, library_size, seq_length)
    else:
        variants = _codon_scan_variants(rng, library_size, seq_length)
    names = list(variants)
    actual_size = len(names)

    flank_5 = _random_seq(rng, flank_length).lower()
    flank_3 = _random_seq(rng, flank_length).lower()

    variants_csv = out_dir / "variants.csv"
    with open(variants_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Name", "Sequence"])
        for name in names:
            writer.writerow([name, flank_5 + variants[name] + flank_3])

    # True abundances, with dropouts zeroed and the rest renormalized.
    abundance = generate_pool(actual_size, skew, seed)
    absent_mask = rng.random(actual_size) < dropout
    if absent_mask.all():                     # never zero out everything
        absent_mask[:] = False
    abundance = abundance * ~absent_mask
    abundance = abundance / abundance.sum()
    absent = [names[i] for i in range(actual_size) if absent_mask[i]]

    present = abundance[abundance > 0]
    realized_skew = float(
        np.percentile(present, 90) / np.percentile(present, 10)
    ) if len(present) > 1 else 1.0

    n_junk = int(round(n_reads * junk_fraction))
    n_real = max(0, n_reads - n_junk)
    draws = rng.multinomial(n_real, abundance)

    true_counts = {name: int(n) for name, n in zip(names, draws)}
    fastq = out_dir / "library.fastq"
    read_id = 0
    with open(fastq, "w") as fh:
        for name, count in zip(names, draws):
            template = flank_5.upper() + variants[name] + flank_3.upper()
            for _ in range(int(count)):
                read = _apply_errors(rng, template, error_rate)
                if rng.random() < truncation_rate:
                    # Trim one end hard enough to fall under a coverage filter.
                    keep = int(len(read) * rng.uniform(0.25, 0.6))
                    read = read[:keep] if rng.random() < 0.5 else read[-keep:]
                fh.write(f"@read{read_id}_{name}\n{read}\n+\n{'5' * len(read)}\n")
                read_id += 1
        for _ in range(n_junk):
            read = _random_seq(rng, seq_length)
            fh.write(f"@read{read_id}_junk\n{read}\n+\n{'5' * len(read)}\n")
            read_id += 1

    params = {
        "mode": mode,
        "library_size": actual_size,
        "seq_length": seq_length,
        "requested_skew": skew,
        "realized_skew": round(realized_skew, 4),
        "dropout": dropout,
        "n_absent": len(absent),
        "n_reads": read_id,
        "n_junk": n_junk,
        "error_rate": error_rate,
        "truncation_rate": truncation_rate,
        "flank_length": flank_length,
        "seed": seed,
        "mean_depth": round(n_real / actual_size, 2),
    }

    truth_json = out_dir / "truth.json"
    with open(truth_json, "w") as fh:
        json.dump({
            "params": params,
            "true_abundance": {n: float(a) for n, a in zip(names, abundance)},
            "true_counts": true_counts,
            "absent": absent,
        }, fh, indent=2)

    return SyntheticLibrary(
        variants_csv=variants_csv,
        fastq=fastq,
        truth_json=truth_json,
        true_abundance={n: float(a) for n, a in zip(names, abundance)},
        true_counts=true_counts,
        absent=absent,
        realized_skew=realized_skew,
        n_reads=read_id,
        n_junk=n_junk,
        params=params,
    )
