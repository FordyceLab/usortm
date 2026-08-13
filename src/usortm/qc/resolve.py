"""Pre-flight check: can these variants be told apart in ONT reads?

Counting reads per variant is only meaningful when variants differ enough
that a read can be attributed to one of them.  A library of single-codon
substitutions in a common backbone cannot be resolved read-by-read at
nanopore error rates, and running the count anyway would produce confident
nonsense rather than an obvious failure.

This module measures how far each variant sits from its nearest neighbour
by self-aligning the reference set with minimap2, which is far cheaper than
all-against-all pairwise alignment.
"""
from __future__ import annotations

import csv as csv_mod
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from usortm.demux.deps import find_minimap2

logger = logging.getLogger(__name__)

# Variants closer than this many edits are not reliably separable in ONT
# reads, whose per-base error rate is a few percent.
DEFAULT_WARN_BELOW = 10

__all__ = ["ResolvabilitySummary", "check_resolvability", "read_variant_sequences"]


@dataclass
class ResolvabilitySummary:
    """How separable the variants in a library are.

    Attributes:
        library_size: Number of rows in the variant CSV.
        n_unique_sequences: Distinct sequences among them.
        duplicate_groups: Lists of names sharing an identical sequence.
        min_distance: Smallest nearest-neighbour distance in the library.
        median_nn_distance: Median nearest-neighbour distance.
        n_below_threshold: Variants whose nearest neighbour is closer
            than `warn_below` edits.
        warn_below: Threshold used.
        verdict: "clean", "marginal", or "smeared".
    """

    library_size: int
    n_unique_sequences: int
    duplicate_groups: list = field(default_factory=list)
    min_distance: int = 0
    median_nn_distance: float = 0.0
    n_below_threshold: int = 0
    warn_below: int = DEFAULT_WARN_BELOW
    verdict: str = "clean"

    @property
    def is_usable(self) -> bool:
        """Whether read-level counting will produce meaningful abundances."""
        return self.verdict != "smeared"

    def to_dict(self) -> dict:
        return {
            "library_size": self.library_size,
            "n_unique_sequences": self.n_unique_sequences,
            "n_duplicate_groups": len(self.duplicate_groups),
            "duplicate_groups": self.duplicate_groups[:20],
            "min_distance": self.min_distance,
            "median_nn_distance": self.median_nn_distance,
            "n_below_threshold": self.n_below_threshold,
            "warn_below": self.warn_below,
            "verdict": self.verdict,
        }


def read_variant_sequences(variants_csv) -> dict:
    """Read a variant CSV into an ordered name -> sequence mapping.

    Mirrors :func:`usortm.demux.utils.csv_to_reference_fasta`: headers are
    whitespace-stripped and only uppercase characters are kept, so lowercase
    flanking regions are excluded.

    Raises:
        ValueError: If the CSV lacks Name/Sequence columns or is empty.
    """
    sequences: dict = {}
    with open(variants_csv, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        else:
            raise ValueError(f"{variants_csv} is empty")

        name_col = _pick_column(reader.fieldnames, ("Name", "name", "variant"))
        seq_col = _pick_column(reader.fieldnames, ("Sequence", "sequence", "seq"))
        if name_col is None or seq_col is None:
            raise ValueError(
                f"{variants_csv} must have 'Name' and 'Sequence' columns; "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            row = {k.strip(): (v or "") for k, v in row.items()}
            name = row[name_col].strip()
            seq = "".join(c for c in row[seq_col] if c.isupper())
            if not name or not seq:
                continue
            sequences[name] = seq

    if not sequences:
        raise ValueError(f"no usable variants found in {variants_csv}")
    return sequences


def _pick_column(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _find_duplicate_groups(sequences: dict) -> list:
    """Group variant names that share a byte-identical sequence."""
    by_seq = defaultdict(list)
    for name, seq in sequences.items():
        by_seq[seq].append(name)
    return [sorted(names) for names in by_seq.values() if len(names) > 1]


def _nearest_neighbour_distances(sequences: dict, minimap2_path, threads):
    """Distance from each variant to its closest other variant.

    Self-aligns the reference set with minimap2 and scores each pair by
    edits within the aligned block plus the unaligned overhang on both
    sides.  Variants with no alignment to any other variant are maximally
    distant and take their own length as the distance.
    """
    distances = {name: len(seq) for name, seq in sequences.items()}

    with tempfile.TemporaryDirectory(prefix="usortm_resolve_") as tmp:
        fasta = os.path.join(tmp, "refs.fasta")
        with open(fasta, "w") as fh:
            for name, seq in sequences.items():
                fh.write(f">{name}\n{seq}\n")

        cmd = [
            minimap2_path, "-c", "-x", "asm20",
            "-N", "10", "-p", "0.1",
            "-t", str(threads),
            fasta, fasta,
        ]
        logger.info("Self-aligning %d references for resolvability check", len(sequences))
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        )
        if proc.returncode != 0:
            logger.warning(
                "minimap2 self-alignment failed (exit %d); "
                "skipping resolvability distances", proc.returncode
            )
            return distances

        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) < 12:
                continue
            qname, tname = fields[0], fields[5]
            if qname == tname:
                continue
            qlen, qstart, qend = int(fields[1]), int(fields[2]), int(fields[3])
            tlen, tstart, tend = int(fields[6]), int(fields[7]), int(fields[8])
            n_match, aln_len = int(fields[9]), int(fields[10])

            # Edits inside the aligned block, plus whatever hangs off each end.
            distance = (
                (aln_len - n_match)
                + (qlen - (qend - qstart))
                + (tlen - (tend - tstart))
            )
            for name in (qname, tname):
                if name in distances and distance < distances[name]:
                    distances[name] = distance

    return distances


def check_resolvability(
    variants_csv,
    *,
    warn_below: int = DEFAULT_WARN_BELOW,
    threads: int = 4,
    minimap2_path=None,
) -> ResolvabilitySummary:
    """Assess whether reads can be attributed to individual variants.

    Args:
        variants_csv: CSV with Name and Sequence columns.
        warn_below: Nearest-neighbour distance below which a variant is
            considered hard to separate.
        threads: minimap2 threads.
        minimap2_path: Path to minimap2; auto-detected if None.

    Returns:
        ResolvabilitySummary. A "smeared" verdict means read-level counting
        will not give trustworthy abundances for this library.
    """
    sequences = read_variant_sequences(variants_csv)
    duplicate_groups = _find_duplicate_groups(sequences)
    n_unique = len(set(sequences.values()))

    if len(sequences) < 2:
        return ResolvabilitySummary(
            library_size=len(sequences),
            n_unique_sequences=n_unique,
            duplicate_groups=duplicate_groups,
            min_distance=0,
            median_nn_distance=0.0,
            warn_below=warn_below,
            verdict="clean",
        )

    if minimap2_path is None:
        minimap2_path = find_minimap2()

    distances = _nearest_neighbour_distances(sequences, minimap2_path, threads)

    # Identical sequences are unattributable regardless of what minimap2
    # reports, so pin them to zero.
    for group in duplicate_groups:
        for name in group:
            distances[name] = 0

    values = np.asarray(list(distances.values()), dtype=float)
    n_below = int((values < warn_below).sum())
    median_nn = float(np.median(values))

    if median_nn < warn_below:
        verdict = "smeared"
    elif n_below > 0:
        verdict = "marginal"
    else:
        verdict = "clean"

    return ResolvabilitySummary(
        library_size=len(sequences),
        n_unique_sequences=n_unique,
        duplicate_groups=duplicate_groups,
        min_distance=int(values.min()),
        median_nn_distance=median_nn,
        n_below_threshold=n_below,
        warn_below=warn_below,
        verdict=verdict,
    )
