"""Synthetic barcoded sequencing runs with known well assignments.

:mod:`usortm.qc.synthetic` generates reads for measuring abundance — they
carry no barcodes and belong to no well, so they cannot drive demultiplexing.
This module builds the other kind: reads assembled the way a real LevSeq
amplicon is, so a whole demux run can be checked against the wells it was
built from.

Each read is laid out as the barcode arrangement expects::

    mask1_front  FBC  mask1_rear  [5' flank | variant | 3' flank]
                                  mask2_front  revcomp(RBC)  mask2_rear

The forward barcode identifies the well within a 96 grid and the reverse
barcode identifies the plate and quadrant, which is what
:func:`usortm.demux.utils.barcode_to_well` decodes.  Reads are emitted in
both orientations so the strand-splitting step has something to do, and
carry ONT-like substitutions, insertions and deletions.

A run may be split across several FASTQs with their own barcode-plate to
sort-plate mappings, which is how libraries larger than the kit's eight
barcode plates are sequenced.  The matching ``plate_map.toml`` is written
alongside, so the output drives ``usortm demux --plate-map`` directly.

Everything is seeded and reproducible; nothing here is committed to the repo.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from usortm.demux.barcodes import DEFAULT_MASKS, LEVSEQ_FBC, LEVSEQ_RBC
from usortm.demux.utils import well_to_barcode
from usortm.qc.synthetic import _apply_errors, _random_seq

__all__ = ["SyntheticDemuxRun", "make_synthetic_demux_run"]

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass
class SyntheticDemuxRun:
    """A generated run and the truth behind it.

    Attributes:
        out_dir: Directory holding every generated file.
        variants_csv: Library CSV, ``name,sequence``, with lowercase flanks.
        vector_fasta: Vector backbone with the variable region marked by X.
        plate_map_toml: Plate map covering the generated FASTQs.
        fastqs: Segment name to FASTQ path.
        truth: ``{"<plate><row><col>": {"variant": ..., "reads": ...}}``.
        n_reads: Total reads written, including unassignable ones.
        segments: Segment name to ``{barcode_plate: sort_plate}``.
    """

    out_dir: Path
    variants_csv: Path
    vector_fasta: Path
    plate_map_toml: Path
    fastqs: dict
    truth: dict
    n_reads: int
    segments: dict

    def expected_wells(self) -> int:
        """Number of wells that were given reads."""
        return len(self.truth)

    def expected_variants(self) -> int:
        """Number of distinct variants placed in wells."""
        return len({w["variant"] for w in self.truth.values()})


def _well_name(plate: int, row: int, col: int) -> str:
    """Well key in the form barcode_to_well() produces, e.g. ``3B12``."""
    return f"{plate}{chr(ord('A') + row - 1)}{col}"


def _build_read(fbc_seq, rbc_seq, amplicon, masks):
    """Assemble one full-length amplicon read in forward orientation."""
    return (
        masks["mask1_front"] + fbc_seq + masks["mask1_rear"]
        + amplicon
        + masks["mask2_front"] + _rc(rbc_seq) + masks["mask2_rear"]
    )


def make_synthetic_demux_run(
    out_dir,
    *,
    library_size: int = 96,
    seq_length: int = 300,
    segments: Optional[dict] = None,
    occupancy: float = 0.75,
    mean_reads_per_well: float = 40,
    depth_sigma: float = 0.5,
    error_rate: float = 0.04,
    flank_5p_length: int = 120,
    flank_3p_length: int = 150,
    junk_fraction: float = 0.02,
    rows: int = 16,
    cols: int = 24,
    seed: int = 0,
) -> SyntheticDemuxRun:
    """Generate a barcoded run with known per-well assignments.

    Args:
        out_dir: Directory for the generated files.
        library_size: Number of distinct variants.
        seq_length: Length of the variable region.
        segments: ``{segment_name: {barcode_plate: sort_plate}}``.  Defaults
            to a single segment covering sort plate 1.
        occupancy: Fraction of wells given reads.
        mean_reads_per_well: Median well depth; depths are drawn lognormally
            around it so a few wells are much deeper than the rest.
        depth_sigma: Spread of the depth distribution, in log space.
        error_rate: Per-base substitution/insertion/deletion probability.
        flank_5p_length: Vector sequence before the variable region.
        flank_3p_length: Vector sequence after it.
        junk_fraction: Fraction of reads that are unrelated sequence, so the
            unclassified tally is not empty.
        rows: Plate rows to use, up to 16.  Lower it for smaller runs.
        cols: Plate columns to use, up to 24.
        seed: Random seed.

    Returns:
        SyntheticDemuxRun.

    Raises:
        ValueError: If a barcode plate is outside 1-8, or the grid exceeds
            384 wells.
    """
    if segments is None:
        segments = {"run1": {1: 1}}
    if not 1 <= rows <= 16 or not 1 <= cols <= 24:
        raise ValueError(f"grid must fit a 384-well plate, got {rows}x{cols}")
    for name, plates in segments.items():
        for bc in plates:
            if not 1 <= bc <= 8:
                raise ValueError(
                    f"segment {name!r}: barcode plate {bc} outside 1-8"
                )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- Library and vector backbone ---
    variants = {
        f"var_{i + 1:04d}": _random_seq(rng, seq_length)
        for i in range(library_size)
    }
    flank_5p = _random_seq(rng, flank_5p_length)
    flank_3p = _random_seq(rng, flank_3p_length)

    variants_csv = out_dir / "variants.csv"
    with open(variants_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "sequence"])
        for name, seq in variants.items():
            # Lowercase flanks mark what csv_to_reference_fasta strips.
            writer.writerow([name, flank_5p.lower() + seq + flank_3p.lower()])

    vector_fasta = out_dir / "vector.fasta"
    vector_fasta.write_text(
        f">vector_backbone\n{flank_5p}{'X' * seq_length}{flank_3p}\n"
    )

    # --- Reads, well by well ---
    masks = DEFAULT_MASKS["fbc"]
    variant_names = list(variants)
    truth: dict = {}
    fastqs: dict = {}
    total_reads = 0

    for seg_name, plates in segments.items():
        fq_path = out_dir / f"{seg_name}.fastq"
        n_seg_reads = 0
        with open(fq_path, "w") as fh:
            for barcode_plate, sort_plate in sorted(plates.items()):
                for row in range(1, rows + 1):
                    for col in range(1, cols + 1):
                        if rng.random() > occupancy:
                            continue
                        variant = variant_names[
                            int(rng.integers(len(variant_names)))
                        ]
                        depth = int(np.clip(
                            rng.lognormal(np.log(mean_reads_per_well),
                                          depth_sigma),
                            1, 100_000,
                        ))
                        fbc_n, rbc_n = well_to_barcode(barcode_plate, row, col)
                        amplicon = flank_5p + variants[variant] + flank_3p

                        for k in range(depth):
                            read = _build_read(
                                LEVSEQ_FBC[fbc_n - 1], LEVSEQ_RBC[rbc_n - 1],
                                amplicon, masks,
                            )
                            # Half the library is sequenced in each direction;
                            # the pipeline orients them before demuxing.
                            if rng.random() < 0.5:
                                read = _rc(read)
                            read = _apply_errors(rng, read, error_rate)
                            rid = f"{seg_name}_p{sort_plate}_{row}_{col}_{k}"
                            fh.write(f"@{rid}\n{read}\n+\n{'I' * len(read)}\n")
                            n_seg_reads += 1

                        # Truth is keyed by SORT plate, which is what the
                        # pipeline reports after applying the plate map.
                        truth[_well_name(sort_plate, row, col)] = {
                            "variant": variant,
                            "reads": depth,
                            "segment": seg_name,
                            "barcode_plate": barcode_plate,
                        }

            n_junk = int(n_seg_reads * junk_fraction)
            for j in range(n_junk):
                junk = _random_seq(rng, int(rng.integers(200, 900)))
                fh.write(f"@{seg_name}_junk_{j}\n{junk}\n+\n{'I' * len(junk)}\n")
            n_seg_reads += n_junk

        fastqs[seg_name] = fq_path
        total_reads += n_seg_reads

    # --- Plate map covering the generated FASTQs ---
    from usortm.demux.plate_map import Segment, write_plate_map

    plate_map_toml = write_plate_map(
        [Segment(name=n, path=fastqs[n], plates=p) for n, p in segments.items()],
        out_dir / "plate_map.toml",
    )

    truth_path = out_dir / "truth.json"
    truth_path.write_text(json.dumps({
        "wells": truth,
        "n_reads": total_reads,
        "library_size": library_size,
        "seq_length": seq_length,
        "flank_5p_len": flank_5p_length,
        "flank_3p_len": flank_3p_length,
        "segments": {n: {str(k): v for k, v in p.items()}
                     for n, p in segments.items()},
        "seed": seed,
    }, indent=2))

    return SyntheticDemuxRun(
        out_dir=out_dir,
        variants_csv=variants_csv,
        vector_fasta=vector_fasta,
        plate_map_toml=plate_map_toml,
        fastqs=fastqs,
        truth=truth,
        n_reads=total_reads,
        segments=segments,
    )
