"""Streak-out candidate detection and per-well pileup visualization.

Identifies wells containing two or more correctly-assembled subpopulations
of reads.  When the minority variant was missed during initial sampling,
users can streak out those wells to isolate it.

Algorithm
---------
For each well where the dominant fraction is below a threshold (default 0.9)
and read depth is sufficient (default ≥50):

1. Group reads by reference (strip strand prefix).
2. Drop groups with fewer than *min_group_reads* reads.
3. For each surviving group, generate a consensus (minimap2 → samtools
   consensus → re-align) and classify the CIGAR as Perfect Match, Silent
   Mutation, or other.
4. If **2+ groups** have a correct consensus (Perfect Match or Silent
   Mutation), the well is a streak-out candidate.  The minority group(s)
   are the recoverable variants.

Outputs
-------
- ``streakout_candidates.csv`` — one row per candidate well
- ``well_{plate}_{well}.html`` — per-well pileup visualization
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import pysam
from Bio.Seq import Seq
from Bio import SeqIO
from seqviewer import (
    PileupGroup,
    PileupView,
    Read,
    grid_from_reads,
    reads_from_alignment,
    render,
)

from usortm.demux.deps import find_minimap2, find_samtools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CIGAR classification (mirrors utils.extract_matches logic)
# ---------------------------------------------------------------------------

def _classify_cigar(cigar: Optional[str], ref_len: int,
                    ref_seq: str, cons_seq: Optional[str]) -> str:
    """Classify a consensus CIGAR string as a match status.

    Expects CIGAR produced with minimap2 ``--eqx`` so that ``=`` denotes
    sequence match and ``X`` denotes mismatch.  Falls back to sequence
    comparison when legacy ``M`` operators are present.

    Returns one of: "Perfect Match", "Silent Mutation", "Partial Match",
    "Other Error", or "Error".
    """
    import re as _re

    if cigar is None or cons_seq is None:
        return "Error"

    # Parse CIGAR into (length, op) pairs
    ops = _re.findall(r'(\d+)([A-Z=])', cigar)
    if not ops:
        return "Error"

    op_letters = set(op for _, op in ops)
    total_ref = sum(int(n) for n, op in ops if op in ('M', '=', 'X', 'D', 'N'))

    # --eqx mode: '=' and 'X' operators
    if '=' in op_letters or 'X' in op_letters:
        has_mismatch = 'X' in op_letters
        has_indel = bool(op_letters & {'I', 'D', 'N'})
        if not has_mismatch and not has_indel and total_ref == ref_len:
            return "Perfect Match"
        # Full-length alignment with only substitutions — check silent
        if not has_indel and total_ref == ref_len and len(cons_seq) == ref_len:
            try:
                if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                    return "Silent Mutation"
            except Exception:
                pass
        if total_ref == ref_len:
            return "Other Error"
        return "Partial Match"

    # Legacy M-only CIGAR: compare sequences directly
    if op_letters == {'M'} and total_ref == ref_len:
        if cons_seq.upper() == ref_seq.upper():
            return "Perfect Match"
        # Check for silent mutation
        if len(cons_seq) == ref_len:
            try:
                if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                    return "Silent Mutation"
            except Exception:
                pass
        return "Other Error"

    if total_ref < ref_len:
        return "Partial Match"
    return "Other Error"


def _is_correct(status: str) -> bool:
    return status in ("Perfect Match", "Silent Mutation")


def _cigar_is_clean(cigar_str: Optional[str]) -> bool:
    """Return True if CIGAR has no mismatches, insertions, or deletions.

    With --eqx alignment, '=' means exact base match and 'X' means mismatch.
    A recoverable group must have only '=' operations (plus soft clips 'S').
    """
    if not cigar_str:
        return False
    import re as _re
    return not _re.search(r"[XIDHN]", cigar_str)


# ---------------------------------------------------------------------------
# Per-group consensus (reuses _process_single_well pattern)
# ---------------------------------------------------------------------------

def _group_consensus(reads_fastq: str, ref_fasta: str, work_dir: str,
                     minimap2_path: str, samtools_path: str,
                     ) -> tuple[Optional[str], Optional[str]]:
    """Align reads → consensus → re-align → extract CIGAR for one group.

    Returns (cigar_str, cons_seq) or (None, None) on failure.
    """
    bam = os.path.join(work_dir, "aligned.bam")
    cons_fa = os.path.join(work_dir, "consensus.fasta")
    cons_bam = os.path.join(work_dir, "consensus.bam")

    # 1) Align reads to reference
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", ref_fasta, reads_fastq],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", bam],
            stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
        )
        mm2.wait()
    except Exception:
        return None, None

    # 2) Generate consensus
    try:
        with open(cons_fa, "w") as fh:
            subprocess.run(
                [samtools_path, "consensus", "-f", "fasta", bam],
                stdout=fh, check=False,
            )
    except Exception:
        return None, None

    # 3) Re-align consensus to reference (--eqx for =/X CIGAR ops)
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", "--eqx", ref_fasta, cons_fa],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", cons_bam],
            stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
        )
        mm2.wait()
    except Exception:
        return None, None

    # 4) Extract CIGAR + consensus sequence
    cigar_str, cons_seq = None, None
    try:
        with pysam.AlignmentFile(cons_bam, "rb") as bf:
            for read in bf:
                if not read.is_unmapped:
                    cigar_str = read.cigarstring
                    break
        if os.path.exists(cons_fa):
            with open(cons_fa) as fh:
                lines = fh.read().splitlines()
                cons_seq = "".join(l for l in lines if not l.startswith(">"))
    except Exception:
        pass

    return cigar_str, cons_seq


# ---------------------------------------------------------------------------
# Orient-ref mode helpers
# ---------------------------------------------------------------------------

def _parse_mpileup_bases(bases_str: str, ref_base: str) -> dict:
    """Count ACGT bases from a samtools mpileup base column."""
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    ref_base = ref_base.upper()
    i = 0
    while i < len(bases_str):
        c = bases_str[i]
        if c in ".,":
            if ref_base in counts:
                counts[ref_base] += 1
        elif c.upper() in "ACGT":
            counts[c.upper()] += 1
        elif c == "^":
            i += 1  # skip mapping-quality char
        elif c in "+-":
            i += 1
            num_str = ""
            while i < len(bases_str) and bases_str[i].isdigit():
                num_str += bases_str[i]
                i += 1
            if num_str:
                i += int(num_str) - 1
        i += 1
    return counts


def _find_bimodal_positions(
    bam_path: str,
    samtools_path: str,
    min_depth: int = 10,
    min_minor_frac: float = 0.15,
    max_minor_frac: float = 0.85,
) -> list:
    """Return positions with bimodal allele frequencies from samtools mpileup.

    Reads the whole BAM without requiring an index.  Returns a list of
    ``(chrom, pos_1based, ref_base, major_base, minor_base, minor_frac)``
    tuples sorted by closeness to 0.5 (most informative first).
    """
    try:
        result = subprocess.run(
            [samtools_path, "mpileup", "--no-BAQ", "-Q", "10", bam_path],
            capture_output=True, text=True, check=False,
        )
    except Exception:
        return []

    bimodal = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            chrom, pos, ref_base = parts[0], int(parts[1]), parts[2].upper()
            depth = int(parts[3])
        except (ValueError, IndexError):
            continue
        if depth < min_depth:
            continue

        counts = _parse_mpileup_bases(parts[4], ref_base)
        total = sum(counts.values())
        if total < min_depth:
            continue

        sorted_bases = sorted(counts.items(), key=lambda x: -x[1])
        if len(sorted_bases) < 2 or sorted_bases[1][1] == 0:
            continue

        minor_frac = sorted_bases[1][1] / total
        if min_minor_frac <= minor_frac <= max_minor_frac:
            bimodal.append((
                chrom, pos, ref_base,
                sorted_bases[0][0], sorted_bases[1][0],
                minor_frac,
            ))

    # Most balanced (closest to 0.5) first
    bimodal.sort(key=lambda x: abs(x[5] - 0.5))
    return bimodal


def _split_reads_by_haplotype(
    bam_path: str,
    bimodal_positions: list,
    well_reads_df,
    min_group_reads: int,
):
    """Split reads into 2 haplotype groups at the most informative bimodal position.

    Fetches all reads from the BAM without requiring an index and groups them
    by their allele at the position closest to 50% minor-allele frequency.

    Returns ``(group_a_df, group_b_df)`` or ``None`` if either group is too small.
    """
    if not bimodal_positions:
        return None

    chrom, pos_1based, _ref_base, _major_base, minor_base, _ = bimodal_positions[0]
    pos_0based = pos_1based - 1

    minor_names: set = set()
    major_names: set = set()

    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if read.reference_start > pos_0based or (read.reference_end or 0) <= pos_0based:
                    continue
                for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                    if rpos == pos_0based:
                        base = read.query_sequence[qpos].upper()
                        if base == minor_base:
                            minor_names.add(read.query_name)
                        else:
                            major_names.add(read.query_name)
                        break
    except Exception:
        return None

    if len(minor_names) < min_group_reads or len(major_names) < min_group_reads:
        return None

    group_a = well_reads_df[well_reads_df["read_name"].isin(minor_names)]
    group_b = well_reads_df[well_reads_df["read_name"].isin(major_names)]
    return group_a, group_b


def _assign_and_classify_group(
    cons_seq: Optional[str],
    ref_ids: list,
    ref_seqs_str: list,
    ref_matrix,
) -> tuple:
    """Find best library variant for a consensus and classify it.

    Returns ``(ref_id, ref_seq, ref_len, status)``.
    """
    if cons_seq is None:
        return None, None, None, "Error"

    max_ref_len = ref_matrix.shape[1]
    cons_upper = cons_seq.upper()
    cons_arr = np.frombuffer(cons_upper.encode(), dtype=np.uint8)
    padded = np.zeros(max_ref_len, dtype=np.uint8)
    padded[: min(len(cons_arr), max_ref_len)] = cons_arr[:max_ref_len]
    matches = np.sum(ref_matrix == padded, axis=1)
    best_idx = int(np.argmax(matches))

    ref_id = ref_ids[best_idx]
    ref_seq = ref_seqs_str[best_idx]
    ref_len = len(ref_seq)

    # Classify by direct comparison (avoids a second subprocess round-trip)
    cons_up = cons_upper
    ref_up = ref_seq.upper()
    if cons_up == ref_up:
        status = "Perfect Match"
    elif len(cons_up) != ref_len:
        status = "Partial Match"
    else:
        try:
            status = (
                "Silent Mutation"
                if Seq.translate(ref_seq) == Seq.translate(cons_seq)
                else "Other Error"
            )
        except Exception:
            status = "Other Error"

    return ref_id, ref_seq, ref_len, status


def detect_streakout_candidates_orient_ref(
    well_df,
    read_df,
    reference_dir: str,
    reference_fasta: str,
    output_dir: str,
    minimap2_path: str = None,
    samtools_path: str = None,
    min_well_reads: int = 20,
    min_group_reads: int = 5,
    workers: int = 4,
) -> list:
    """Streak-out detection for orient-ref mode.

    Standard detection (group by ref_name) is blind when all reads share the
    same orient reference.  This function instead:

    1. Runs ``samtools mpileup`` on each well's existing BAM to find positions
       with bimodal allele frequencies (15–85 % minor allele).
    2. Splits reads into 2 haplotype groups at the most informative position.
    3. Generates a consensus per group (aligned to the orient ref, which is
       ≥99 % identical to the true variant).
    4. Assigns each consensus to the best library variant via numpy similarity.
    5. Flags wells where 2+ groups produce a correct consensus.

    No index on the per-well BAMs is required.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    ref_records = list(SeqIO.parse(reference_fasta, "fasta"))
    if not ref_records:
        return []

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")
    well_bam_dir = os.path.join(output_dir, "wells", "consensus")

    # Use the pre-built full-length combined FASTA for variant assignment
    # (flank_5p + insert + flank_3p per variant).  This is the same reference
    # used by assign_variants_from_reads and lives in output_dir.  If it is
    # absent, fall back to the insert-only library_reference FASTA.
    full_length_combined = os.path.join(output_dir, "full_length_refs.fasta")
    align_fasta = full_length_combined if os.path.exists(full_length_combined) else reference_fasta

    candidate_wells = well_df[well_df["depth"] >= min_well_reads]
    if candidate_wells.empty:
        return []

    logger.info(
        "Screening %d wells for streak-out candidates (orient-ref mode, depth >= %d)",
        len(candidate_wells), min_well_reads,
    )

    # Via utils so this bar stands down when the CLI owns the terminal,
    # like every other progress bar in the pipeline.
    from usortm.demux.utils import _bar

    # Grouped once: _process runs per candidate and would otherwise rescan the
    # whole read table each time, which is O(candidates x reads).
    _by_well = {k: g for k, g in read_df.groupby("well_pos")}

    def _process(row):
        wp = row["global_well"]
        bam_path = os.path.join(well_bam_dir, f"{wp}.bam")
        if not os.path.exists(bam_path):
            return None

        bimodal = _find_bimodal_positions(bam_path, samtools_path)
        if not bimodal:
            return None

        well_reads = _by_well.get(wp)
        if well_reads is None or well_reads.empty:
            return None

        split = _split_reads_by_haplotype(bam_path, bimodal, well_reads, min_group_reads)
        if split is None:
            return None

        plate = str(int(row["plate"]))
        well = str(row["well"])
        depth = int(row["depth"])
        orient_ref_name = str(row["major_ref"])
        orient_fa = os.path.join(single_ref_dir, f"{orient_ref_name}.fasta")
        if not os.path.exists(orient_fa):
            return None

        group_results = []
        for group_df in split:
            if len(group_df) < min_group_reads:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                fq_path = os.path.join(tmp, "group.fastq")
                with open(fq_path, "w") as fh:
                    for _, r in group_df.iterrows():
                        fh.write(
                            f"@{r['read_name']}\n{r['read_seq']}\n+\n{r['read_qual']}\n"
                        )

                # Assign variant by aligning reads directly to the full-length
                # library (majority vote over PAF hits).  This is the same
                # strategy as assign_variants_from_reads and is far more
                # reliable than aligning to the orient reference then comparing
                # the consensus — the orient-ref insert differs from the reads'
                # actual insert, producing a noisy consensus that gets matched
                # to the wrong library variant.
                from collections import Counter as _Counter
                ref_counts: _Counter = _Counter()
                try:
                    mm2 = subprocess.Popen(
                        [minimap2_path, "-x", "map-ont", "--secondary=no",
                         align_fasta, fq_path],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    )
                    for raw in mm2.stdout:
                        parts = raw.decode("utf-8", errors="replace").split("\t", 7)
                        if len(parts) >= 6:
                            ref_counts[parts[5]] += 1
                    mm2.wait()
                except Exception:
                    pass

                if not ref_counts:
                    continue
                ref_id = ref_counts.most_common(1)[0][0]

                # Compute consensus against the assigned variant for status check
                variant_fa = os.path.join(single_ref_dir, f"{ref_id}.fasta")
                _cigar, status = None, "Error"
                if os.path.exists(variant_fa):
                    tmp2 = os.path.join(tmp, "consensus")
                    os.makedirs(tmp2)
                    _cigar, cons_seq = _group_consensus(
                        fq_path, variant_fa, tmp2, minimap2_path, samtools_path,
                    )
                    if _cigar is not None:
                        ref_record = next(SeqIO.parse(variant_fa, "fasta"))
                        ref_seq_actual = str(ref_record.seq)
                        status = _classify_cigar(
                            _cigar, len(ref_seq_actual), ref_seq_actual, cons_seq,
                        )

            frac = len(group_df) / depth
            group_results.append({
                "variant": ref_id,
                "reads": len(group_df),
                "frac": round(frac, 4),
                "status": status,
                "cigar": _cigar,
                "is_major": (ref_id == orient_ref_name),
                "read_names": list(group_df["read_name"]),
            })

        # A well with bimodal reads IS a multiple-colony well regardless of
        # whether each sub-consensus is clean.  We only require 2+ groups to
        # have been produced from the split.  Only groups with a clean consensus
        # (Perfect Match / Silent Mutation) are listed as recoverable — mutated
        # sub-sequences are not worth streaking out to recover.
        if len(group_results) < 2:
            return None

        top_frac = max((g["frac"] for g in group_results), default=1.0)
        groups_sorted = sorted(group_results, key=lambda g: -g["frac"])
        recoverable = [g["variant"] for g in groups_sorted if _cigar_is_clean(g.get("cigar"))]
        return {
            "plate": plate,
            "well": well,
            "global_well": wp,
            "total_reads": depth,
            "top_frac": round(top_frac, 4),
            "groups": groups_sorted,
            "recoverable_variants": recoverable,
        }

    candidates = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, row): row["global_well"]
            for _, row in candidate_wells.iterrows()
        }
        for future in _bar(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result is not None:
                    candidates.append(result)
            except Exception as exc:
                logger.warning("Streakout check failed: %s", exc)

    candidates.sort(key=lambda c: c["global_well"])
    return candidates


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _process_well_for_streakout(
    well_pos: str,
    well_reads: pd.DataFrame,
    well_row: pd.Series,
    reference_dir: str,
    minimap2_path: str,
    samtools_path: str,
    min_group_reads: int,
) -> Optional[dict]:
    """Analyse a single well for streak-out candidacy."""
    plate = str(int(well_row["plate"]))
    well = str(well_row["well"])
    depth = int(well_row["depth"])
    top_frac = float(well_row["major_freq"])

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")

    # Group reads by reference (strip strand prefix)
    rdf = well_reads.copy()
    rdf["_ref"] = rdf["ref_name"].str.replace(r'^(fwd|rev):', '', regex=True)
    groups = rdf.groupby("_ref")

    group_results = []
    for ref_id, grp in groups:
        if len(grp) < min_group_reads:
            continue

        ref_fasta = os.path.join(single_ref_dir, f"{ref_id}.fasta")
        if not os.path.exists(ref_fasta):
            continue

        # Read reference sequence + length
        ref_record = next(SeqIO.parse(ref_fasta, "fasta"))
        ref_seq = str(ref_record.seq)
        ref_len = len(ref_seq)

        # Write group reads to temp FASTQ and run consensus pipeline
        with tempfile.TemporaryDirectory() as tmp:
            fq_path = os.path.join(tmp, "group.fastq")
            with open(fq_path, "w") as fq:
                for _, r in grp.iterrows():
                    fq.write(f"@{r['read_name']}\n{r['read_seq']}\n+\n{r['read_qual']}\n")

            cigar, cons_seq = _group_consensus(
                fq_path, ref_fasta, tmp, minimap2_path, samtools_path,
            )

        status = _classify_cigar(cigar, ref_len, ref_seq, cons_seq)
        n_reads = len(grp)
        frac = n_reads / depth if depth else 0

        group_results.append({
            "variant": ref_id,
            "reads": n_reads,
            "frac": round(frac, 4),
            "status": status,
            "cigar": cigar,
            "is_major": (frac >= top_frac - 0.01),
        })

    # A well with 2+ read groups is a multiple-colony well regardless of
    # whether each sub-consensus is clean.  Only groups with a clean consensus
    # (Perfect Match / Silent Mutation) are listed as recoverable — mutated
    # sub-sequences are not worth streaking out to recover.
    if len(group_results) < 2:
        return None

    groups_sorted = sorted(group_results, key=lambda g: -g["frac"])
    recoverable = [g["variant"] for g in groups_sorted if _cigar_is_clean(g.get("cigar"))]

    return {
        "plate": plate,
        "well": well,
        "global_well": well_pos,
        "total_reads": depth,
        "top_frac": round(top_frac, 4),
        "groups": groups_sorted,
        "recoverable_variants": recoverable,
    }


def detect_streakout_candidates(
    well_df: pd.DataFrame,
    read_df: pd.DataFrame,
    reference_dir: str,
    output_dir: str,
    minimap2_path: str = None,
    samtools_path: str = None,
    min_well_reads: int = 20,
    min_group_reads: int = 5,
    max_top_frac: float = 0.9,
    workers: int = 4,
    reference_fasta: str = None,
) -> list[dict]:
    """Detect wells with multiple correctly-assembled subpopulations.

    When *reference_fasta* is provided (orient-ref mode), dispatches to
    :func:`detect_streakout_candidates_orient_ref` which uses pileup
    bimodality instead of ref_name grouping.

    Otherwise, for each well where ``major_freq < max_top_frac`` and
    ``depth >= min_well_reads``, groups reads by reference, generates a
    per-group consensus, and checks whether 2+ groups produce a correct
    consensus (Perfect Match or Silent Mutation).

    Args:
        well_df: Per-well summary DataFrame (from ``generate_well_df``).
        read_df: Per-read DataFrame (from ``format_df``).
        reference_dir: Directory containing ``single_ref_fastas/`` subdirectory.
        output_dir: Pipeline output directory.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
        min_well_reads: Minimum total reads in a well to consider.
        min_group_reads: Minimum reads in a group to attempt consensus.
        max_top_frac: Maximum dominant fraction to flag as potential mixed well.
        workers: Number of parallel workers.
        reference_fasta: Full library FASTA (required for orient-ref mode).

    Returns:
        List of candidate dicts, one per streak-out well.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    # Orient-ref mode: all reads share the same ref_name, so we use
    # pileup bimodality instead of ref_name grouping.
    if reference_fasta is not None:
        return detect_streakout_candidates_orient_ref(
            well_df=well_df,
            read_df=read_df,
            reference_dir=reference_dir,
            reference_fasta=reference_fasta,
            output_dir=output_dir,
            minimap2_path=minimap2_path,
            samtools_path=samtools_path,
            min_well_reads=min_well_reads,
            min_group_reads=min_group_reads,
            workers=workers,
        )

    # Filter candidate wells
    mask = (well_df["major_freq"] < max_top_frac) & (well_df["depth"] >= min_well_reads)
    candidate_wells = well_df[mask]

    if candidate_wells.empty:
        return []

    logger.info(
        "Screening %d wells for streak-out candidates (frac < %.2f, depth >= %d)",
        len(candidate_wells), max_top_frac, min_well_reads,
    )

    candidates = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        _grouped = {k: g for k, g in read_df.groupby("well_pos")}
        for _, row in candidate_wells.iterrows():
            wp = row["global_well"]
            well_reads = _grouped.get(wp)
            if well_reads is None or well_reads.empty:
                continue
            fut = pool.submit(
                _process_well_for_streakout,
                wp, well_reads, row, reference_dir,
                minimap2_path, samtools_path, min_group_reads,
            )
            futures[fut] = wp

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result is not None:
                    candidates.append(result)
            except Exception as exc:
                logger.warning("Streakout analysis failed for %s: %s",
                               futures[fut], exc)

    candidates.sort(key=lambda c: c["global_well"])
    return candidates


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def save_streakout_results(candidates: list[dict], output_dir: str) -> dict:
    """Write streak-out candidates to CSV and return summary dict.

    Writes ``streakout_candidates.csv`` to *output_dir*.

    Returns:
        Summary dict suitable for inclusion in ``demux_summary.json``.
    """
    csv_path = os.path.join(output_dir, "streakout_candidates.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plate", "well", "total_reads", "top_frac",
            "n_groups", "recoverable_variants", "groups_json",
        ])
        for c in candidates:
            recoverable_set = set(c["recoverable_variants"])
            groups_info = [
                {
                    "variant": g["variant"],
                    "reads": g["reads"],
                    "frac": g["frac"],
                    "is_recoverable": g["variant"] in recoverable_set,
                }
                for g in c["groups"]
            ]
            writer.writerow([
                c["plate"],
                c["well"],
                c["total_reads"],
                c["top_frac"],
                len(c["groups"]),
                ";".join(c["recoverable_variants"]),
                json.dumps(groups_info),
            ])

    return {
        "candidates": len(candidates),
        "recoverable_variants": list({
            v for c in candidates for v in c["recoverable_variants"]
        }),
        "csv_path": csv_path,
    }


# ---------------------------------------------------------------------------
# Per-well pileup HTML
# ---------------------------------------------------------------------------

def generate_well_pileup_html(
    well_pos: str,
    read_df: pd.DataFrame,
    reference_dir: str,
    candidate_info: dict,
    output_path: str,
    minimap2_path: str = None,
    samtools_path: str = None,
    flank_5p_len: int = 0,
    flank_3p_len: int = 0,
    bam_path: str = None,
) -> None:
    """Generate an interactive pileup HTML for one streak-out candidate well.

    For orient-ref mode (groups have ``read_names``), reads are extracted
    directly from *bam_path* so they display against the reference they were
    actually aligned to — avoiding the mis-alignment caused by re-aligning
    against a different variant's insert sequence.

    For standard mode, reads are re-aligned to their respective references.

    Args:
        well_pos: Global well identifier (e.g. "1A3").
        read_df: Full per-read DataFrame.
        reference_dir: Directory containing ``single_ref_fastas/``.
        candidate_info: Candidate dict from :func:`detect_streakout_candidates`.
        output_path: Path to write the HTML file.
        minimap2_path: Path to minimap2. Auto-detected if None.
        samtools_path: Path to samtools. Auto-detected if None.
        bam_path: Per-well BAM (orient-ref mode only). When provided the
            reads are extracted from this BAM instead of being re-aligned.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")

    # Filter reads for this well
    well_reads = read_df[read_df["well_pos"] == well_pos].copy()
    well_reads["_ref"] = well_reads["ref_name"].str.replace(
        r'^(fwd|rev):', '', regex=True,
    )

    recoverable_set = set(candidate_info.get("recoverable_variants", []))

    def _make_section(ginfo: dict, grp: pd.DataFrame) -> Optional[dict]:
        """Build one pileup section dict for a group (standard mode)."""
        ref_id = ginfo["variant"]
        ref_fasta = os.path.join(single_ref_dir, f"{ref_id}.fasta")
        if not os.path.exists(ref_fasta):
            return None
        ref_record = next(SeqIO.parse(ref_fasta, "fasta"))
        ref_seq = str(ref_record.seq)
        ref_len = len(ref_seq)

        frac = ginfo.get("frac", len(grp) / max(candidate_info["total_reads"], 1))
        is_recoverable = ref_id in recoverable_set
        status = "Clean" if _cigar_is_clean(ginfo.get("cigar")) else "Mutation"

        pileup_rows = _build_pileup_grid(
            grp, ref_fasta, ref_seq,
            minimap2_path, samtools_path,
        )
        return {
            "ref_id": ref_id,
            "n_reads": len(pileup_rows),
            "frac": frac,
            "status": status,
            "is_recoverable": is_recoverable,
            "ref_seq": ref_seq,
            "pileup_rows": pileup_rows,
        }

    group_sections = []

    # Orient-ref mode: groups carry read_names from the haplotype split.
    # Re-align each group's reads to its own assigned variant FASTA so the
    # pileup reflects the correct reference (E2F1 reads vs E2F1 reference,
    # POU5F1 reads vs POU5F1 reference) instead of the BAM orient reference.
    if bam_path and os.path.exists(bam_path) and any(
        "read_names" in g for g in candidate_info["groups"]
    ):
        for ginfo in candidate_info["groups"]:
            read_names = set(ginfo.get("read_names", []))
            frac = ginfo.get("frac", len(read_names) / max(candidate_info["total_reads"], 1))
            is_recoverable = ginfo["variant"] in recoverable_set
            status = "Clean" if _cigar_is_clean(ginfo.get("cigar")) else "Mutation"

            variant_fasta = os.path.join(single_ref_dir, f"{ginfo['variant']}.fasta")
            if os.path.exists(variant_fasta) and minimap2_path and samtools_path:
                pileup_rows = _build_pileup_from_bam_realign(
                    bam_path, read_names, variant_fasta,
                    minimap2_path, samtools_path,
                )
                ref_seq = str(next(SeqIO.parse(variant_fasta, "fasta")).seq)
            else:
                pileup_rows = []
                ref_seq = ""

            group_sections.append({
                "ref_id": ginfo["variant"],
                "n_reads": len(pileup_rows),
                "frac": frac,
                "status": status,
                "is_recoverable": is_recoverable,
                "ref_seq": ref_seq,
                "pileup_rows": pileup_rows,
            })
    elif any("read_names" in g for g in candidate_info["groups"]):
        # Orient-ref mode but no BAM — fall back to re-alignment
        for ginfo in candidate_info["groups"]:
            read_names = set(ginfo.get("read_names", []))
            grp = well_reads[well_reads["read_name"].isin(read_names)]
            if grp.empty:
                continue
            section = _make_section(ginfo, grp)
            if section is not None:
                group_sections.append(section)
    else:
        # Regular mode: reads assigned to different references — group by ref_name.
        group_lookup = {g["variant"]: g for g in candidate_info["groups"]}
        for ref_id, grp in well_reads.groupby("_ref"):
            ginfo = group_lookup.get(ref_id, {"variant": ref_id, "frac": len(grp) / max(candidate_info["total_reads"], 1)})
            section = _make_section(ginfo, grp)
            if section is not None:
                group_sections.append(section)

    # Sort: major group (highest read fraction) first
    group_sections.sort(key=lambda s: -s["frac"])

    flank_lengths = None
    if flank_5p_len or flank_3p_len:
        flank_lengths = (flank_5p_len, flank_3p_len)
    html = _render_pileup_html(well_pos, candidate_info, group_sections,
                               flank_lengths=flank_lengths)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def _build_pileup_from_bam_realign(
    bam_path: str,
    read_names: set,
    target_fasta: str,
    minimap2_path: str,
    samtools_path: str,
    min_overlap_pos: int = -1,
) -> list:
    """Take a group's reads out of *bam_path* and align them to *target_fasta*.

    In orient-ref mode every read in a well is stored against one orientation
    reference, so a group displayed straight from that BAM mismatches across
    the insert wherever its own variant differs from the orientation reference.
    Recovering the reads as sequences and re-aligning puts each group against
    the variant it was assigned to.

    Returns the same row format as :func:`_build_pileup_grid`.
    """
    ref_record = next(SeqIO.parse(target_fasta, "fasta"), None)
    if ref_record is None:
        return []
    return grid_from_reads(
        reads_from_alignment(bam_path, read_names),
        target_fasta, str(ref_record.seq),
        minimap2=minimap2_path, samtools=samtools_path,
        min_overlap_pos=min_overlap_pos,
    )


def _reads_for_pileup(group_reads: pd.DataFrame) -> list:
    """Turn a group's rows into the neutral read form seqviewer aligns."""
    return [
        Read(name=r["read_name"], seq=r["read_seq"], qual=r.get("read_qual"))
        for _, r in group_reads.iterrows()
    ]


def _build_pileup_grid(
    group_reads: pd.DataFrame,
    ref_fasta: str,
    ref_seq: str,
    minimap2_path: str,
    samtools_path: str,
    ref_index: str = None,
    min_overlap_pos: int = -1,
) -> list:
    """Align a group's reads to its reference and build the display grid.

    Alignment happens here rather than reusing the well's consensus BAM.  The
    pileup is meant to be an independent look at the reads, and one drawn from
    the alignment the variant call came from could not contradict that call.

    Reads that do not cross *min_overlap_pos* are dropped; it defaults to the
    reference midpoint.  Concatemer split-reads cover one flank and stop, so
    they would otherwise fill the grid with rows that say nothing about the
    insert.

    Args:
        group_reads: Rows carrying ``read_name``, ``read_seq``, ``read_qual``.
        ref_fasta: FASTA to align against.
        ref_seq: That FASTA's sequence, which sets the grid's width.
        minimap2_path: minimap2 executable.
        samtools_path: samtools executable.
        ref_index: A prebuilt ``.mmi`` to use in place of *ref_fasta*.
        min_overlap_pos: Reference position a read must cross.  Negative means
            the midpoint; 0 keeps every aligned read.

    Returns:
        One ``(base, is_match)`` row per surviving read, clustered by mismatch
        pattern so subpopulations sit together.
    """
    return grid_from_reads(
        _reads_for_pileup(group_reads), ref_fasta, ref_seq,
        minimap2=minimap2_path, samtools=samtools_path,
        ref_index=ref_index, min_overlap_pos=min_overlap_pos,
    )


def _render_pileup_html(well_pos: str, candidate: dict,
                        groups: list,
                        flank_lengths: tuple = None,
                        features: list = None) -> str:
    """Render one well's pileup page.

    The page is seqviewer's, which is where this renderer now lives; what is
    left here is the mapping from uSort-M's group dicts onto its model.

    *features* are drawn as a track over the reference bar, which is what puts
    the tags either side of the variable region in view -- so a change can be
    read against what it sits next to rather than against a bare coordinate.
    They are stated in the coordinates of the group references, which is what
    :mod:`usortm.demux.annotations` produces.
    """
    if flank_lengths and not (flank_lengths[0] or flank_lengths[1]):
        flank_lengths = None
    ref_len = max((len(g["ref_seq"]) for g in groups), default=0) or None
    view = PileupView(
        title=f"Pileup: Plate {candidate['plate']} Well {candidate['well']}",
        groups=[
            PileupGroup(
                name=g["ref_id"],
                ref_seq=g["ref_seq"],
                rows=g["pileup_rows"],
                n_reads=g["n_reads"],
                fraction=g["frac"],
                status=g["status"],
                highlighted=g["is_recoverable"],
                parent=g.get("parent", ""),
            )
            for g in groups
        ],
        total_reads=candidate.get("total_reads", 0),
        highlight_ids=list(candidate.get("recoverable_variants", [])),
        highlight_label="Recoverable",
        flanks=flank_lengths,
        features=list(features or []),
        ref_len=ref_len,
    )
    return render(view)


def _generate_one_pick_pileup(
    well_pos: str,
    source_plate: str,
    source_well: str,
    variant: str,
    reads: int,
    consensus_fraction: float,
    cons_check: str = "",
    well_reads: pd.DataFrame = None,
    single_ref_dir: str = "",
    output_path: str = "",
    minimap2_path: str = None,
    samtools_path: str = None,
    ref_index: str = None,
    flank_5p_len: int = 0,
    flank_3p_len: int = 0,
    parent_ref_fasta: str = None,
    features: list = None,
) -> Optional[str]:
    """Generate a pileup HTML for one picked well.

    Args:
        parent_ref_fasta: The unmutated construct.  When given, the well's reads
            are shown against it as a second group, so a change reads as a
            column that disagrees rather than as an absence.
        features: Annotations to draw over the reference bar, in the
            coordinates of the group references.

    Returns *output_path* on success, or None if the reference FASTA is
    missing or alignment produces no rows.
    """
    ref_fasta = os.path.join(single_ref_dir, f"{variant}.fasta")
    if not os.path.exists(ref_fasta):
        logger.warning("Pick pileup: ref FASTA not found for %s (%s)", variant, ref_fasta)
        return None

    ref_record = next(SeqIO.parse(ref_fasta, "fasta"), None)
    if ref_record is None:
        return None
    ref_seq = str(ref_record.seq)
    ref_len = len(ref_seq)

    pileup_rows = _build_pileup_grid(
        well_reads, ref_fasta, ref_seq,
        minimap2_path, samtools_path, ref_index=ref_index,
    )

    # Use the number of reads that actually aligned to the variable region
    # (after flank filtering) as the displayed count, not the raw read count
    # which includes concatemer reads that only cover the 5' flank.
    n_variable_reads = len(pileup_rows)

    candidate_info = {
        "plate": source_plate,
        "well": source_well,
        "total_reads": n_variable_reads,
        "top_frac": consensus_fraction,
        "recoverable_variants": [],
        "groups": [{"variant": variant, "frac": consensus_fraction, "status": ""}],
    }

    # The unmutated parent as a row of its own.  The variant reference already
    # carries its designed change, so reads matching it agree everywhere and
    # the page says least exactly where the interest is; against the parent the
    # change is the one column where the two disagree.
    parent = ""
    if parent_ref_fasta and os.path.exists(parent_ref_fasta):
        parent_record = next(SeqIO.parse(parent_ref_fasta, "fasta"), None)
        if parent_record is not None and len(parent_record.seq) == ref_len:
            parent = str(parent_record.seq)

    _display_status = cons_check if cons_check else ""
    _is_recoverable = cons_check in ("Perfect Match", "Silent Mutation")
    group_sections = [{
        "ref_id": variant,
        "n_reads": n_variable_reads,
        "frac": consensus_fraction,
        "status": _display_status,
        "is_recoverable": _is_recoverable,
        "ref_seq": ref_seq,
        "pileup_rows": pileup_rows,
        "parent": parent,
    }]

    flank_lengths = None
    if flank_5p_len or flank_3p_len:
        flank_lengths = (flank_5p_len, flank_3p_len)
    html = _render_pileup_html(well_pos, candidate_info, group_sections,
                               flank_lengths=flank_lengths,
                               features=features)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(html)
    return output_path


def _clear_stale_pileups(pileup_dir: str, keep=None) -> int:
    """Remove pileup pages that this run will not rewrite.

    Args:
        pileup_dir: Directory holding ``well_<plate>_<well>.html`` pages.
        keep: Filenames this run will regenerate; ``None`` clears all.

    Returns:
        How many files were removed.
    """
    removed = 0
    for name in os.listdir(pileup_dir):
        if not (name.startswith("well_") and name.endswith(".html")):
            continue
        if keep is not None and name in keep:
            continue
        try:
            os.remove(os.path.join(pileup_dir, name))
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info("Removed %d pileup page(s) not regenerated by this run",
                    removed)
    return removed


def _build_parent_reference(single_ref_dir: str, out_dir: str) -> Optional[str]:
    """Write the unmutated construct the library was built from, if derivable.

    Every member of a substitution scan differs from the parent at one codon,
    so the parent can be recovered by vote even though it is not itself a
    member.  A library whose members genuinely differ has no such consensus
    and gets no parent group.
    """
    from usortm.demux.protein_call import derive_parent_insert

    fastas = sorted(glob.glob(os.path.join(single_ref_dir, "*.fasta")))
    if len(fastas) < 4:
        return None

    seqs = []
    for path in fastas:
        rec = next(SeqIO.parse(path, "fasta"), None)
        if rec is not None:
            seqs.append(str(rec.seq))

    parent = derive_parent_insert(seqs)
    if parent is None:
        logger.debug(
            "No parent reference: the library's members are not a "
            "single-substitution scan",
        )
        return None

    path = os.path.join(out_dir, ".parent.fasta")
    with open(path, "w") as fh:
        fh.write(f">parent\n{parent}\n")
    return path


def _pileup_features(annotation_file, single_ref_dir: str,
                     tasks: list) -> list:
    """Annotations placed on the reference the pileups are drawn against."""
    if not annotation_file or not tasks:
        return []
    from usortm.demux.annotations import features_for_reference

    ref_fasta = os.path.join(single_ref_dir, f"{tasks[0]['variant']}.fasta")
    rec = next(SeqIO.parse(ref_fasta, "fasta"), None) if os.path.exists(
        ref_fasta) else None
    if rec is None:
        return []
    features = features_for_reference(annotation_file, str(rec.seq))
    if features:
        logger.info("Drawing %d annotations from %s",
                    len(features), annotation_file)
    return features


def _pick_pileup_worker(task: dict) -> bool:
    """Build and write one picked well's pileup page.

    Defined at module level and given only picklable arguments so it can run
    in another process.  Building a grid is better than three quarters Python
    -- one tuple per reference position per read -- so threads serialise it on
    the interpreter lock and more of them buy almost nothing.  Processes do
    not share that lock.

    Returns whether a page was written.
    """
    from usortm.demux.utils import load_well_reads

    well_pos = task["well_pos"]
    records = task.get("reads_records")
    if records is not None:
        well_reads = pd.DataFrame(records)
    else:
        well_reads = load_well_reads(task["well_fastqs_dir"], well_pos)

    if well_reads is None or well_reads.empty:
        logger.debug("No reads found for well %s", well_pos)
        return False

    fname = f"well_{task['source_plate']}_{task['source_well']}.html"
    out_path = os.path.join(task["pileup_dir"], fname)
    result = _generate_one_pick_pileup(
        well_pos=well_pos,
        source_plate=task["source_plate"],
        source_well=task["source_well"],
        variant=task["variant"],
        reads=task["reads"],
        consensus_fraction=task["consensus_fraction"],
        cons_check=task.get("cons_check", ""),
        well_reads=well_reads,
        single_ref_dir=task["single_ref_dir"],
        output_path=out_path,
        minimap2_path=task["minimap2_path"],
        samtools_path=task["samtools_path"],
        ref_index=task["ref_index"],
        flank_5p_len=task["flank_5p_len"],
        flank_3p_len=task["flank_3p_len"],
        parent_ref_fasta=task.get("parent_ref_fasta"),
        features=task.get("features"),
    )
    return result is not None


def _map_pileup_tasks(tasks: list, workers: int):
    """Run *tasks* across processes, yielding ``(succeeded, task)`` as they land.

    Falls back to threads where a process pool cannot start -- a frozen build,
    a sandbox without shared memory -- so the stage still completes, just
    without the speedup.
    """
    from concurrent.futures import ProcessPoolExecutor

    if len(tasks) < 2 or workers < 2:
        for task in tasks:
            try:
                yield _pick_pileup_worker(task), task
            except Exception as exc:
                logger.warning("Pick pileup failed for %s: %s",
                               task["well_pos"], exc)
                yield False, task
        return

    for pool_cls in (ProcessPoolExecutor, ThreadPoolExecutor):
        try:
            with pool_cls(max_workers=workers) as pool:
                futures = {pool.submit(_pick_pileup_worker, t): t
                           for t in tasks}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        yield fut.result(), task
                    except Exception as exc:
                        logger.warning("Pick pileup failed for %s: %s",
                                       task["well_pos"], exc)
                        yield False, task
            return
        except Exception as exc:
            if pool_cls is ThreadPoolExecutor:
                raise
            logger.warning(
                "Could not run pileups across processes (%s); falling back to "
                "threads, which will be slower", exc,
            )


def generate_pick_pileups(
    pick_list: list,
    demux_output_dir: str,
    output_dir: str,
    workers: int = 4,
    minimap2_path: str = None,
    samtools_path: str = None,
    progress_callback=None,
    annotation_file=None,
) -> dict:
    """Generate per-well pileup HTMLs for all picked (non-empty) hits.

    Read identities come from ``read_df.csv`` in *demux_output_dir*; their
    sequences come from the per-well FASTQs alongside it.
    One HTML is written per unique source well to
    ``<output_dir>/pileup/well_{plate}_{well}.html``.

    Args:
        pick_list: List of hit dicts from ``pick._generate_pick_list()``.
            Must include ``source_plate``, ``source_well``, ``variant``,
            ``reads``, ``consensus_fraction`` keys.  Empty placeholder
            entries (``empty=True``) are skipped.
        demux_output_dir: Path to the ``demux_output/`` directory produced
            by ``usortm demux``.
        output_dir: Directory where pileup HTMLs are written
            (``<output_dir>/pileup/``).
        workers: Number of parallel alignment workers.
        minimap2_path: Path to minimap2 binary; auto-detected if None.
        samtools_path: Path to samtools binary; auto-detected if None.

    Returns:
        Nested dict ``{str(target_plate): {target_well: relative_url}}``
        where *relative_url* is relative to *output_dir*
        (e.g. ``"pileup/well_1_A3.html"``).
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    # Load flank lengths from demux summary (present when --vector-fasta was used)
    flank_5p_len = 0
    flank_3p_len = 0
    summary_path = os.path.join(demux_output_dir, "demux_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as _sf:
                _summary = json.load(_sf)
            flank_5p_len = int(_summary.get("flank_5p_len", 0))
            flank_3p_len = int(_summary.get("flank_3p_len", 0))
        except Exception:
            pass

    # Load per-read sequences from demux output
    read_df_path = os.path.join(demux_output_dir, "read_df.csv")
    if not os.path.exists(read_df_path):
        logger.warning("generate_pick_pileups: read_df.csv not found at %s", read_df_path)
        return {}

    # Only older demux outputs keep read sequences in this table; newer ones
    # leave them in the per-well FASTQs.  Reading the header settles which,
    # for the cost of one line rather than the whole file -- which on a real
    # run is gigabytes that the FASTQ path then never looks at.
    header = pd.read_csv(read_df_path, nrows=0)
    has_sequences = "read_seq" in header.columns
    read_df = (pd.read_csv(read_df_path, dtype={"plate": str})
               if has_sequences else None)

    single_ref_dir = os.path.join(demux_output_dir, "reference_fasta", "single_ref_fastas")
    if not os.path.isdir(single_ref_dir):
        logger.warning(
            "generate_pick_pileups: single_ref_fastas not found at %s", single_ref_dir
        )
        return {}

    pileup_dir = os.path.join(output_dir, "pileup")
    os.makedirs(pileup_dir, exist_ok=True)

    # Deduplicate by source well (one hit per unique source plate+well)
    seen: set[tuple] = set()
    tasks: list[dict] = []
    for hit in pick_list:
        if hit.get("empty"):
            continue
        sp = str(hit["source_plate"])
        sw = hit["source_well"]
        if not sp or not sw:
            continue
        key = (sp, sw)
        if key in seen:
            continue
        seen.add(key)
        well_pos = f"{sp}{sw}"
        tasks.append({
            "well_pos": well_pos,
            "source_plate": sp,
            "source_well": sw,
            "variant": hit["variant"],
            "reads": hit["reads"],
            "consensus_fraction": hit["consensus_fraction"],
            "cons_check": hit.get("cons_check", ""),
            "target_plate": str(hit.get("target_plate", "")),
            "target_well": hit.get("target_well", ""),
        })

    # Clear pileups this call will not regenerate.  Each call is authoritative
    # for its directory, and a well rendered by an earlier run but not this one
    # otherwise keeps a page built from whatever reads existed then -- which
    # reads as a well whose depth collapsed rather than as a stale file.
    _clear_stale_pileups(
        pileup_dir,
        keep={f"well_{t['source_plate']}_{t['source_well']}.html" for t in tasks},
    )

    # Build lookup: well_pos → reads DataFrame.
    #
    # read_df.csv carries each read's identity and assignment but not its
    # sequence — that lives in the per-well FASTQs, so pull the reads for the
    # picked wells from there.  Older demux outputs kept the sequences in the
    # CSV, so those are still used when present.
    well_fastqs_dir = os.path.join(demux_output_dir, "wells", "fastqs")
    if has_sequences:
        read_df["well_pos"] = read_df["well_pos"].astype(str)
        grouped = {wp: grp for wp, grp in read_df.groupby("well_pos")}
        # Records rather than frames: each well's reads cross a process
        # boundary, and only the picked wells' reads need to.
        well_reads_map = {
            t["well_pos"]: grouped[t["well_pos"]].to_dict("records")
            for t in tasks if t["well_pos"] in grouped
        }
    else:
        # Left to the workers, which load only their own well.  Loading every
        # picked well here would hold them all in one process, and do it on
        # one core.
        well_reads_map = None

    # Pre-build minimap2 .mmi indexes per unique variant (avoids re-indexing per well)
    unique_variants = {t["variant"] for t in tasks}
    mmi_dir = os.path.join(pileup_dir, ".mmi_cache")
    os.makedirs(mmi_dir, exist_ok=True)
    variant_mmi: dict[str, str] = {}
    for variant in unique_variants:
        ref_fasta = os.path.join(single_ref_dir, f"{variant}.fasta")
        if not os.path.exists(ref_fasta):
            continue
        mmi_path = os.path.join(mmi_dir, f"{variant}.mmi")
        try:
            subprocess.run(
                [minimap2_path, "-d", mmi_path, ref_fasta],
                stderr=subprocess.DEVNULL, check=True,
            )
            variant_mmi[variant] = mmi_path
        except Exception as exc:
            logger.debug("Failed to pre-build index for %s: %s", variant, exc)

    # Result: target_plate → {target_well → relative pileup URL}
    url_map: dict[str, dict[str, str]] = {}

    # Everything a worker needs, packed per task so it can be sent to another
    # process.  The grid itself never comes back: each worker writes its own
    # page and returns only whether it managed to.
    parent_ref_fasta = _build_parent_reference(single_ref_dir, pileup_dir)
    features = _pileup_features(annotation_file, single_ref_dir, tasks)

    for task in tasks:
        task.update(
            parent_ref_fasta=parent_ref_fasta,
            features=features,
            reads_records=(well_reads_map or {}).get(task["well_pos"]),
            well_fastqs_dir=None if has_sequences else well_fastqs_dir,
            single_ref_dir=single_ref_dir,
            pileup_dir=pileup_dir,
            minimap2_path=minimap2_path,
            samtools_path=samtools_path,
            ref_index=variant_mmi.get(task["variant"]),
            flank_5p_len=flank_5p_len,
            flank_3p_len=flank_3p_len,
        )

    for ok, task in _map_pileup_tasks(tasks, workers):
        if progress_callback:
            progress_callback(task["well_pos"], success=ok)
        if not ok:
            continue
        tp = task["target_plate"]
        tw = task["target_well"]
        if tp and tw:
            rel_url = f"pileup/well_{task['source_plate']}_{task['source_well']}.html"
            url_map.setdefault(tp, {})[tw] = rel_url

    # Clean up pre-built indexes
    import shutil
    shutil.rmtree(mmi_dir, ignore_errors=True)

    return url_map
