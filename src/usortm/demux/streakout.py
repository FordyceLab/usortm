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
    minimap2_path: str,
    samtools_path: str,
    min_well_reads: int = 50,
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
    ref_records = list(SeqIO.parse(reference_fasta, "fasta"))
    if not ref_records:
        return []

    ref_ids = [r.id for r in ref_records]
    ref_seqs_str = [str(r.seq).upper() for r in ref_records]
    max_ref_len = max(len(s) for s in ref_seqs_str)
    ref_matrix = np.zeros((len(ref_ids), max_ref_len), dtype=np.uint8)
    for i, seq in enumerate(ref_seqs_str):
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        ref_matrix[i, : len(arr)] = arr

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")
    well_bam_dir = os.path.join(output_dir, "wells", "consensus")

    candidate_wells = well_df[well_df["depth"] >= min_well_reads]
    if candidate_wells.empty:
        return []

    logger.info(
        "Screening %d wells for streak-out candidates (orient-ref mode, depth >= %d)",
        len(candidate_wells), min_well_reads,
    )

    from tqdm import tqdm

    def _process(row):
        wp = row["global_well"]
        bam_path = os.path.join(well_bam_dir, f"{wp}.bam")
        if not os.path.exists(bam_path):
            return None

        bimodal = _find_bimodal_positions(bam_path, samtools_path)
        if not bimodal:
            return None

        well_reads = read_df[read_df["well_pos"] == wp]
        if well_reads.empty:
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
                _cigar, cons_seq = _group_consensus(
                    fq_path, orient_fa, tmp, minimap2_path, samtools_path,
                )

            ref_id, ref_seq, ref_len, status = _assign_and_classify_group(
                cons_seq, ref_ids, ref_seqs_str, ref_matrix,
            )
            if ref_id is None:
                continue

            frac = len(group_df) / depth
            group_results.append({
                "variant": ref_id,
                "reads": len(group_df),
                "frac": round(frac, 4),
                "status": status,
                "is_major": (ref_id == orient_ref_name),
            })

        correct_groups = [g for g in group_results if _is_correct(g["status"])]
        if len(correct_groups) < 2:
            return None

        recoverable = [g["variant"] for g in correct_groups if not g["is_major"]]
        top_frac = max((g["frac"] for g in group_results), default=1.0)
        return {
            "plate": plate,
            "well": well,
            "global_well": wp,
            "total_reads": depth,
            "top_frac": round(top_frac, 4),
            "groups": sorted(group_results, key=lambda g: -g["frac"]),
            "recoverable_variants": recoverable,
        }

    candidates = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, row): row["global_well"]
            for _, row in candidate_wells.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
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
            "is_major": (frac >= top_frac - 0.01),
        })

    # Check: 2+ groups with correct consensus?
    correct_groups = [g for g in group_results if _is_correct(g["status"])]
    if len(correct_groups) < 2:
        return None

    # Identify recoverable (minority) variants
    recoverable = [
        g["variant"] for g in correct_groups if not g["is_major"]
    ]

    return {
        "plate": plate,
        "well": well,
        "global_well": well_pos,
        "total_reads": depth,
        "top_frac": round(top_frac, 4),
        "groups": sorted(group_results, key=lambda g: -g["frac"]),
        "recoverable_variants": recoverable,
    }


def detect_streakout_candidates(
    well_df: pd.DataFrame,
    read_df: pd.DataFrame,
    reference_dir: str,
    output_dir: str,
    minimap2_path: str = None,
    samtools_path: str = None,
    min_well_reads: int = 50,
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
        for _, row in candidate_wells.iterrows():
            wp = row["global_well"]
            well_reads = read_df[read_df["well_pos"] == wp]
            if well_reads.empty:
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
            "n_groups", "recoverable_variants",
        ])
        for c in candidates:
            writer.writerow([
                c["plate"],
                c["well"],
                c["total_reads"],
                c["top_frac"],
                len(c["groups"]),
                ";".join(c["recoverable_variants"]),
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
) -> None:
    """Generate an interactive pileup HTML for one streak-out candidate well.

    Aligns reads from each reference group to their respective reference,
    builds a character grid showing match/mismatch coloring, and writes
    a self-contained HTML file.

    Args:
        well_pos: Global well identifier (e.g. "1A3").
        read_df: Full per-read DataFrame.
        reference_dir: Directory containing ``single_ref_fastas/``.
        candidate_info: Candidate dict from :func:`detect_streakout_candidates`.
        output_path: Path to write the HTML file.
        minimap2_path: Path to minimap2. Auto-detected if None.
        samtools_path: Path to samtools. Auto-detected if None.
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

    # Build group info lookup from candidate_info
    group_lookup = {g["variant"]: g for g in candidate_info["groups"]}

    group_sections = []
    for ref_id, grp in well_reads.groupby("_ref"):
        ref_fasta = os.path.join(single_ref_dir, f"{ref_id}.fasta")
        if not os.path.exists(ref_fasta):
            continue

        ref_record = next(SeqIO.parse(ref_fasta, "fasta"))
        ref_seq = str(ref_record.seq)
        ref_len = len(ref_seq)

        ginfo = group_lookup.get(ref_id, {})
        n_reads = len(grp)
        frac = ginfo.get("frac", n_reads / candidate_info["total_reads"])
        status = ginfo.get("status", "")
        is_recoverable = ref_id in candidate_info["recoverable_variants"]

        # Align to reference and parse BAM for pileup
        pileup_rows = _build_pileup_grid(
            grp, ref_fasta, ref_seq, ref_len,
            minimap2_path, samtools_path,
        )

        group_sections.append({
            "ref_id": ref_id,
            "n_reads": n_reads,
            "frac": frac,
            "status": status,
            "is_recoverable": is_recoverable,
            "ref_seq": ref_seq,
            "pileup_rows": pileup_rows,
        })

    # Sort: major group first
    group_sections.sort(key=lambda s: -s["frac"])

    html = _render_pileup_html(well_pos, candidate_info, group_sections)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def _build_pileup_grid(
    group_reads: pd.DataFrame,
    ref_fasta: str,
    ref_seq: str,
    ref_len: int,
    minimap2_path: str,
    samtools_path: str,
    ref_index: str = None,
) -> list[list[tuple[str, bool]]]:
    """Align group reads and build a character grid for pileup display.

    Returns a list of rows, where each row is a list of
    (base_char, is_match) tuples indexed by reference position.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fq_path = os.path.join(tmp, "reads.fastq")
        bam_path = os.path.join(tmp, "aligned.bam")

        # Write FASTQ
        with open(fq_path, "w") as fq:
            for _, r in group_reads.iterrows():
                fq.write(f"@{r['read_name']}\n{r['read_seq']}\n+\n{r['read_qual']}\n")

        # Align (use pre-built .mmi index if available)
        mm2_ref = ref_index if ref_index else ref_fasta
        try:
            mm2 = subprocess.Popen(
                [minimap2_path, "-a", "--MD", "--secondary=no", mm2_ref, fq_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [samtools_path, "sort", "-o", bam_path],
                stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
            )
            mm2.wait()
            subprocess.run(
                [samtools_path, "index", bam_path],
                stderr=subprocess.DEVNULL, check=False,
            )
        except Exception as exc:
            logger.warning("Pileup alignment failed: %s", exc)
            return []

        # Parse BAM for pileup
        rows = []
        try:
            with pysam.AlignmentFile(bam_path, "rb") as bf:
                for read in bf:
                    if read.is_unmapped:
                        continue
                    row = [("-", True)] * ref_len  # default: gap
                    pairs = read.get_aligned_pairs(with_seq=True)
                    for qpos, rpos, rbase in pairs:
                        if rpos is None or rpos >= ref_len:
                            continue
                        if qpos is None:
                            row[rpos] = ("-", True)  # deletion
                        else:
                            qbase = read.query_sequence[qpos]
                            is_match = qbase.upper() == ref_seq[rpos].upper()
                            row[rpos] = (qbase, is_match)
                    rows.append(row)
        except Exception as exc:
            logger.warning("BAM parsing failed: %s", exc)

        if not rows:
            logger.warning(
                "Pileup produced 0 aligned rows for %d reads against %s",
                len(group_reads), ref_fasta,
            )

    return rows


def _render_pileup_html(well_pos: str, candidate: dict,
                         groups: list[dict]) -> str:
    """Render the pileup HTML page for one well.

    Uses an HTML5 canvas matrix: each read is a row of colored cells.
    Green = match, per-base color = mismatch, light gray = gap.
    """
    import html as _html
    import json as _json

    plate = candidate["plate"]
    well = candidate["well"]
    title = f"Pileup: Plate {plate} Well {well}"

    sections_html = []
    for idx, g in enumerate(groups):
        star = " &#9733;" if g["is_recoverable"] else ""
        status_class = "status-correct" if _is_correct(g["status"]) else "status-other"

        # Compute per-read identity from pileup data
        identity_str = ""
        if g["pileup_rows"]:
            total_bases = 0
            total_matches = 0
            for row in g["pileup_rows"]:
                aligned = [(b, m) for b, m in row if b != "-"]
                total_bases += len(aligned)
                total_matches += sum(1 for _, m in aligned if m)
            if total_bases > 0:
                identity = total_matches / total_bases
                identity_str = f" &middot; Read identity: {identity:.1%}"

        ref_len = len(g["ref_seq"])

        header = (
            f'<div class="group-header">'
            f'<span class="ref-name">{_html.escape(g["ref_id"])}{star}</span>'
            f'<span class="group-meta">'
            f'{g["n_reads"]} reads ({g["frac"]:.0%}) &middot; '
            f'{ref_len} bp &middot; '
            f'Consensus: <span class="{status_class}">'
            f'{_html.escape(g["status"])}</span>'
            f'{identity_str}'
            f'</span></div>'
        )

        # Encode pileup data compactly for JS:
        # '.' = match, base letter = mismatch, '-' = gap
        rows_encoded = []
        for row in g["pileup_rows"]:
            chars = []
            for base_char, is_match in row:
                if base_char == "-":
                    chars.append("-")
                elif is_match:
                    chars.append(".")
                else:
                    chars.append(base_char.upper())
            rows_encoded.append("".join(chars))

        # Build consensus from pileup: majority base at each position
        from collections import Counter
        consensus_encoded = []
        ref_seq = g["ref_seq"]
        for col_idx in range(ref_len):
            counts = Counter()
            for row in g["pileup_rows"]:
                base, _ = row[col_idx]
                if base != "-":
                    counts[base.upper()] += 1
            if counts:
                cons_base = counts.most_common(1)[0][0]
                if cons_base == ref_seq[col_idx].upper():
                    consensus_encoded.append(".")
                else:
                    consensus_encoded.append(cons_base)
            else:
                consensus_encoded.append("-")
        consensus_str = "".join(consensus_encoded)

        ref_seq_js = _json.dumps(ref_seq)
        rows_js = _json.dumps(rows_encoded)
        cons_js = _json.dumps(consensus_str)
        n_rows = len(rows_encoded)
        n_cols = ref_len

        if n_rows == 0:
            pileup_block = (
                f'<div class="pileup-empty">'
                f'No aligned reads available ({g["n_reads"]} reads unaligned)'
                f'</div>'
            )
        else:
            pileup_block = (
                f'<div class="pileup-container">'
                f'<div class="pileup-scroll" id="scroll-{idx}">'
                f'<canvas id="pileup-{idx}"></canvas>'
                f'</div>'
                f'<div class="pileup-info">{n_rows} aligned reads &times; '
                f'{n_cols} bp</div>'
                f'</div>'
                f'<script>'
                f'(function(){{'
                f'var ref={ref_seq_js};'
                f'var cons={cons_js};'
                f'var rows={rows_js};'
                f'drawPileup("pileup-{idx}",ref,cons,rows);'
                f'}})();'
                f'</script>'
            )

        sections_html.append(f'{header}\n{pileup_block}')

    body = '\n<hr class="group-sep">\n'.join(sections_html)

    recoverable_list = ", ".join(candidate["recoverable_variants"])
    recoverable_line = (
        f' &middot; Recoverable: {_html.escape(recoverable_list)}'
        if recoverable_list else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<style id="usortm-theme-bridge">
:root {{
    --usortm-bg: #fafafa;
    --text: #1e293b;
    --muted: #94a3b8;
    --card-bg: #ffffff;
    --border: #e5e7eb;
}}
[data-theme="dark"] {{
    --usortm-bg: #1a1a2e;
    --text: #e0e0e0;
    --muted: #64748b;
    --card-bg: #16213e;
    --border: #334155;
}}
html, body {{
    background: var(--usortm-bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 1.5rem;
}}
h1 {{
    font-size: 1.4rem;
    margin: 0 0 0.25rem;
}}
.well-meta {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}}
.group-header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 1rem 0 0.5rem;
}}
.ref-name {{
    font-weight: 700;
    font-size: 1.05rem;
}}
.group-meta {{
    color: var(--muted);
    font-size: 0.85rem;
}}
.status-correct {{
    color: #059669;
    font-weight: 600;
}}
.status-other {{
    color: #ef4444;
}}
.pileup-container {{
    margin-bottom: 0.5rem;
}}
.pileup-scroll {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 60vh;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0;
}}
.pileup-scroll canvas {{
    display: block;
}}
.pileup-info {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.25rem;
}}
.pileup-empty {{
    font-size: 0.85rem;
    color: var(--muted);
    font-style: italic;
    padding: 1rem;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
}}
.legend {{
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.3rem;
}}
.legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid var(--border);
}}
.group-sep {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}}
</style>
<script>
function drawPileup(canvasId, refSeq, cons, rows) {{
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var nCols = refSeq.length;
  var nRows = rows.length;
  var cellW = nCols < 200 ? 4 : nCols < 500 ? 3 : 2;
  var cellH = nRows < 100 ? 3 : 2;
  var refH = Math.max(cellH, 6);
  var consH = refH;
  var gap = 1;
  canvas.width = nCols * cellW;
  canvas.height = refH + gap + consH + gap + nRows * cellH;
  var ctx = canvas.getContext('2d');
  var matchColor = '#059669';
  var gapColor = '#d1d5db';
  var refColor = '#1e293b';
  var consMatchColor = '#059669';
  var baseColors = {{'A':'#ef4444','T':'#3b82f6','C':'#f59e0b','G':'#8b5cf6'}};
  if (document.documentElement.getAttribute('data-theme') === 'dark') {{
    matchColor = '#34d399';
    gapColor = '#334155';
    refColor = '#e0e0e0';
    consMatchColor = '#34d399';
    baseColors = {{'A':'#f87171','T':'#60a5fa','C':'#fbbf24','G':'#a78bfa'}};
  }}
  // Reference row
  ctx.fillStyle = refColor;
  for (var i = 0; i < nCols; i++) {{
    ctx.fillRect(i * cellW, 0, cellW, refH);
  }}
  // Consensus row
  var consY = refH + gap;
  for (var i = 0; i < cons.length; i++) {{
    var ch = cons[i];
    if (ch === '.') {{
      ctx.fillStyle = consMatchColor;
    }} else if (ch === '-') {{
      ctx.fillStyle = gapColor;
    }} else {{
      ctx.fillStyle = baseColors[ch] || '#94a3b8';
    }}
    ctx.fillRect(i * cellW, consY, cellW, consH);
  }}
  // Read rows
  var readsY = consY + consH + gap;
  for (var r = 0; r < nRows; r++) {{
    var row = rows[r];
    var y = readsY + r * cellH;
    for (var c = 0; c < row.length; c++) {{
      var ch = row[c];
      if (ch === '.') {{
        ctx.fillStyle = matchColor;
      }} else if (ch === '-') {{
        ctx.fillStyle = gapColor;
      }} else {{
        ctx.fillStyle = baseColors[ch] || '#94a3b8';
      }}
      ctx.fillRect(c * cellW, y, cellW, cellH);
    }}
  }}
  // Tooltip
  var tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;background:#1e293b;color:#fff;padding:4px 8px;'
    + 'border-radius:4px;font-size:11px;pointer-events:none;display:none;z-index:10;'
    + 'font-family:SF Mono,Menlo,Consolas,monospace;';
  document.body.appendChild(tooltip);
  canvas.addEventListener('mousemove', function(e) {{
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var yp = e.clientY - rect.top;
    var col = Math.floor(x / cellW);
    if (col < 0 || col >= nCols) {{ tooltip.style.display = 'none'; return; }}
    if (yp < refH) {{
      tooltip.textContent = 'Ref pos ' + (col + 1) + ': ' + refSeq[col];
    }} else if (yp < consY + consH) {{
      var ch = cons[col];
      var base = ch === '.' ? refSeq[col] : ch;
      var note = ch === '.' ? ' (match)' : ch === '-' ? '' : ' (mismatch)';
      tooltip.textContent = 'Consensus pos ' + (col + 1) + ': ' + base + note;
    }} else {{
      var row_idx = Math.floor((yp - readsY) / cellH);
      if (row_idx >= 0 && row_idx < nRows) {{
        var ch = rows[row_idx][col];
        var label = ch === '.' ? refSeq[col] + ' (match)' : ch === '-' ? 'gap' : ch + ' (mismatch)';
        tooltip.textContent = 'Read ' + (row_idx + 1) + ', pos ' + (col + 1) + ': ' + label;
      }} else {{
        tooltip.style.display = 'none'; return;
      }}
    }}
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 24) + 'px';
  }});
  canvas.addEventListener('mouseleave', function() {{
    tooltip.style.display = 'none';
  }});
}}
</script>
</head>
<body>
<h1>{_html.escape(title)}</h1>
<div class="well-meta">
    {candidate["total_reads"]} total reads &middot;
    Top fraction: {candidate["top_frac"]:.0%}{recoverable_line}
</div>
<div class="legend">
    <span style="font-weight:600;">Legend:</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#059669;"></span> Match</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ef4444;"></span> A</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#3b82f6;"></span> T</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#f59e0b;"></span> C</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#8b5cf6;"></span> G</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#d1d5db;"></span> Gap</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#1e293b;"></span> Reference</span>
    <span style="color:var(--muted);">|</span>
    <span style="font-size:0.75rem;color:var(--muted);">Rows: Reference &rarr; Consensus &rarr; Reads</span>
</div>
{body}
<script id="usortm-theme-sync">
(function () {{
  try {{
    var stored = localStorage.getItem('usortm-theme');
    if (stored === 'dark') {{
      document.documentElement.setAttribute('data-theme', 'dark');
    }}
  }} catch (e) {{}}
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Pick pileup generation
# ---------------------------------------------------------------------------

def _generate_one_pick_pileup(
    well_pos: str,
    source_plate: str,
    source_well: str,
    variant: str,
    reads: int,
    consensus_fraction: float,
    well_reads: pd.DataFrame,
    single_ref_dir: str,
    output_path: str,
    minimap2_path: str,
    samtools_path: str,
    ref_index: str = None,
) -> Optional[str]:
    """Generate a pileup HTML for one picked well.

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
        well_reads, ref_fasta, ref_seq, ref_len,
        minimap2_path, samtools_path, ref_index=ref_index,
    )

    candidate_info = {
        "plate": source_plate,
        "well": source_well,
        "total_reads": reads,
        "top_frac": consensus_fraction,
        "recoverable_variants": [],
        "groups": [{"variant": variant, "frac": consensus_fraction, "status": ""}],
    }

    group_sections = [{
        "ref_id": variant,
        "n_reads": reads,
        "frac": consensus_fraction,
        "status": "",
        "is_recoverable": False,
        "ref_seq": ref_seq,
        "pileup_rows": pileup_rows,
    }]

    html = _render_pileup_html(well_pos, candidate_info, group_sections)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(html)
    return output_path


def generate_pick_pileups(
    pick_list: list,
    demux_output_dir: str,
    output_dir: str,
    workers: int = 4,
    minimap2_path: str = None,
    samtools_path: str = None,
    progress_callback=None,
) -> dict:
    """Generate per-well pileup HTMLs for all picked (non-empty) hits.

    Reads are sourced from ``read_df.csv`` in *demux_output_dir*.
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

    # Load per-read sequences from demux output
    read_df_path = os.path.join(demux_output_dir, "read_df.csv")
    if not os.path.exists(read_df_path):
        logger.warning("generate_pick_pileups: read_df.csv not found at %s", read_df_path)
        return {}

    read_df = pd.read_csv(read_df_path, dtype={"plate": str})

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
            "target_plate": str(hit.get("target_plate", "")),
            "target_well": hit.get("target_well", ""),
        })

    # Build lookup: well_pos → reads DataFrame
    read_df["well_pos"] = read_df["well_pos"].astype(str)
    well_reads_map = {wp: grp for wp, grp in read_df.groupby("well_pos")}

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

    def _run(task: dict):
        well_pos = task["well_pos"]
        well_reads = well_reads_map.get(well_pos, pd.DataFrame())
        if well_reads.empty:
            logger.debug("No reads found for well %s in read_df", well_pos)
            return None, task

        fname = f"well_{task['source_plate']}_{task['source_well']}.html"
        out_path = os.path.join(pileup_dir, fname)
        result = _generate_one_pick_pileup(
            well_pos=well_pos,
            source_plate=task["source_plate"],
            source_well=task["source_well"],
            variant=task["variant"],
            reads=task["reads"],
            consensus_fraction=task["consensus_fraction"],
            well_reads=well_reads,
            single_ref_dir=single_ref_dir,
            output_path=out_path,
            minimap2_path=minimap2_path,
            samtools_path=samtools_path,
            ref_index=variant_mmi.get(task["variant"]),
        )
        return result, task

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result, task = fut.result()
            except Exception as exc:
                logger.warning("Pick pileup failed for %s: %s", futures[fut]["well_pos"], exc)
                if progress_callback:
                    progress_callback(futures[fut]["well_pos"], success=False)
                continue
            if progress_callback:
                progress_callback(task["well_pos"], success=result is not None)
            if result is None:
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
