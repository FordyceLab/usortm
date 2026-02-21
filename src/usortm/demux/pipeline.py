"""End-to-end LevSeq demultiplexing pipeline.

Orchestrates the full workflow: reference alignment (to determine read
direction), Dorado barcode demux on oriented reads, per-well consensus
generation, and variant calling.  Wires together the functions in
utils.py with the barcode config generators in barcodes.py.

Because the first 12 LevSeq forward barcodes (NB01-NB12) differ from the
reverse barcodes, but NB13-NB96 and RB13-RB96 are reverse complements of
each other, read direction **must** be resolved before barcode
demultiplexing.  The pipeline therefore aligns raw reads to a multi-entry
reference library first, splits by strand, and then feeds the
direction-resolved FASTQ to Dorado.
"""

import gzip
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from Bio import SeqIO

from usortm.demux.deps import check_all_dependencies
from usortm.demux.barcodes import (
    get_rbc_count_for_plates,
    write_levseq_fbc_fasta,
    write_levseq_fbc_toml,
    write_levseq_rbc_fasta,
    write_levseq_rbc_toml,
)
from usortm.demux import utils

logger = logging.getLogger(__name__)


def _count_fastq_reads(fastq_path: str) -> int:
    """Count reads in a FASTQ file (4 lines per record)."""
    open_fn = gzip.open if str(fastq_path).endswith(".gz") else open
    n_lines = 0
    with open_fn(fastq_path, "rt") as fh:
        for _ in fh:
            n_lines += 1
    return n_lines // 4


def _extract_reads_gzip_aware(
    input_fastq: str,
    output_fastq: str,
    num_reads: int,
) -> None:
    """Extract the first *num_reads* from a FASTQ file (plain or gzipped)."""
    open_fn = gzip.open if input_fastq.endswith(".gz") else open
    reads_written = 0
    with open_fn(input_fastq, "rt") as fh_in, open(output_fastq, "w") as fh_out:
        while reads_written < num_reads:
            lines = [fh_in.readline() for _ in range(4)]
            if not lines[0]:
                break
            fh_out.writelines(lines)
            reads_written += 1


def run_levseq_pipeline(
    fastq: Path,
    output_dir: Path,
    reference: Optional[Path] = None,
    n_plates: int = 1,
    min_reads: int = 100,
    min_fraction: float = 0.8,
    threads: int = 4,
    progress_callback: Optional[Callable[[str], None]] = None,
    mask_config: Optional[dict] = None,
    subsample: Optional[int] = None,
) -> dict:
    """Run the full LevSeq demultiplexing pipeline.

    Stages:
        1. Check external tool dependencies
        2. Generate Dorado barcode config files (TOML + FASTA)
        3. Multi-ref alignment + strand split (determines read direction)
        4. Dorado FBC demux on oriented reads
        5. Dorado RBC demux on oriented reads
        6. Build merged read DataFrame
        7. Map barcodes to 384-well positions
        8. Generate per-well summary
        9. Generate per-well consensus sequences
        10. Call variants from consensus CIGAR strings
        11. Translate results to CLI output format

    Args:
        fastq: Path to input FASTQ file.
        output_dir: Directory for all pipeline outputs.
        reference: Path to reference FASTA (single or multi-entry).
        n_plates: Number of 384-well plates used.
        min_reads: Minimum reads per well to pass QC.
        min_fraction: Minimum consensus fraction to pass QC.
        threads: Number of threads for alignment.
        progress_callback: Optional function called with stage descriptions.
        mask_config: Optional dict with ``fbc`` and ``rbc`` sub-dicts
            containing mask sequences for Dorado barcode TOML files.
            Falls back to DEFAULT_MASKS if not provided.
        subsample: Optional number of reads to subsample before processing.

    Returns:
        Dict with keys: input_reads, aligned_reads, demuxed_reads,
        assigned_reads, wells_with_data, wells_passing,
        well_assignments. Compatible with demux_cmd._save_demux_results().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str):
        """Report progress if a callback was provided."""
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # --- Stage 1: Check dependencies ---
    _progress("Checking dependencies...")
    tool_paths = check_all_dependencies()

    # --- Stage 2: Generate Dorado barcode config files ---
    _progress("Generating barcode config files...")
    config_dir = output_dir / "dorado_config"
    n_rbc = get_rbc_count_for_plates(n_plates)

    fbc_masks = mask_config.get("fbc") if mask_config else None
    rbc_masks = mask_config.get("rbc") if mask_config else None
    scoring = mask_config.get("scoring") if mask_config else None
    fbc_toml = write_levseq_fbc_toml(config_dir, masks=fbc_masks, scoring=scoring)
    rbc_toml = write_levseq_rbc_toml(config_dir, n_barcodes=n_rbc, masks=rbc_masks, scoring=scoring)
    fbc_fasta = write_levseq_fbc_fasta(config_dir)
    rbc_fasta = write_levseq_rbc_fasta(config_dir, n_barcodes=n_rbc)

    # --- Count input reads ---
    _progress("Counting input reads...")
    input_reads = _count_fastq_reads(str(fastq))
    logger.info("Input FASTQ contains %d reads", input_reads)

    # Pipeline stats accumulator
    pipeline_stats = {"input_reads": input_reads}

    # --- Subsample if requested ---
    if subsample is not None and subsample < input_reads:
        _progress(f"Subsampling to {subsample:,} reads...")
        sub_path = output_dir / "subsampled.fastq"
        _extract_reads_gzip_aware(str(fastq), str(sub_path), subsample)
        fastq = sub_path
        logger.info("Subsampled %d reads to %s", subsample, sub_path)

    # --- Stage 3: Multi-ref alignment + strand split ---
    # This must happen BEFORE barcode demux because NB13-NB96 and
    # RB13-RB96 are reverse complements.  Aligning to the library
    # determines read direction so Dorado sees correct barcode
    # orientation.
    ref_map = None
    oriented_fq = str(fastq)  # default: use raw FASTQ if no reference

    if reference is not None:
        _progress("Aligning reads to reference library...")
        align_dir = output_dir / "alignment"
        oriented_fq, ref_map, align_stats = utils.align_and_split_by_strand(
            multi_ref_fasta=str(reference),
            fastq=str(fastq),
            output_dir=str(align_dir),
            minimap2_path=tool_paths["minimap2"],
            samtools_path=tool_paths["samtools"],
            threads=threads,
        )
        pipeline_stats["align"] = align_stats
        _progress("Strand split complete.")

    # --- Stage 4: Dorado FBC demux (on oriented reads) ---
    _progress("Running forward barcode demultiplexing...")
    fbc_output = output_dir / "fbc"
    fbc_output.mkdir(exist_ok=True)
    utils.demux(
        data=oriented_fq,
        output=str(fbc_output),
        toml=str(fbc_toml),
        barcodes=str(fbc_fasta),
        kit_name="levSeq_bcs_map",
        dorado_path=tool_paths["dorado"],
        output_fastq=True,
        emit_summary=True,
    )

    # --- Stage 5: Dorado RBC demux (on oriented reads) ---
    _progress("Running reverse barcode demultiplexing...")
    rbc_output = output_dir / "rbc"
    rbc_output.mkdir(exist_ok=True)
    utils.demux(
        data=oriented_fq,
        output=str(rbc_output),
        toml=str(rbc_toml),
        barcodes=str(rbc_fasta),
        kit_name="levSeq_bcs_map",
        dorado_path=tool_paths["dorado"],
        output_fastq=True,
        emit_summary=True,
    )

    # --- Stage 6: Build read DataFrame ---
    _progress("Assembling read DataFrame...")
    read_df = utils.create_read_df(
        base_dir=str(output_dir),
        ref_map=ref_map,
        oriented_fastq=oriented_fq,
    )
    pipeline_stats["demux"] = {
        "fbc_classified": read_df.attrs.get("fbc_classified", 0),
        "rbc_classified": read_df.attrs.get("rbc_classified", 0),
        "ref_assigned": read_df.attrs.get("ref_assigned", 0),
        "union_reads": len(read_df),
    }

    # --- Stage 7: Map barcodes to well positions ---
    _progress("Mapping barcodes to well positions...")
    fbc_df, rbc_df = _build_barcode_name_dfs(n_fbc=96, n_rbc=n_rbc)
    pre_filter = len(read_df)
    read_df = utils.format_df(
        read_df,
        fbc_df=fbc_df,
        rbc_df=rbc_df,
        ref_fasta=str(reference) if reference else None,
    )
    pipeline_stats["demux"]["complete_assignments"] = len(read_df)
    pipeline_stats["demux"]["dropped_incomplete"] = pre_filter - len(read_df)

    # --- Stage 8: Per-well summary ---
    _progress("Generating per-well summary...")
    well_df = utils.generate_well_df(read_df)

    # --- Stage 9: Per-well consensus ---
    if reference is not None:
        _progress("Generating per-well consensus sequences...")
        ref_dir = output_dir / "reference_fasta"
        _prepare_single_ref_fastas(reference, ref_dir)

        well_df = utils.generate_per_well_consensus(
            well_df,
            read_df,
            str(output_dir),
            str(ref_dir),
            minimap2_path=tool_paths["minimap2"],
            samtools_path=tool_paths["samtools"],
        )

        # --- Stage 10: Variant calling ---
        _progress("Calling variants from consensus...")
        well_df = utils.extract_matches(well_df)

    _progress("Finalizing results...")

    # Save intermediate DataFrames for debugging / power users
    read_df.to_csv(output_dir / "read_df.csv", index=False)
    well_df.to_csv(output_dir / "well_df.csv", index=False)

    # --- Stage 11: Translate to CLI output format ---
    results = _translate_to_cli_format(
        read_df=read_df,
        well_df=well_df,
        min_reads=min_reads,
        pipeline_stats=pipeline_stats,
    )

    return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_barcode_name_dfs(
    n_fbc: int = 96,
    n_rbc: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build barcode name-mapping DataFrames for utils.format_df().

    Args:
        n_fbc: Number of forward barcodes.
        n_rbc: Number of reverse barcodes.

    Returns:
        Tuple of (fbc_df, rbc_df) with a 'name' column each.
    """
    fbc_df = pd.DataFrame({
        "name": [f"FB{i + 1:02d}" for i in range(n_fbc)]
    })
    rbc_df = pd.DataFrame({
        "name": [f"RB{i + 1:02d}" for i in range(n_rbc)]
    })
    return fbc_df, rbc_df


def _prepare_single_ref_fastas(
    multi_ref_fasta: Path,
    output_dir: Path,
) -> None:
    """Split a multi-entry reference FASTA into individual files.

    Each entry is written to output_dir/single_ref_fastas/<id>.fasta.

    Args:
        multi_ref_fasta: Path to multi-entry (or single-entry) reference FASTA.
        output_dir: Parent directory. Files go into single_ref_fastas/ subdirectory.
    """
    single_dir = Path(output_dir) / "single_ref_fastas"
    single_dir.mkdir(parents=True, exist_ok=True)

    for record in SeqIO.parse(str(multi_ref_fasta), "fasta"):
        out_path = single_dir / f"{record.id}.fasta"
        SeqIO.write([record], str(out_path), "fasta")


def _translate_to_cli_format(
    read_df: pd.DataFrame,
    well_df: pd.DataFrame,
    min_reads: int,
    pipeline_stats: Optional[dict] = None,
) -> dict:
    """Convert pipeline DataFrames to the dict format expected by the CLI.

    The output matches the contract consumed by demux_cmd._save_demux_results(),
    pick.py, and report.py:
        - input_reads: int (total reads in the input FASTQ)
        - aligned_reads: int (reads that mapped to a reference)
        - demuxed_reads: int (reads with both FBC + RBC assignments)
        - assigned_reads: int (reads assigned to a well)
        - wells_with_data: int
        - wells_passing: int
        - well_assignments: dict[str, dict]

    Args:
        read_df: Per-read DataFrame with well_pos column.
        well_df: Per-well summary DataFrame.
        min_reads: Minimum read depth to consider a well "passing".
        pipeline_stats: Optional dict of per-stage counts collected during
            the pipeline run.

    Returns:
        Results dict compatible with CLI save/display functions.
    """
    stats = pipeline_stats or {}

    input_reads = stats.get("input_reads", 0)

    align = stats.get("align", {})
    aligned_reads = align.get("mapped", 0)

    demux = stats.get("demux", {})
    demuxed_reads = demux.get("complete_assignments", len(read_df))

    # Count reads that were successfully assigned to a well
    if "well_pos" in read_df.columns:
        assigned_reads = int(read_df["well_pos"].notna().sum())
    else:
        assigned_reads = 0

    wells_with_data = len(well_df)
    wells_passing = int((well_df["depth"] >= min_reads).sum())

    # Build per-well assignment dict
    well_assignments = {}
    for _, row in well_df.iterrows():
        plate = str(int(row["plate"]))
        well = str(row["well"])
        key = f"{plate}_{well}"

        # Extract variant name from major_ref
        variant = str(row.get("major_ref", "unknown"))
        if ":" in variant:
            variant = variant.split(":")[-1]

        # Append consensus check status if available
        cons_check = row.get("cons_check", "")
        if cons_check and cons_check != "Error":
            variant = f"{variant}|{cons_check}"

        well_assignments[key] = {
            "plate": plate,
            "well": well,
            "reads": int(row["depth"]),
            "variant": variant,
            "consensus_fraction": float(row.get("major_freq", 0.0)),
        }

    return {
        "input_reads": input_reads,
        "aligned_reads": aligned_reads,
        "demuxed_reads": demuxed_reads,
        "assigned_reads": assigned_reads,
        "wells_with_data": wells_with_data,
        "wells_passing": wells_passing,
        "well_assignments": well_assignments,
        # Keep total_reads as alias for backward compat
        "total_reads": input_reads,
    }
