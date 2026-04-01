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
from typing import Callable, Optional, Tuple

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


def _open_fastq(fastq_path: str):
    """Return the right open function for a FASTQ file.

    Detects gzip by magic bytes so files without a .gz extension
    (e.g. downloaded via a URL that omits the extension) are handled correctly.
    """
    with open(fastq_path, "rb") as f:
        magic = f.read(2)
    return gzip.open if magic == b'\x1f\x8b' else open


def _count_fastq_reads(fastq_path: str) -> int:
    """Count reads in a FASTQ file (4 lines per record)."""
    open_fn = _open_fastq(fastq_path)
    n_lines = 0
    with open_fn(fastq_path, "rt") as fh:
        for _ in fh:
            n_lines += 1
    return n_lines // 4


def _compute_read_length_hist(fastq_path: str) -> dict:
    """Return a read-length histogram dict for embedding in demux_summary.json.

    Single-pass over the FASTQ sequence lines (line index % 4 == 1).

    Returns:
        Dict with keys bin_size, counts (50 ints), median (int), n_reads (int),
        or empty dict if the file has no reads.
    """
    import statistics

    open_fn = _open_fastq(fastq_path)
    lengths = []
    try:
        with open_fn(fastq_path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % 4 == 1:
                    lengths.append(len(line.rstrip()))
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError(
            f"Cannot read FASTQ file '{fastq_path}' as text. "
            "The file may be in raw nanopore format (pod5/fast5) rather than "
            "basecalled FASTQ. Check the download URL from your sequencing provider."
        ) from exc
    if not lengths:
        return {}
    max_len = max(lengths)
    bin_size = max(1, (max_len + 49) // 50)
    bins = [0] * 50
    for ln in lengths:
        bins[min(ln // bin_size, 49)] += 1
    return {
        "bin_size": bin_size,
        "counts": bins,
        "median": int(statistics.median(lengths)),
        "n_reads": len(lengths),
    }


def _extract_reads_gzip_aware(
    input_fastq: str,
    output_fastq: str,
    num_reads: int,
) -> int:
    """Extract the first *num_reads* from a FASTQ file (plain or gzipped).

    Returns the number of reads actually written (may be less than
    *num_reads* if the input file is shorter).
    """
    open_fn = _open_fastq(input_fastq)
    reads_written = 0
    with open_fn(input_fastq, "rt") as fh_in, open(output_fastq, "w") as fh_out:
        while reads_written < num_reads:
            lines = [fh_in.readline() for _ in range(4)]
            if not lines[0]:
                break
            fh_out.writelines(lines)
            reads_written += 1
    return reads_written


def run_levseq_pipeline(
    fastq: Path,
    output_dir: Path,
    reference: Optional[Path] = None,
    n_plates: int = 1,
    min_reads: int = 100,
    min_fraction: float = 0.8,
    threads: int = 4,
    workers: int = 4,
    progress_callback: Optional[Callable[[str], None]] = None,
    mask_config: Optional[dict] = None,
    subsample: Optional[int] = None,
    orient_ref: Optional[Path] = None,
    vector_fasta: Optional[Path] = None,
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
        10.5. Screen for streak-out candidates (mixed wells with correct subpops)
        11. Translate results to CLI output format

    Args:
        fastq: Path to input FASTQ file.
        output_dir: Directory for all pipeline outputs.
        reference: Path to reference FASTA (single or multi-entry).
        n_plates: Number of 384-well plates used.
        min_reads: Minimum reads per well to pass QC.
        min_fraction: Minimum consensus fraction to pass QC.
        threads: Number of threads for alignment.
        workers: Number of parallel workers for per-well consensus.
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

    # Pipeline stats accumulator
    pipeline_stats = {}

    # --- Subsample if requested ---
    if subsample is not None:
        _progress(f"Subsampling to {subsample:,} reads...")
        sub_path = output_dir / "subsampled.fastq"
        n_extracted = _extract_reads_gzip_aware(str(fastq), str(sub_path), subsample)
        fastq = sub_path
        pipeline_stats["input_reads"] = n_extracted
        logger.info("Subsampled %d reads to %s", n_extracted, sub_path)

    # --- Read length histogram (also provides read count) ---
    _progress("Computing read length histogram...")
    read_len_hist = _compute_read_length_hist(str(fastq))
    if read_len_hist:
        pipeline_stats["read_len_hist"] = read_len_hist
        if "input_reads" not in pipeline_stats:
            pipeline_stats["input_reads"] = read_len_hist.get("n_reads", 0)
    input_reads = pipeline_stats.get("input_reads", 0)

    # --- Parse vector FASTA early (needed for Stage 3 auto-orient and Stage 9) ---
    flank_5p = None
    flank_3p = None
    frame_offset = 0
    if vector_fasta is not None:
        flank_5p, flank_3p = utils.parse_vector_fasta(str(vector_fasta))
        # First ATG in the 5' flank is treated as the start codon.
        # frame_offset = how many bases into the variable region the next codon
        # boundary falls, so translation stays in-frame.
        first_atg = flank_5p.upper().find("ATG")
        if first_atg >= 0:
            frame_offset = (len(flank_5p) - first_atg) % 3

    # --- Stage 3: Multi-ref alignment + strand split ---
    # This must happen BEFORE barcode demux because NB13-NB96 and
    # RB13-RB96 are reverse complements.  Aligning to the library
    # determines read direction so Dorado sees correct barcode
    # orientation.
    ref_map = None
    oriented_fq = str(fastq)  # default: use raw FASTQ if no reference

    # Auto-orient against vector backbone when --vector-fasta is provided
    # and no explicit --orient-ref was given.
    if orient_ref is None and vector_fasta is not None and reference is not None:
        _progress("Building orientation reference from vector backbone...")
        orient_dir = output_dir / "alignment"
        orient_dir.mkdir(parents=True, exist_ok=True)
        auto_orient = orient_dir / "vector_orient_ref.fasta"
        _build_orient_ref_from_flanks(flank_5p, flank_3p, auto_orient)
        orient_ref = auto_orient

    if reference is not None:
        # When --orient-ref is provided (or auto-generated from vector),
        # align against that single reference for fast orientation.
        # Otherwise fall back to the full multi-ref library.
        align_ref = str(orient_ref) if orient_ref is not None else str(reference)
        if orient_ref is not None:
            _progress("Orienting reads against single reference...")
        else:
            _progress("Aligning reads to reference library...")
        align_dir = output_dir / "alignment"

        def _align_progress(n_done, total):
            if n_done is None:
                _progress("Aligning reads to reference library... (cached)")
            elif total:
                pct = int(100 * n_done / total)
                _progress(f"Aligning reads to reference library... {n_done:,}/{total:,} ({pct}%)")
            else:
                _progress(f"Aligning reads to reference library... {n_done:,} aligned")

        oriented_fq, ref_map, align_stats = utils.align_and_split_by_strand(
            multi_ref_fasta=align_ref,
            fastq=str(fastq),
            output_dir=str(align_dir),
            minimap2_path=tool_paths["minimap2"],
            samtools_path=tool_paths["samtools"],
            threads=threads,
            progress_callback=_align_progress,
            total_reads=input_reads,
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
        orient_ref_fasta=str(orient_ref) if orient_ref else None,
    )
    pipeline_stats["demux"]["complete_assignments"] = len(read_df)
    pipeline_stats["demux"]["dropped_incomplete"] = pre_filter - len(read_df)

    # --- Stage 8: Per-well summary ---
    _progress("Generating per-well summary...")
    well_df = utils.generate_well_df(read_df)

    # --- Stage 9: Per-well consensus ---
    if reference is not None:
        ref_dir = output_dir / "reference_fasta"
        if vector_fasta is not None:
            _prepare_full_length_ref_fastas(reference, ref_dir, flank_5p, flank_3p)
        else:
            _prepare_single_ref_fastas(reference, ref_dir)

        if orient_ref is not None:
            # --- Orient-ref / vector-fasta mode ---
            # The orient_ref (N-spacer) is fine for read orientation, but
            # consensus must be generated against real per-variant references.
            #
            # Flow:
            #   1. Filter out concatemer reads (too short to reach variable region)
            #   2. Write per-well FASTQs
            #   3. Assign variants by aligning reads to the library
            #   4. Generate consensus against per-variant full-length refs

            # Drop reads that cannot reach the variable region.  Concatemer
            # split-reads (~150–330 bp) cover only the 5' flank and produce
            # blank rows in pileups and inflate the well read count.  We
            # require reads to be at least half the minimum amplicon length
            # (flank_5p + flank_3p), which for LP014 is (119+1005)//2 = 562 bp.
            # This cleanly separates concatemer reads from full-length reads
            # regardless of variable-insert size.
            _min_read_len = (len(flank_5p) + len(flank_3p)) // 2
            n_before = len(read_df)
            read_df = read_df[
                read_df["read_seq"].str.len() >= _min_read_len
            ].reset_index(drop=True)
            n_removed = n_before - len(read_df)
            if n_removed:
                _progress(
                    f"Filtered {n_removed:,} flank-only reads "
                    f"(<{_min_read_len} bp) that cannot overlap the variable region"
                )
                # Update per-well depths to reflect only variable-spanning reads
                _depth = read_df.groupby("well_pos").size()
                well_df["depth"] = (
                    well_df["global_well"].map(_depth).fillna(0).astype(int)
                )

            _progress("Writing per-well FASTQs...")
            utils.write_per_well_fastqs(read_df, str(output_dir))

            _progress("Assigning variants from read alignments...")
            well_fastqs_dir = str(output_dir / "wells" / "fastqs")
            well_df = utils.assign_variants_from_reads(
                well_df, read_df, str(reference),
                well_fastqs_dir=well_fastqs_dir,
                minimap2_path=tool_paths["minimap2"],
                workers=workers,
                full_length_ref_dir=str(ref_dir / "single_ref_fastas"),
            )

            _progress("Generating consensus against assigned references...")
            well_df = utils.generate_per_well_consensus(
                well_df,
                read_df,
                str(output_dir),
                str(ref_dir),
                minimap2_path=tool_paths["minimap2"],
                samtools_path=tool_paths["samtools"],
                workers=workers,
            )

            # Backfill read_df ref_name from well_df so plate map shows
            # the reassigned variant instead of the orient-ref name.
            well_to_ref = dict(zip(
                well_df["global_well"],
                well_df["major_ref"],
            ))
            if "well_pos" in read_df.columns:
                read_df["ref_name"] = read_df["well_pos"].map(
                    lambda w: f"fwd:{well_to_ref[w]}" if w in well_to_ref else None
                )
                if "ref_id" in read_df.columns:
                    read_df["ref_id"] = read_df["well_pos"].map(well_to_ref)
        else:
            # --- Standard multi-ref mode ---
            _progress("Generating per-well consensus sequences...")
            well_df = utils.generate_per_well_consensus(
                well_df,
                read_df,
                str(output_dir),
                str(ref_dir),
                minimap2_path=tool_paths["minimap2"],
                samtools_path=tool_paths["samtools"],
                workers=workers,
            )

        # --- Stage 10: Variant calling ---
        _progress("Calling variants from consensus...")
        if vector_fasta is not None and flank_5p is not None:
            consensus_dir = str(output_dir / "wells" / "consensus")
            well_df = utils.extract_matches(
                well_df,
                flank_5p_len=len(flank_5p),
                flank_3p_len=len(flank_3p),
                consensus_dir=consensus_dir,
                frame_offset=frame_offset,
            )
        else:
            well_df = utils.extract_matches(well_df)

        # --- Stage 10.5: Streak-out candidate detection ---
        _progress("Screening for streak-out candidates...")
        from usortm.demux.streakout import (
            detect_streakout_candidates,
            save_streakout_results,
            generate_well_pileup_html,
        )

        streakout_dir = output_dir / "streakout"
        candidates = detect_streakout_candidates(
            well_df, read_df, str(ref_dir), str(output_dir),
            minimap2_path=tool_paths["minimap2"],
            samtools_path=tool_paths["samtools"],
            workers=workers,
            reference_fasta=str(reference) if reference is not None else None,
        )

        if candidates:
            streakout_dir.mkdir(exist_ok=True)
            save_streakout_results(candidates, str(streakout_dir))
            _flank_5p_len = len(flank_5p) if flank_5p is not None else 0
            _flank_3p_len = len(flank_3p) if flank_3p is not None else 0
            well_bam_dir = output_dir / "wells" / "consensus"
            for cand in candidates:
                _bam = str(well_bam_dir / f"{cand['global_well']}.bam")
                generate_well_pileup_html(
                    cand["global_well"], read_df, str(ref_dir), cand,
                    str(streakout_dir / f"well_{cand['plate']}_{cand['well']}.html"),
                    minimap2_path=tool_paths["minimap2"],
                    samtools_path=tool_paths["samtools"],
                    flank_5p_len=_flank_5p_len,
                    flank_3p_len=_flank_3p_len,
                    bam_path=_bam,
                )

        pipeline_stats["streakout"] = {
            "candidates": len(candidates),
            "recoverable_variants": list({
                v for c in candidates for v in c["recoverable_variants"]
            }),
        }

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

    if flank_5p is not None:
        results["flank_5p_len"] = len(flank_5p)
        results["flank_3p_len"] = len(flank_3p)

    return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_barcode_name_dfs(
    n_fbc: int = 96,
    n_rbc: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def _build_orient_ref_from_flanks(
    flank_5p: str,
    flank_3p: str,
    output_path: Path,
    variable_len: int = 300,
) -> None:
    """Build a single-entry orientation reference from vector flanking regions.

    Creates a FASTA with: flank_5p + N*variable_len + flank_3p
    minimap2 anchors on the conserved flanking regions for read orientation.
    The N spacer is not used for consensus — that happens downstream against
    per-variant references.
    """
    seq = flank_5p + ("N" * variable_len) + flank_3p
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f">orient_ref\n{seq}\n")


def _prepare_full_length_ref_fastas(
    multi_ref_fasta: Path,
    output_dir: Path,
    flank_5p: str,
    flank_3p: str,
) -> None:
    """Build full-length reference FASTAs by prepending/appending flanks.

    Each entry in the multi-entry reference FASTA (variable-only sequences)
    is wrapped with the 5' and 3' flanking sequences from the vector
    template and written to output_dir/single_ref_fastas/<id>.fasta.

    Args:
        multi_ref_fasta: Path to multi-entry reference FASTA (variable-only).
        output_dir: Parent directory. Files go into single_ref_fastas/ subdirectory.
        flank_5p: 5' flanking sequence to prepend.
        flank_3p: 3' flanking sequence to append.
    """
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq as BioSeq

    single_dir = Path(output_dir) / "single_ref_fastas"
    single_dir.mkdir(parents=True, exist_ok=True)

    for record in SeqIO.parse(str(multi_ref_fasta), "fasta"):
        full_seq = flank_5p + str(record.seq) + flank_3p
        full_record = SeqRecord(
            BioSeq(full_seq),
            id=record.id,
            description=record.description,
        )
        out_path = single_dir / f"{record.id}.fasta"
        SeqIO.write([full_record], str(out_path), "fasta")
        # Remove stale minimap2 / samtools indexes so downstream steps
        # regenerate them from the new full-length sequence.
        for _ext in (".mmi", ".fai"):
            _stale = Path(str(out_path) + _ext)
            if _stale.exists():
                _stale.unlink()


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
    if "depth" in well_df.columns:
        wells_passing = int((well_df["depth"] >= min_reads).sum())
    else:
        wells_passing = 0

    # Build per-well assignment dict
    well_assignments = {}
    for _, row in well_df.iterrows():
        plate = str(int(row["plate"]))
        well = str(row["well"])
        key = f"{plate}_{well}"

        # Extract variant name — strip strand prefix only, no suffix
        variant = str(row.get("major_ref", "unknown"))
        if ":" in variant:
            variant = variant.split(":")[-1]

        # cons_check stored separately, not appended to name
        _cc = row.get("cons_check")
        cons_check_val = str(_cc) if (_cc is not None and pd.notna(_cc)) else ""

        depth = row.get("depth", 0)
        if pd.isna(depth):
            depth = 0

        entry = {
            "plate": plate,
            "well": well,
            "reads": int(depth),
            "variant": variant,
            "consensus_fraction": float(row.get("major_freq", 0.0)),
            "cons_check": cons_check_val,
        }

        # Include flanking check if available
        _fc = row.get("flank_check")
        if _fc is not None and pd.notna(_fc):
            entry["flank_check"] = str(_fc)

        well_assignments[key] = entry

    # Compute actual sequence length stats from ref_len column
    seq_len_stats = {}
    if "ref_len" in well_df.columns:
        ref_lens = well_df["ref_len"].dropna().astype(int)
        if len(ref_lens) > 0:
            seq_len_stats = {
                "seq_len_min": int(ref_lens.min()),
                "seq_len_max": int(ref_lens.max()),
                "seq_len_median": int(ref_lens.median()),
            }

    result: dict = {
        "input_reads": input_reads,
        "aligned_reads": aligned_reads,
        "demuxed_reads": demuxed_reads,
        "assigned_reads": assigned_reads,
        "wells_with_data": wells_with_data,
        "wells_passing": wells_passing,
        "well_assignments": well_assignments,
        # Keep total_reads as alias for backward compat
        "total_reads": input_reads,
        **seq_len_stats,
    }
    if "read_len_hist" in stats:
        result["read_len_hist"] = stats["read_len_hist"]
    if "streakout" in stats:
        result["streakout"] = stats["streakout"]
    return result
