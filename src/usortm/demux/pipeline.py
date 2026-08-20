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


def _extract_reads_gzip_aware(
    input_fastq: str,
    output_fastq: str,
    num_reads: int,
) -> int:
    """Extract the first *num_reads* from a FASTQ file (plain or gzipped).

    Returns the number of reads actually written (may be less than
    *num_reads* if the input file is shorter).
    """
    from usortm.demux.utils import resolve_fastq_inputs

    reads_written = 0
    with open(output_fastq, "w") as fh_out:
        for path in resolve_fastq_inputs(input_fastq):
            if reads_written >= num_reads:
                break
            open_fn = _open_fastq(path)
            with open_fn(path, "rt") as fh_in:
                while reads_written < num_reads:
                    lines = [fh_in.readline() for _ in range(4)]
                    if not lines[0]:
                        break
                    fh_out.writelines(lines)
                    reads_written += 1
    return reads_written


BARCODE_YIELD_CRITICAL = 0.02
BARCODE_YIELD_POOR = 0.20

#: Reads a well needs before it counts as having data.  Below this there is no
#: consensus worth calling, so counting such wells overstates what a run
#: recovered.  Matches the floor of the lowest quality tier.
WELL_DATA_MIN_READS = 20


def _check_barcode_yield(demux_stats: dict) -> Optional[dict]:
    """Flag a run where reads aligned but almost none carried a barcode.

    Dorado finds a barcode by the mask sequences flanking it, so masks built
    for a different backbone classify nothing while alignment still succeeds.
    That combination looks like a finished run with empty wells rather than a
    misconfiguration, so it is called out explicitly.

    Args:
        demux_stats: The ``demux`` entry of the pipeline stats.

    Returns:
        Dict with ``headline``, ``detail`` and ``severity`` when the yield is
        suspect, otherwise None.
    """
    total = demux_stats.get("ref_assigned", 0) or demux_stats.get("union_reads", 0)
    if not total:
        return None

    fbc = demux_stats.get("fbc_classified", 0)
    rbc = demux_stats.get("rbc_classified", 0)
    worst = min(fbc, rbc) / total
    if worst >= BARCODE_YIELD_POOR:
        return None

    severity = "critical" if worst < BARCODE_YIELD_CRITICAL else "low"
    headline = (
        f"Barcode classification is {'near zero' if severity == 'critical' else 'low'}: "
        f"{fbc:,} forward and {rbc:,} reverse of {total:,} aligned reads"
    )
    detail = (
        "Reads aligned to the reference, so they are the right molecules — "
        "Dorado just could not find the barcodes in them. That almost always "
        "means the mask sequences do not match this construct's backbone. "
        "Run `usortm masks derive <project>` to read the real flanking "
        "sequences off these reads."
    )
    return {"headline": headline, "detail": detail, "severity": severity,
            "fbc_frac": fbc / total, "rbc_frac": rbc / total}


def _wells_per_plate(well_df) -> dict:
    """Count wells per plate for the live dashboard.

    Returns an empty dict rather than raising: this feeds a display, and a
    malformed well table must not stop the run.
    """
    if well_df is None or len(well_df) == 0 or "plate" not in well_df.columns:
        return {}
    try:
        return {str(int(p)): int(n)
                for p, n in well_df["plate"].value_counts().items()}
    except (TypeError, ValueError):
        return {}


def _run_streakout(well_df, read_df, ref_dir, output_dir, tool_paths,
                   workers, reference, flank_5p, flank_3p):
    """Detect mixed wells worth streaking out, and draw a page for each.

    Both halves cost: detection builds a consensus per subpopulation, and the
    pages are rendered one well at a time.
    """
    from usortm.demux.streakout import (
        detect_streakout_candidates,
        save_streakout_results,
        generate_well_pileup_html,
    )

    candidates = detect_streakout_candidates(
        well_df, read_df, str(ref_dir), str(output_dir),
        minimap2_path=tool_paths["minimap2"],
        samtools_path=tool_paths["samtools"],
        workers=workers,
        reference_fasta=str(reference) if reference is not None else None,
    )
    if not candidates:
        return []

    streakout_dir = output_dir / "streakout"
    streakout_dir.mkdir(exist_ok=True)
    save_streakout_results(candidates, str(streakout_dir))
    flank_5p_len = len(flank_5p) if flank_5p is not None else 0
    flank_3p_len = len(flank_3p) if flank_3p is not None else 0
    well_bam_dir = output_dir / "wells" / "consensus"
    for cand in candidates:
        generate_well_pileup_html(
            cand["global_well"], read_df, str(ref_dir), cand,
            str(streakout_dir / f"well_{cand['plate']}_{cand['well']}.html"),
            minimap2_path=tool_paths["minimap2"],
            samtools_path=tool_paths["samtools"],
            flank_5p_len=flank_5p_len,
            flank_3p_len=flank_3p_len,
            bam_path=str(well_bam_dir / f"{cand['global_well']}.bam"),
        )
    return candidates


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
    reads_per_well: int = 20,
    plate_map: Optional[dict] = None,
    live_label: Optional[str] = None,
    live_report=None,
    resume: bool = False,
    streakout: bool = False,
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
        plate_map: Optional ``{barcode_plate: sort_plate}`` mapping for runs
            that reuse barcode plates across FASTQs.  When given, the number
            of reverse barcodes follows the mapping's highest barcode plate
            rather than *n_plates*, and reads on barcode plates outside the
            mapping are dropped.

    Returns:
        Dict with keys: input_reads, aligned_reads, demuxed_reads,
        assigned_reads, wells_with_data, wells_passing,
        well_assignments. Compatible with demux_cmd._save_demux_results().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # When a caller is rendering progress, keep the library's own bars and
    # status lines off the terminal so they do not interleave with it.
    utils.set_console_quiet(progress_callback is not None)

    # A dashboard that fills in as the run establishes each figure, so a
    # multi-hour run is inspectable before it ends.  A caller running several
    # segments passes one in, so the whole run reports to a single page rather
    # than one buried per segment.
    from usortm.demux.live import LiveReport

    if live_report is not None:
        live = live_report
        live.begin_segment(live_label or "")
    else:
        live = LiveReport(output_dir, label=live_label or "")

    def _progress(msg: str):
        """Report progress if a callback was provided."""
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # --- Stage 1: Check dependencies ---
    _progress("Checking dependencies...")
    tool_paths = check_all_dependencies()

    # --- Stage 2: Generate Dorado barcode config files ---
    live.set_stage("config")
    _progress("Generating barcode config files...")
    config_dir = output_dir / "dorado_config"
    if plate_map:
        # Reverse barcodes are generated contiguously from RB01, so cover up
        # to this segment's highest barcode plate.  Plates in that span but
        # absent from the mapping are filtered out later, in format_df.
        n_rbc = max(plate_map) * 4
    else:
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

    # Read lengths and the read count both come from the alignment, which has
    # to touch every read anyway.  Measuring them here instead meant
    # decompressing the whole input first, minutes before a run starts, and
    # the count that came out was the same one the aligner produces.
    read_len_hist = {}
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

        # The aligner saw every read, so its tallies are the authoritative
        # count and the exact length distribution.  Both survive the alignment
        # cache, so a resumed run still reports them without re-reading.
        read_len_hist = align_stats.get("read_len_hist") or {}
        if read_len_hist:
            pipeline_stats["read_len_hist"] = read_len_hist
        if not pipeline_stats.get("input_reads"):
            pipeline_stats["input_reads"] = (
                align_stats.get("mapped", 0) + align_stats.get("unmapped", 0)
            )
        input_reads = pipeline_stats.get("input_reads", 0)
        live.update(input_reads=input_reads, aligned=align_stats.get("mapped"),
                    read_len_hist=read_len_hist or None)
        _progress("Strand split complete.")

    # --- Stages 4 and 5: Dorado barcode demux (on oriented reads) ---
    # Only the barcode call per read is needed downstream, and that is in the
    # summary Dorado writes anyway.  Emitting FASTQs as well would write a
    # second and third full copy of the reads — measured at 1.24x the input
    # each, against 0.35x for summary-only — for data nothing reads: the
    # sequences come from the oriented FASTQ, not from here.
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
        output_fastq=False,
        emit_summary=True,
    )

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
        output_fastq=False,
        emit_summary=True,
    )

    # --- Stage 6: Build read DataFrame ---
    live.set_stage("readdf")
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

    live.update(fbc=pipeline_stats["demux"]["fbc_classified"],
                rbc=pipeline_stats["demux"]["rbc_classified"])
    barcode_warning = _check_barcode_yield(pipeline_stats["demux"])
    if barcode_warning:
        pipeline_stats["barcode_warning"] = barcode_warning
        live.update(warning=barcode_warning)
        _progress(barcode_warning["headline"])
        logger.warning(barcode_warning["headline"])

    # --- Stage 7: Map barcodes to well positions ---
    live.set_stage("wells")
    _progress("Mapping barcodes to well positions...")
    fbc_df, rbc_df = _build_barcode_name_dfs(n_fbc=96, n_rbc=n_rbc)
    pre_filter = len(read_df)
    read_df = utils.format_df(
        read_df,
        fbc_df=fbc_df,
        rbc_df=rbc_df,
        ref_fasta=str(reference) if reference else None,
        orient_ref_fasta=str(orient_ref) if orient_ref else None,
        plate_map=plate_map,
    )
    pipeline_stats["demux"]["complete_assignments"] = len(read_df)
    pipeline_stats["demux"]["dropped_incomplete"] = pre_filter - len(read_df)

    # --- Stage 8: Per-well summary ---
    _progress("Generating per-well summary...")
    well_df = utils.generate_well_df(read_df)
    live.update(wells=len(well_df), plates=_wells_per_plate(well_df))

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
            # The cutoff comes from the vector's flanks, so it only applies
            # when --vector-fasta supplied them.  With a bare --orient-ref
            # there is no amplicon length to reason about and every read is
            # kept.
            if flank_5p is not None and flank_3p is not None:
                _min_read_len = (len(flank_5p) + len(flank_3p)) // 2
                n_before = len(read_df)
                read_df = read_df[
                    read_df["read_seq"].str.len() >= _min_read_len
                ].reset_index(drop=True)
                n_removed = n_before - len(read_df)
                if n_removed:
                    _progress(
                        f"Filtered {n_removed:,} flank-only reads "
                        f"(<{_min_read_len} bp) that cannot overlap the "
                        "variable region"
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

            def _assign_progress(n_done, total):
                """Aligning every well's reads to the whole library is the
                longest single step, so report it rather than sit silent."""
                if total:
                    pct = int(100 * n_done / total)
                    _progress(
                        f"Assigning variants from read alignments... "
                        f"{n_done:,}/{total:,} ({pct}%)"
                    )

            well_df = utils.assign_variants_from_reads(
                well_df, read_df, str(reference),
                well_fastqs_dir=well_fastqs_dir,
                minimap2_path=tool_paths["minimap2"],
                workers=workers,
                progress_callback=_assign_progress,
                full_length_ref_dir=str(ref_dir / "single_ref_fastas"),
                reads_per_well=reads_per_well,
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
                resume=resume,
            )

            # Backfill read_df ref_name from well_df so plate map shows
            # the assigned variant instead of the orient-ref name.
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
                resume=resume,
            )

        # --- Stage 10: Variant calling ---
        live.set_stage("variants")
        _progress("Calling variants from consensus...")
        # The library's own sequences, so a well carrying the unmutated parent
        # can be told apart from a damaged one.  The parent is not a library
        # member, so a well holding it is assigned some variant it never had
        # and fails every check against it.
        library_inserts = []
        if reference:
            try:
                library_inserts = [
                    str(rec.seq) for rec in SeqIO.parse(str(reference), "fasta")
                ]
            except Exception:
                pass

        if vector_fasta is not None and flank_5p is not None:
            consensus_dir = str(output_dir / "wells" / "consensus")
            def _match_progress(n_done, total):
                _progress(f"Calling variants from consensus... "
                          f"{n_done:,}/{total:,}")

            well_df = utils.extract_matches(
                well_df,
                flank_5p_len=len(flank_5p),
                flank_3p_len=len(flank_3p),
                consensus_dir=consensus_dir,
                frame_offset=frame_offset,
                workers=workers,
                progress_callback=_match_progress,
                library_inserts=library_inserts,
            )
        else:
            well_df = utils.extract_matches(well_df, workers=workers)

        # --- Stage 10.5: Streak-out candidate detection ---
        #
        # Off unless asked for.  Streaking out a mixed well is a deliberate
        # decision taken about a handful of wells, not something every run
        # needs answered, and the stage pays for itself twice over: detection
        # builds a consensus per subpopulation, and each candidate then gets a
        # pileup page rendered one at a time.
        live.set_stage("streakout")
        candidates = []
        if not streakout:
            _progress("Skipping streak-out screening (--streakout to enable)")
        else:
            _progress("Screening for streak-out candidates...")
            candidates = _run_streakout(
                well_df, read_df, ref_dir, output_dir, tool_paths, workers,
                reference, flank_5p, flank_3p,
            )

        pipeline_stats["streakout"] = {
            "candidates": len(candidates),
            "recoverable_variants": list({
                v for c in candidates for v in c["recoverable_variants"]
            }),
        }


    # --- Stage 10.6: Consensus hotspot detection ---
    if reference is not None:
        _flank_5p_len = len(flank_5p) if flank_5p is not None else 0
        _flank_3p_len = len(flank_3p) if flank_3p is not None else 0
        hotspots = utils.detect_consensus_hotspots(
            well_df,
            threshold=0.1,
            flank_5p_len=_flank_5p_len,
            flank_3p_len=_flank_3p_len,
        )
        pipeline_stats["consensus_hotspots"] = hotspots

    live.set_stage("done")
    _progress("Finalizing results...")

    # Save intermediate DataFrames for debugging / power users
    utils.write_read_df_csv(read_df, output_dir / "read_df.csv")
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

    # What produced this run.  Recorded from the binaries actually used, since
    # which ones get found depends on PATH and can differ between runs on the
    # same machine.
    try:
        from usortm.demux.deps import tool_versions

        results["versions"] = tool_versions(tool_paths)
    except Exception as exc:
        logger.warning("Could not record tool versions: %s", exc)

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

    # A well with a handful of reads has no usable consensus and nothing can
    # be called from it, so counting it as having data overstates the run.
    # Twenty is the floor the quality tiers already use.
    if "depth" in well_df.columns:
        wells_with_data = int((well_df["depth"] >= WELL_DATA_MIN_READS).sum())
        wells_passing = int((well_df["depth"] >= min_reads).sum())
    else:
        wells_with_data = len(well_df)
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

        # Include protein-level check if available
        _pc = row.get("protein_check")
        if _pc is not None and pd.notna(_pc) and _pc:
            entry["protein_check"] = str(_pc)

        # Include assignment confidence if available
        _ac = row.get("assignment_confidence")
        if _ac is not None and pd.notna(_ac):
            entry["assignment_confidence"] = float(_ac)

        # Include flanking check if available
        _fc = row.get("flank_check")
        if _fc is not None and pd.notna(_fc):
            entry["flank_check"] = str(_fc)

        # Include N-base count in variable region if > 0
        _vn = row.get("var_n_count")
        if _vn is not None and pd.notna(_vn) and int(_vn) > 0:
            entry["var_n_count"] = int(_vn)

        # Include per-column mismatch flags if available
        _nfp = row.get("n_flagged_positions")
        if _nfp is not None and pd.notna(_nfp):
            entry["n_flagged_positions"] = int(_nfp)
        _mmf = row.get("max_mismatch_frac")
        if _mmf is not None and pd.notna(_mmf):
            entry["max_mismatch_frac"] = round(float(_mmf), 4)

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
    if "barcode_warning" in stats:
        result["barcode_warning"] = stats["barcode_warning"]
    return result
