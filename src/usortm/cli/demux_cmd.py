"""Demultiplex sequencing data for a uSort-M project.

Orchestrates the LevSeq demultiplexing pipeline: Dorado barcode demux,
reference alignment, consensus generation, and variant calling.
"""

from typing import Optional
from pathlib import Path
import csv
import gzip
import json
import os
import shutil
from datetime import datetime

import typer
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from usortm.demux.deps import check_all_dependencies
from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"


def demux(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory (created by 'usortm plan').",
        exists=True,
    ),
    fastq: Optional[Path] = typer.Option(
        None,
        "--fastq", "-f",
        help=(
            "Path to FASTQ file or directory containing FASTQ files. "
            "Omit when --plate-map lists the FASTQs instead."
        ),
        exists=True,
        file_okay=True,
        dir_okay=True,
    ),
    plate_map_file: Optional[Path] = typer.Option(
        None,
        "--plate-map",
        help=(
            "TOML file mapping each FASTQ's barcode plates to sort plates. "
            "Needed when a run reuses barcode plates across separate FASTQs "
            "(more sort plates than the kit's 8 barcode plates). Defaults to "
            "<project>/plate_map.toml, or prompts if neither is present."
        ),
        exists=True,
        dir_okay=False,
    ),
    barcodes: Optional[Path] = typer.Option(
        None,
        "--barcodes", "-b",
        help="CSV file mapping wells to barcodes (overrides project default).",
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help=(
            "Reference FASTA for alignment. Required unless --library-csv "
            "is given (or a re-order round supplies one)."
        ),
    ),
    library_csv: Optional[Path] = typer.Option(
        None,
        "--library-csv", "-l",
        help=(
            "Library CSV with Name,Sequence columns. "
            "Auto-converted to reference FASTA (uppercase variable region)."
        ),
    ),
    min_reads: int = typer.Option(
        100,
        "--min-reads",
        help="Minimum reads per well to call a variant.",
    ),
    min_fraction: float = typer.Option(
        0.8,
        "--min-fraction",
        help="Minimum fraction of reads supporting consensus.",
    ),
    threads: int = typer.Option(
        4,
        "--threads", "-t",
        help="Number of threads for alignment.",
    ),
    subsample: Optional[int] = typer.Option(
        None,
        "--subsample", "-n",
        help="Subsample to this many reads before running the pipeline.",
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Number of parallel workers for per-well consensus alignment.",
    ),
    orient_ref: Optional[Path] = typer.Option(
        None,
        "--orient-ref",
        help=(
            "Single reference FASTA for read orientation only. "
            "Use when the library has many near-identical variants "
            "(e.g. site-saturation mutagenesis) to avoid slow "
            "multi-ref alignment. Reads are oriented against this "
            "reference but still assigned to variants from --reference."
        ),
    ),
    vector_fasta: Optional[Path] = typer.Option(
        None,
        "--vector-fasta",
        help=(
            "Vector FASTA with X's marking the variable region. "
            "Enables flanking region mismatch detection for encoded tags "
            "(e.g. SNAP, eGFP, His) adjacent to the variable sequence. "
            "Also auto-generates an orientation reference from the "
            "conserved backbone, replacing the slow multi-ref alignment "
            "(equivalent to --orient-ref but automatic)."
        ),
    ),
    mask_config_file: Optional[str] = typer.Option(
        None,
        "--mask-config",
        help=(
            "Barcode mask config: a preset name (see 'usortm config list') "
            "or path to a TOML file with [fbc] mask sequences. "
            "RBC masks are auto-derived if omitted. "
            "Defaults to cutinase backbone masks."
        ),
    ),
    reads_per_well: int = typer.Option(
        20,
        "--reads-per-well",
        help=(
            "Number of reads to sample per well for variant assignment "
            "(used in --orient-ref / --vector-fasta mode). Increase for "
            "libraries with near-identical variants (e.g. single-substitution)."
        ),
        min=1,
    ),
    round_num: int = typer.Option(
        1,
        "--round",
        help="Sequencing round to demultiplex (1 for initial sort, 2+ for re-order rounds).",
        min=1,
    ),
):
    """
    Demultiplex sequencing data for a [#4096E3]uSort-M[/#4096E3] project.

    Runs the full LevSeq pipeline: barcode demux with Dorado, reference
    alignment with minimap2, per-well consensus, and variant calling.

    [bold]Input requirements:[/bold]

    \u2022 Project directory from 'usortm plan'
    \u2022 FASTQ file from nanopore sequencing
    \u2022 Reference FASTA (--reference) or library CSV (--library-csv)

    [bold]Example:[/bold]

        usortm demux my_project/ --fastq reads.fastq --reference ref.fasta
    """
    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(
            f"[red]Error:[/red] Not a valid uSort-M project "
            f"(missing {PROJECT_STATE_FILE})"
        )
        console.print("Run 'usortm plan' first to create a project.")
        raise typer.Exit(1)

    with open(state_file) as f:
        project = json.load(f)

    # ------------------------------------------------------------------
    # Resolve round-specific paths and parameters
    # ------------------------------------------------------------------
    round_state_file: Optional[Path] = None
    round_state: Optional[dict] = None

    if round_num > 1:
        round_dir = project_dir / "rounds" / str(round_num)
        round_state_file = round_dir / "usortm_round.json"
        if not round_state_file.exists():
            console.print(f"[red]Error:[/red] Round {round_num} has not been planned yet.")
            console.print(
                f"Run: [cyan]usortm plan <variants.csv> "
                f"--output {project_dir}/ --round {round_num}[/cyan]"
            )
            raise typer.Exit(1)
        with open(round_state_file) as f:
            round_state = json.load(f)
        demux_output = project_dir / "rounds" / str(round_num) / "demux_output"
        effective_params = round_state  # n_plates, barcode_kit, library_size from round

        # Auto-use round variants as library reference when not explicitly provided.
        # Prefer matching against the original library (sequence-based, naming-agnostic)
        # so re-order rounds always use clean sequences without cloning artifacts.
        if library_csv is None and reference is None:
            round_variants = round_dir / "variants.csv"
            if round_variants.exists():
                original_ref = project_dir / "demux_output" / "library_reference.fasta"
                subset_ref = _extract_original_subset(
                    round_variants_csv=round_variants,
                    original_ref_fasta=original_ref,
                    demux_output=demux_output,
                )
                if subset_ref is not None:
                    reference = subset_ref
                    console.print(
                        f"[green]\u2713[/green] Matched round {round_num} variants to "
                        f"original library sequences: {subset_ref}"
                    )
                else:
                    # Fall back to round CSV if original library not available
                    library_csv = round_variants
                    console.print(
                        f"[green]\u2713[/green] Auto-using round {round_num} variants "
                        f"as reference: {round_variants}"
                    )
    else:
        demux_output = project_dir / "demux_output"
        effective_params = project
        round_dir = None

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Demultiplexing",
        border_style=BORDER_STYLE,
    ))
    console.print()

    # Auto-convert CSV passed as --reference (convenience shortcut)
    if (
        reference is not None
        and library_csv is None
        and reference.suffix.lower() == ".csv"
    ):
        from usortm.demux.utils import csv_to_reference_fasta
        csv_source = reference
        ref_fasta_path = demux_output / "library_reference.fasta"
        ref_fasta_path.parent.mkdir(parents=True, exist_ok=True)
        csv_to_reference_fasta(
            csv_path=str(csv_source),
            fasta_path=str(ref_fasta_path),
            strip_flanking=True,
        )
        reference = ref_fasta_path
        console.print(
            f"[green]\u2713[/green] Auto-converted CSV to reference FASTA "
            f"({ref_fasta_path})"
        )

    # Convert library CSV to reference FASTA if provided
    if library_csv is not None:
        if reference is not None:
            console.print(
                "[yellow]Warning:[/yellow] Both --reference and --library-csv "
                "provided. Using --library-csv (overrides --reference)."
            )
        from usortm.demux.utils import csv_to_reference_fasta
        ref_fasta_path = demux_output / "library_reference.fasta"
        ref_fasta_path.parent.mkdir(parents=True, exist_ok=True)
        csv_to_reference_fasta(
            csv_path=str(library_csv),
            fasta_path=str(ref_fasta_path),
            strip_flanking=True,
        )
        reference = ref_fasta_path
        console.print(
            f"[green]\u2713[/green] Converted library CSV to reference FASTA "
            f"({ref_fasta_path})"
        )

    # A reference is required.  Without one the pipeline skips alignment,
    # so no read gets a reference or a sequence and every row is dropped
    # later — producing an empty result that looks like a barcode problem.
    if reference is None:
        console.print(
            "[red]Error:[/red] a reference is required for demultiplexing."
        )
        console.print("Pass one of:")
        console.print(
            "  [cyan]--reference[/cyan]    ref.fasta      "
            "(reference FASTA)"
        )
        console.print(
            "  [cyan]--library-csv[/cyan]  variants.csv   "
            "(Name,Sequence CSV — converted for you)"
        )
        raise typer.Exit(1)

    # Check external tool dependencies before starting
    try:
        tools = check_all_dependencies()
        for name, path in tools.items():
            console.print(f"[green]\u2713[/green] Found {name}: {path}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Install missing tools or add them to your PATH.")
        raise typer.Exit(1)

    console.print()

    # Parse mask config if provided
    mask_config = None
    if mask_config_file is not None:
        resolved = _resolve_mask_config(mask_config_file)
        mask_config = _load_mask_config(resolved)
        console.print(f"[green]\u2713[/green] Loaded mask config from {resolved}")
    else:
        # Check for default mask config in project directory
        default_mask = project_dir / "mask_config.toml"
        if default_mask.exists():
            mask_config = _load_mask_config(default_mask)
            console.print(f"[green]\u2713[/green] Using project mask config ({default_mask})")
        else:
            # Interactive preset selection
            mask_config = _prompt_preset_selection()

    # Create output directory (path already resolved for the correct round above)
    demux_output.mkdir(parents=True, exist_ok=True)

    # Resolve how each FASTQ's barcode plates map onto sort plates.  A single
    # FASTQ with matching plate numbers is the ordinary case and runs straight
    # into demux_output; anything else is demuxed per FASTQ and merged, since
    # a reused barcode plate means the same barcode denotes different sort
    # plates in different files.
    segments = _resolve_plate_map(
        plate_map_file, project_dir, fastq,
        effective_params.get("n_plates", 1),
    )
    single_plain_run = len(segments) == 1 and _is_identity_map(segments[0].plates)

    per_segment = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting pipeline...", total=None)

        for segment in segments:
            if single_plain_run:
                seg_dir = demux_output
                label = ""
            else:
                seg_dir = demux_output / "segments" / segment.name
                seg_dir.mkdir(parents=True, exist_ok=True)
                label = f"[{segment.name}] "

            def on_progress(msg: str, _label=label):
                """Update the spinner text as the pipeline progresses."""
                progress.update(task, description=f"{_label}{msg}")

            seg_fastq = segment.path
            if seg_fastq.is_dir():
                seg_fastq = _concat_fastq_dir(seg_fastq, seg_dir)

            results = _run_demux(
                fastq=seg_fastq,
                output_dir=seg_dir,
                reference=reference,
                min_reads=min_reads,
                min_fraction=min_fraction,
                threads=threads,
                workers=workers,
                project_params=effective_params,
                progress_callback=on_progress,
                mask_config=mask_config,
                subsample=subsample,
                orient_ref=orient_ref,
                vector_fasta=vector_fasta,
                reads_per_well=reads_per_well,
                plate_map=None if single_plain_run else segment.plates,
            )
            per_segment.append((segment, results, seg_dir))

    if single_plain_run:
        results = per_segment[0][1]
    else:
        console.print()
        console.print("Merging segments onto a single plate numbering...")
        results = _merge_segment_results(per_segment, demux_output, min_reads)
        # Per-well artefacts are named by sort plate, which is unique to one
        # segment, so the merged view can link them all into place.
        for _segment, _results, seg_dir in per_segment:
            for sub in ("wells", "streakout", "reference_fasta"):
                _link_or_copy_tree(seg_dir / sub, demux_output / sub)
        console.print(
            f"[green]✓[/green] Merged {len(per_segment)} FASTQ segment(s) "
            f"covering sort plates "
            f"{', '.join(str(p) for seg in segments for p in seg.sort_plates)}"
        )

    # Save results (skip recovery-curve simulation for round > 1)
    _save_demux_results(results, demux_output, project=(None if round_num > 1 else project))

    # Build streakout wells set for plate map annotation
    streakout_wells = set()
    streakout_info = results.get("streakout", {})
    if streakout_info.get("candidates", 0) > 0:
        streakout_csv = demux_output / "streakout" / "streakout_candidates.csv"
        if streakout_csv.exists():
            with open(streakout_csv) as _sf:
                for row in csv.DictReader(_sf):
                    streakout_wells.add(f"{row['plate']}_{row['well']}")

    # Build mutation wells set (wells assigned to a library member but with a
    # non-synonymous/indel mutation in the consensus sequence).
    # Exclude streakout candidates — they already have a blue triangle and their
    # bad consensus is a result of mixed reads, not a true mutation.
    mutation_wells = set()
    silent_mutation_wells = set()
    assignments_csv = demux_output / "well_assignments.csv"
    if assignments_csv.exists():
        with open(assignments_csv) as _af:
            for row in csv.DictReader(_af):
                cons = row.get("cons_check", "")
                reads = int(row.get("reads", 0))
                well_key = f"{row['plate']}_{row['well']}"
                if reads >= 20 and well_key not in streakout_wells:
                    if cons in ("Other Error", "Error"):
                        mutation_wells.add(well_key)
                    elif cons == "Silent Mutation":
                        silent_mutation_wells.add(well_key)

    # Generate interactive plate map (Bokeh is an optional dependency)
    try:
        from usortm.demux.viz import load_plate_map_reads, save_plate_map_html

        read_df_path = demux_output / "read_df.csv"
        if read_df_path.exists():
            read_df = load_plate_map_reads(read_df_path)
            if read_df.empty:
                console.print(
                    "[yellow]⚠[/yellow] Plate map skipped: no reads were assigned to wells "
                    "(no reads had both FBC + RBC barcodes classified AND aligned to the reference). "
                    "Check demux_output/read_df.csv and consider using a larger subsample or "
                    "verifying your reference FASTA."
                )
            else:
                # Filter out ghost plates (< 20 total reads = likely barcode switching)
                _plate_col = read_df["well_pos"].str.split("_").str[0]
                _plate_totals = _plate_col.groupby(_plate_col).transform("count")
                read_df = read_df[_plate_totals >= 20]
                plate_map_path = demux_output / "plate_map.html"
                save_plate_map_html(
                    read_df, str(plate_map_path),
                    title="Demux Plate Map",
                    streakout_wells=streakout_wells,
                    mutation_wells=mutation_wells,
                    silent_mutation_wells=silent_mutation_wells,
                    min_reads=min_reads,
                )
                console.print(
                    f"[green]\u2713[/green] Plate map saved to {plate_map_path}"
                )
    except ImportError:
        pass  # Bokeh not installed — skip plate map
    except ValueError as e:
        console.print(f"[yellow]⚠[/yellow] Plate map skipped: {e}")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not generate plate map: {e}")

    # Update project state
    demux_step_data = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        # Kept for single-FASTQ runs, which is what most projects are.
        "fastq": str(segments[0].path.absolute()),
        "input_reads": results.get("input_reads", 0),
        "aligned_reads": results.get("aligned_reads", 0),
        "demuxed_reads": results.get("demuxed_reads", 0),
        "assigned_reads": results["assigned_reads"],
        "wells_with_data": results["wells_with_data"],
    }
    if not single_plain_run:
        # Record how each FASTQ's barcode plates were read, so the run can be
        # reproduced and the plate numbering explained after the fact.
        demux_step_data["segments"] = [
            {
                "name": seg.name,
                "fastq": str(seg.path.absolute()),
                "plates": {str(bc): sort for bc, sort in sorted(seg.plates.items())},
            }
            for seg in segments
        ]

    if round_num > 1:
        # Update round-specific state file
        round_state["workflow_steps"]["demux"] = demux_step_data
        with open(round_state_file, "w") as f:
            json.dump(round_state, f, indent=2)
        # Sync step status to master project JSON
        project.setdefault("rounds", {}).setdefault(str(round_num), {}).setdefault(
            "workflow_steps", {}
        )["demux"] = demux_step_data
        with open(state_file, "w") as f:
            json.dump(project, f, indent=2)
    else:
        project["workflow_steps"]["demux"] = demux_step_data
        with open(state_file, "w") as f:
            json.dump(project, f, indent=2)

    # Display summary table
    console.print()
    summary_table = Table(
        title="Demultiplexing Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="muted")
    summary_table.add_column("Value", justify="right")

    input_reads = results.get("input_reads") or 0
    aligned_reads = results.get("aligned_reads") or 0
    demuxed_reads = results.get("demuxed_reads") or 0
    assigned_reads = results.get("assigned_reads") or 0

    def _pct(n: int, total: int) -> str:
        if total > 0:
            return f"{n:,} ({n / total * 100:.1f}%)"
        return f"{n:,}"

    summary_table.add_row("Input reads", f"{input_reads:,}")
    if aligned_reads or input_reads:
        summary_table.add_row("Aligned", _pct(aligned_reads, input_reads))
    summary_table.add_row("Demuxed (FBC+RBC)", _pct(demuxed_reads, input_reads))
    summary_table.add_row("Assigned to wells", _pct(assigned_reads, input_reads))
    summary_table.add_row(
        "Wells with data", f"{results['wells_with_data']:,}"
    )
    summary_table.add_row(
        f"Wells \u2265{min_reads} reads", f"{results['wells_passing']:,}"
    )

    console.print(summary_table)
    console.print()

    # Streak-out candidate summary
    if streakout_info.get("candidates", 0) > 0:
        n_cands = streakout_info["candidates"]
        n_recov = len(streakout_info.get("recoverable_variants", []))
        console.print(
            f"[green]\u2713[/green] {n_cands} streak-out candidate(s) detected "
            f"({n_recov} recoverable variant(s))"
        )
        console.print(f"  Output: {demux_output / 'streakout'}/")
        console.print()

    # Flanking region summary (when --vector-fasta was used)
    if vector_fasta is not None:
        flank_counts: dict[str, int] = {}
        for data in results.get("well_assignments", {}).values():
            fc = data.get("flank_check", "")
            if fc:
                flank_counts[fc] = flank_counts.get(fc, 0) + 1
        if flank_counts:
            console.print("[bold]Flanking region check:[/bold]")
            for label in ["OK", "5' mismatch", "3' mismatch", "5'+3' mismatch", "No alignment"]:
                count = flank_counts.get(label, 0)
                if count > 0:
                    console.print(f"  {label}: {count:,} wells")
            console.print()

    round_flag = f" --round {round_num}" if round_num > 1 else ""
    console.print("[green]\u2713[/green] Demultiplexing complete!")
    console.print(f"  Results saved to: {demux_output}/")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(
        f"  [cyan]usortm pick {project_dir}/{round_flag}[/cyan]  "
        "\u2192 Generate hit-picking list"
    )
    console.print()


def _extract_original_subset(
    round_variants_csv: Path,
    original_ref_fasta: Path,
    demux_output: Path,
) -> Optional[Path]:
    """Match re-order round variants to original library sequences by similarity.

    Re-order round variants.csv files may contain cloning artifacts (e.g.
    restriction enzyme recognition sites) in their sequences that are absent
    from the physical library members.  This function identifies which entries
    in the original (round-1) library reference best match each round variant
    purely by sequence identity — completely independent of naming conventions.

    Steps:
      1. Strip all lowercase characters from each round variant sequence to
         obtain the uppercase (insert) region.
      2. For each round variant, score its identity against every original
         library entry using a fast k-mer overlap approach.
      3. Return a subset FASTA containing the best-matching original sequences.

    Args:
        round_variants_csv: Path to the re-order round variants.csv.
        original_ref_fasta: Path to the original library_reference.fasta.
        demux_output: Output directory where the subset FASTA will be written.

    Returns:
        Path to the subset FASTA file, or None if the original reference is
        missing or no matches are found.
    """
    if not original_ref_fasta.exists() or not round_variants_csv.exists():
        return None

    try:
        from Bio import SeqIO
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq as _BioSeq
    except ImportError:
        return None

    # ---- load original library sequences ---------------------------------
    orig_records = {
        rec.id: rec
        for rec in SeqIO.parse(str(original_ref_fasta), "fasta")
    }
    if not orig_records:
        return None

    # Build k-mer sets for each original sequence (k=15, every position).
    # NOTE: step must be 1 — a larger step skips positions and breaks matching
    # when the two sequences are offset by an amount not divisible by the step.
    _K = 15
    _STEP = 1

    def _kmers(seq: str) -> set:
        seq = seq.upper()
        return {seq[i: i + _K] for i in range(0, len(seq) - _K + 1, _STEP)}

    orig_kmers = {name: _kmers(str(rec.seq)) for name, rec in orig_records.items()}

    # ---- read round variants and strip lowercase (preserve CSV order) ----
    # round_order: list of round-2 names in the order they appear in the CSV.
    # round_stripped: round2_name -> uppercase insert sequence.
    round_order: list[str] = []
    round_stripped: dict[str, str] = {}
    with open(round_variants_csv) as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            name = row.get("Name", "").strip()
            raw_seq = row.get("Sequence", "")
            stripped = "".join(c for c in raw_seq if c.isupper())
            if name and stripped:
                round_order.append(name)
                round_stripped[name] = stripped

    if not round_stripped:
        return None

    # ---- match each round variant to the best original entry -------------
    # matched: list of (round2_name, orig_name) in round-2 CSV order.
    matched: list[tuple[str, str]] = []
    unmatched: list[str] = []

    for _rname in round_order:
        _rseq = round_stripped[_rname]
        _rkmers = _kmers(_rseq)
        best_name, best_score = None, -1
        for _oname, _okmers in orig_kmers.items():
            if _okmers:
                # k-mer overlap: shared k-mers / smaller set
                shared = len(_rkmers & _okmers)
                denom = min(len(_rkmers), len(_okmers))
                score = shared / denom if denom else 0.0
            else:
                # Original sequence is shorter than K; fall back to substring check.
                _oseq = str(orig_records[_oname].seq).upper()
                score = 1.0 if _oseq in _rseq.upper() else 0.0
            if score > best_score:
                best_score = score
                best_name = _oname
        # Require at least 50 % k-mer overlap to accept a match
        if best_name is not None and best_score >= 0.5:
            matched.append((_rname, best_name))
        else:
            unmatched.append(_rname)

    if not matched:
        return None

    if unmatched:
        console.print(
            f"[yellow]⚠[/yellow] {len(unmatched)} round variant(s) could not be "
            f"matched to the original library and will use their own sequences: "
            + ", ".join(unmatched[:5]) + ("…" if len(unmatched) > 5 else "")
        )

    # ---- write subset FASTA ----------------------------------------------
    # Records are named using the round-2 name so that the demux pipeline
    # assigns round-2 names to wells.  This ensures the pick command can
    # match demux results against the round-2 library order (which also uses
    # round-2 names), preventing duplicate placeholder rows.
    # Records are written in round-2 CSV order so the plate layout follows
    # the order specified in the reorder variants file.
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq as _BioSeq

    demux_output.mkdir(parents=True, exist_ok=True)
    subset_path = demux_output / "library_reference.fasta"
    subset_records = []
    for round2_name, orig_name in matched:
        orig_rec = orig_records[orig_name]
        renamed = SeqRecord(_BioSeq(str(orig_rec.seq)), id=round2_name, description="")
        subset_records.append(renamed)
    SeqIO.write(subset_records, str(subset_path), "fasta")

    console.print(
        f"[green]\u2713[/green] Matched {len(matched)} / "
        f"{len(round_stripped)} round variants to original library sequences"
    )
    return subset_path


def _load_mask_config(mask_file: Path) -> dict:
    """Load barcode mask sequences from a TOML file.

    Supports two formats:

    **Full format** (both ``[fbc]`` and ``[rbc]`` sections provided)::

        [fbc]
        mask1_front = "..."
        ...

        [rbc]
        mask1_front = "..."
        ...

    **Simplified format** (``[fbc]`` only — RBC auto-derived)::

        [meta]
        description = "My backbone"

        [fbc]
        mask1_front = "..."
        ...

    When ``[rbc]`` is absent, it is automatically derived from ``[fbc]``
    using the reverse-complement swap pattern.

    Returns:
        Dict with ``fbc`` and ``rbc`` sub-dicts (and optional ``meta``).
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(mask_file, "rb") as f:
        config = tomllib.load(f)

    if "fbc" in config and "rbc" not in config:
        from usortm.demux.barcodes import fbc_to_rbc_masks
        config["rbc"] = fbc_to_rbc_masks(config["fbc"])

    return config


def _resolve_mask_config(value: str) -> Path:
    """Resolve a ``--mask-config`` value to a TOML file path.

    Accepts either a filesystem path or a preset name.
    """
    path = Path(value)
    if path.is_file():
        return path

    from usortm.demux.presets import get_preset
    try:
        return get_preset(value)
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] '{value}' is not a file or known preset."
        )
        console.print("Run [cyan]usortm config list[/cyan] to see available presets.")
        raise typer.Exit(1)


def _prompt_preset_selection() -> Optional[dict]:
    """Interactively prompt the user to select a mask preset.

    Returns the loaded mask config dict, or None to use defaults.
    """
    import questionary
    from usortm.demux.presets import list_presets

    presets = list_presets()
    if not presets:
        return None

    choices = [
        questionary.Choice(
            title=f"{p['name']} — {p['description']}" if p["description"] else p["name"],
            value=i + 1,
        )
        for i, p in enumerate(presets)
    ]
    choices.append(questionary.Choice(title="None (use built-in defaults)", value=0))

    import sys
    if not sys.stdin.isatty():
        return None

    try:
        answer = questionary.select("Select a mask config preset:", choices=choices).ask()
    except KeyboardInterrupt:
        return None

    if answer is None or answer == 0:
        return None

    selected = presets[answer - 1]
    console.print(f"[green]\u2713[/green] Using preset: {selected['name']}")
    return _load_mask_config(selected["path"])


def _is_identity_map(plates: dict) -> bool:
    """True when every barcode plate maps to the sort plate of the same number."""
    return all(bc == sort for bc, sort in plates.items())


def _describe_segments(segments: list) -> None:
    """Print a table of the resolved FASTQ-to-sort-plate mapping."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("FASTQ")
    table.add_column("Barcode plates")
    table.add_column("Sort plates")
    for seg in segments:
        table.add_row(
            seg.name,
            ", ".join(str(b) for b in seg.barcode_plates),
            ", ".join(str(s) for s in seg.sort_plates),
        )
    console.print(table)


def _prompt_plate_map(fastq: Optional[Path], n_plates: int, project_dir: Path):
    """Interactively build a plate map when none was supplied.

    Offers the ordinary one-FASTQ layout first, since that is what most runs
    use.  A multi-FASTQ mapping is collected FASTQ by FASTQ and offered for
    saving, so the next run can pass ``--plate-map`` instead.

    Args:
        fastq: The ``--fastq`` value, if one was given.
        n_plates: Sort plate count from the project.
        project_dir: Project directory, used to offer a save location.

    Returns:
        List of Segment, or None to fall back to the default behaviour.
    """
    import sys
    from usortm.demux.plate_map import (
        MAX_BARCODE_PLATES, PlateMapError, Segment, identity_segment,
        write_plate_map,
    )

    if not sys.stdin.isatty():
        return None

    import questionary

    console.print(
        "\n[muted]Barcode plates normally match sort plates one-to-one. They "
        "differ when a run has more sort plates than the kit has barcode "
        f"plates ({MAX_BARCODE_PLATES}), so barcode plates are reused across "
        "separate FASTQs.[/muted]"
    )

    try:
        choice = questionary.select(
            "How do barcode plates map to sort plates?",
            choices=[
                questionary.Choice(
                    title=f"One FASTQ, barcode plate = sort plate (1-{n_plates})",
                    value="identity",
                ),
                questionary.Choice(
                    title="Multiple FASTQs with reused barcode plates — configure now",
                    value="custom",
                ),
            ],
        ).ask()
    except KeyboardInterrupt:
        return None

    if choice is None:
        return None
    if choice == "identity":
        return [identity_segment(fastq, n_plates)] if fastq else None

    segments: list = []
    while True:
        idx = len(segments) + 1
        try:
            path_str = questionary.text(
                f"FASTQ #{idx} — path to file or directory:"
            ).ask()
            if not path_str:
                break
            pairs_str = questionary.text(
                f"FASTQ #{idx} — barcode:sort plate pairs "
                "(e.g. '7:7, 8:8, 1:9, 2:10'):"
            ).ask()
            if not pairs_str:
                break
        except KeyboardInterrupt:
            return None

        try:
            plates = _parse_plate_pairs(pairs_str)
        except PlateMapError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        path = Path(path_str).expanduser()
        if not path.exists():
            console.print(f"[red]Error:[/red] {path} does not exist.")
            continue

        segments.append(Segment(name=path.stem or f"segment{idx}", path=path,
                                plates=plates))
        console.print(f"  [green]✓[/green] {path.name}: {segments[-1].describe()}")

        try:
            if not questionary.confirm("Add another FASTQ?", default=True).ask():
                break
        except KeyboardInterrupt:
            return None

    if not segments:
        return None

    from usortm.demux.plate_map import _validate_across_segments
    try:
        _validate_across_segments(segments)
    except PlateMapError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return None

    console.print()
    _describe_segments(segments)

    try:
        if questionary.confirm(
            "Save this mapping for future runs?", default=True
        ).ask():
            saved = write_plate_map(segments, project_dir / "plate_map.toml")
            console.print(
                f"[green]✓[/green] Saved to {saved} — reuse with "
                f"[cyan]--plate-map {saved}[/cyan]"
            )
    except KeyboardInterrupt:
        pass

    return segments


def _parse_plate_pairs(text: str) -> dict:
    """Parse ``'7:7, 8:8, 1:9'`` into ``{7: 7, 8: 8, 1: 9}``."""
    from usortm.demux.plate_map import parse_plate_map

    plates: dict = {}
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk and "=" not in chunk:
            from usortm.demux.plate_map import PlateMapError
            raise PlateMapError(
                f"'{chunk}' is not a barcode:sort pair (expected e.g. '1:9')."
            )
        sep = ":" if ":" in chunk else "="
        bc_str, sort_str = chunk.split(sep, 1)
        plates[bc_str.strip()] = sort_str.strip()

    # Reuse the config validator so interactive and file input agree.
    doc = {"fastq": [{"path": "x", "plates": {
        k: int(v) if str(v).lstrip("-").isdigit() else v for k, v in plates.items()
    }}]}
    return parse_plate_map(doc)[0].plates


def _resolve_plate_map(
    plate_map_file: Optional[Path],
    project_dir: Path,
    fastq: Optional[Path],
    n_plates: int,
):
    """Decide which FASTQ-to-sort-plate mapping this run uses.

    Order of precedence: an explicit ``--plate-map``, then a ``plate_map.toml``
    saved in the project directory, then an interactive prompt, and finally the
    ordinary single-FASTQ layout.

    Returns:
        List of Segment.
    """
    from usortm.demux.plate_map import (
        PlateMapError, identity_segment, load_plate_map,
    )

    if plate_map_file is not None:
        try:
            segments = load_plate_map(plate_map_file)
        except PlateMapError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Loaded plate map from {plate_map_file}")
        _describe_segments(segments)
        return segments

    default_map = project_dir / "plate_map.toml"
    if default_map.exists():
        try:
            segments = load_plate_map(default_map)
        except PlateMapError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Using project plate map ({default_map})")
        _describe_segments(segments)
        return segments

    prompted = _prompt_plate_map(fastq, n_plates, project_dir)
    if prompted:
        return prompted

    if fastq is None:
        console.print(
            "[red]Error:[/red] no FASTQ given. Pass [cyan]--fastq[/cyan], or "
            "[cyan]--plate-map[/cyan] for a run whose barcode plates are "
            "reused across several FASTQs."
        )
        raise typer.Exit(1)
    return [identity_segment(fastq, n_plates)]


def _merge_segment_results(per_segment: list, output_dir: Path, min_reads: int) -> dict:
    """Combine per-segment pipeline results onto one sort-plate numbering.

    Sort plates are unique to a single segment, so per-well keys and per-well
    artefacts never collide and the tables concatenate directly.

    Args:
        per_segment: List of ``(segment, results, segment_dir)`` tuples.
        output_dir: Project demux output directory.
        min_reads: Minimum depth for a well to count as passing.

    Returns:
        A merged results dict in the same shape a single run returns.
    """
    import pandas as pd

    merged: dict = {
        "input_reads": 0, "aligned_reads": 0, "demuxed_reads": 0,
        "assigned_reads": 0, "well_assignments": {},
    }
    read_frames, well_frames = [], []
    streak_candidates, streak_variants = 0, set()
    hists = []

    for segment, results, seg_dir in per_segment:
        for key in ("input_reads", "aligned_reads", "demuxed_reads", "assigned_reads"):
            merged[key] += int(results.get(key, 0) or 0)
        merged["well_assignments"].update(results.get("well_assignments", {}))

        streak = results.get("streakout") or {}
        streak_candidates += int(streak.get("candidates", 0) or 0)
        streak_variants.update(streak.get("recoverable_variants", []) or [])

        if results.get("read_len_hist"):
            hists.append(results["read_len_hist"])
        for key in ("flank_5p_len", "flank_3p_len"):
            if key in results:
                merged[key] = results[key]

        for name, frames in (("read_df.csv", read_frames), ("well_df.csv", well_frames)):
            path = seg_dir / name
            if path.exists():
                frame = pd.read_csv(path)
                frame["segment"] = segment.name
                frames.append(frame)

    merged["total_reads"] = merged["input_reads"]

    if read_frames:
        combined_reads = pd.concat(read_frames, ignore_index=True)
        combined_reads.to_csv(output_dir / "read_df.csv", index=False)
    if well_frames:
        combined_wells = pd.concat(well_frames, ignore_index=True)
        combined_wells.to_csv(output_dir / "well_df.csv", index=False)
        merged["wells_with_data"] = len(combined_wells)
        if "depth" in combined_wells.columns:
            merged["wells_passing"] = int(
                (combined_wells["depth"] >= min_reads).sum()
            )
        else:
            merged["wells_passing"] = 0
        if "ref_len" in combined_wells.columns:
            ref_lens = combined_wells["ref_len"].dropna().astype(int)
            if len(ref_lens):
                merged["seq_len_min"] = int(ref_lens.min())
                merged["seq_len_max"] = int(ref_lens.max())
                merged["seq_len_median"] = int(ref_lens.median())
    else:
        merged["wells_with_data"] = 0
        merged["wells_passing"] = 0

    hist = _merge_read_len_hists(hists)
    if hist:
        merged["read_len_hist"] = hist
    if streak_candidates or streak_variants:
        merged["streakout"] = {
            "candidates": streak_candidates,
            "recoverable_variants": sorted(streak_variants),
        }
    return merged


def _merge_read_len_hists(hists: list) -> dict:
    """Combine per-segment read-length histograms.

    Counts add directly when the segments share a bin size.  When they do not,
    rebinning 50 fixed bins would invent precision that is not there, so the
    histogram covering the most reads is reported instead.
    """
    hists = [h for h in hists if h and h.get("counts")]
    if not hists:
        return {}
    if len(hists) == 1:
        return hists[0]

    bin_sizes = {h.get("bin_size") for h in hists}
    if len(bin_sizes) == 1:
        n_bins = len(hists[0]["counts"])
        counts = [sum(h["counts"][i] for h in hists) for i in range(n_bins)]
        total = sum(h.get("n_reads", 0) for h in hists)
        medians = sorted(h.get("median", 0) for h in hists)
        return {
            "bin_size": hists[0]["bin_size"],
            "counts": counts,
            "median": medians[len(medians) // 2],
            "n_reads": total,
        }
    return max(hists, key=lambda h: h.get("n_reads", 0))


def _link_or_copy_tree(src: Path, dest: Path) -> None:
    """Mirror *src* into *dest*, hard-linking files where the filesystem allows.

    Per-well FASTQs and BAMs are large, and a run's segments live under the
    same output directory, so linking keeps the merged view free rather than
    doubling the run's disk footprint.
    """
    if not src.exists():
        return
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        target = dest / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


def _concat_fastq_dir(directory: Path, output_dir: Path) -> Path:
    """Concatenate all FASTQ files in a directory into a single file.

    Scans for ``*.fastq``, ``*.fastq.gz``, ``*.fq``, ``*.fq.gz`` files
    (recursively), prints the found files, and concatenates them into
    ``output_dir/combined.fastq``.

    Args:
        directory: Directory to scan for FASTQ files.
        output_dir: Directory for the combined output file.

    Returns:
        Path to the combined FASTQ file.
    """
    patterns = ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz"]
    found = sorted(
        f for p in patterns for f in directory.rglob(p)
    )
    if not found:
        console.print(
            f"[red]Error:[/red] No FASTQ files found in {directory}"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]\u2713[/green] Found {len(found)} FASTQ file(s) in {directory}:"
    )
    for f in found:
        size_mb = f.stat().st_size / 1_048_576
        console.print(f"  {f.name}  ({size_mb:.1f} MB)")

    out_path = output_dir / "combined.fastq"
    console.print(f"Concatenating into {out_path} ...")
    with open(out_path, "wb") as out_fh:
        for f in found:
            if f.suffix == ".gz" or f.name.endswith(".fastq.gz") or f.name.endswith(".fq.gz"):
                with gzip.open(f, "rb") as in_fh:
                    for chunk in iter(lambda: in_fh.read(1 << 20), b""):
                        out_fh.write(chunk)
            else:
                with open(f, "rb") as in_fh:
                    for chunk in iter(lambda: in_fh.read(1 << 20), b""):
                        out_fh.write(chunk)
    console.print(f"[green]\u2713[/green] Combined FASTQ written to {out_path}")
    return out_path


def _run_demux(
    fastq: Path,
    output_dir: Path,
    reference: Optional[Path],
    min_reads: int,
    min_fraction: float,
    threads: int,
    workers: int = 4,
    project_params: dict = None,
    progress_callback=None,
    mask_config: dict = None,
    subsample: Optional[int] = None,
    orient_ref: Optional[Path] = None,
    vector_fasta: Optional[Path] = None,
    reads_per_well: int = 20,
    plate_map: Optional[dict] = None,
) -> dict:
    """Run the demultiplexing pipeline based on the project's barcode kit.

    Currently supports the LevSeq barcode kit. Delegates to the pipeline
    module which handles Dorado demux, alignment, and variant calling.

    Args:
        fastq: Path to input FASTQ file.
        output_dir: Output directory for results.
        reference: Optional reference FASTA for variant calling.
        min_reads: Minimum reads per well.
        min_fraction: Minimum consensus fraction.
        threads: Number of alignment threads.
        workers: Number of parallel workers for per-well consensus.
        project_params: Project state dict (from usortm_project.json).
        progress_callback: Optional progress update function.
        mask_config: Optional dict with ``fbc`` and ``rbc`` mask sequences.
        subsample: Optional number of reads to subsample before processing.
        reads_per_well: Number of reads to sample per well for variant
            assignment in orient-ref / vector-fasta mode.
        plate_map: Optional ``{barcode_plate: sort_plate}`` mapping for this
            FASTQ, when a run reuses barcode plates across several files.

    Returns:
        Results dict with input_reads, aligned_reads, demuxed_reads,
        assigned_reads, wells_with_data, wells_passing, and
        well_assignments.
    """
    # Extract project parameters
    barcode_kit = "levseq"
    n_plates = 1
    if project_params:
        barcode_kit = project_params.get("barcode_kit", "levseq")
        n_plates = project_params.get("n_plates", 1)

    if barcode_kit.lower() == "levseq":
        from usortm.demux.pipeline import run_levseq_pipeline
        return run_levseq_pipeline(
            fastq=fastq,
            output_dir=output_dir,
            reference=reference,
            n_plates=n_plates,
            min_reads=min_reads,
            min_fraction=min_fraction,
            threads=threads,
            workers=workers,
            progress_callback=progress_callback,
            mask_config=mask_config,
            subsample=subsample,
            orient_ref=orient_ref,
            vector_fasta=vector_fasta,
            reads_per_well=reads_per_well,
            plate_map=plate_map,
        )
    else:
        raise NotImplementedError(
            f"Barcode kit '{barcode_kit}' is not yet supported for "
            "automated demux. Use 'levseq' or run Dorado manually."
        )


def _load_barcode_map(barcode_file: Path) -> dict:
    """Load barcode-to-well mapping from a CSV file.

    Supports three barcode CSV formats:
        - Single barcode: column 'barcode_seq'
        - Dual barcodes: columns 'fwd_barcode' and 'rev_barcode'
        - Barcode IDs: column 'barcode_id'

    Args:
        barcode_file: Path to the barcode CSV file.

    Returns:
        Dict mapping barcode key to {plate, well} info.
    """
    barcode_map = {}

    with open(barcode_file, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        for row in reader:
            plate = row.get("plate", "1")
            well = row.get("well", "")

            if "barcode_seq" in headers and row.get("barcode_seq"):
                barcode = row["barcode_seq"]
                barcode_map[barcode] = {"plate": plate, "well": well}
            elif "fwd_barcode" in headers and "rev_barcode" in headers:
                fwd = row.get("fwd_barcode", "")
                rev = row.get("rev_barcode", "")
                if fwd and rev:
                    barcode_map[f"{fwd}_{rev}"] = {
                        "plate": plate, "well": well
                    }
            elif "barcode_id" in headers:
                barcode_id = row.get("barcode_id", "")
                barcode_map[barcode_id] = {"plate": plate, "well": well}

    return barcode_map


def _compute_recovery_curve(library_size: int, skew: float = 4.0) -> Optional[dict]:
    """Run sortm simulations over a range of fold-sampling values.

    Returns dict with 'fold_samplings', 'coverage_means', and 'coverage_stds'
    lists, or None if the simulate module is unavailable.
    """
    try:
        from usortm.simulate.sortm import sortm
        import numpy as np
    except (ImportError, SystemError):
        return None

    fold_samplings = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    coverage_means = []
    coverage_stds = []
    for fs in fold_samplings:
        result = sortm(
            n_sims=30,
            lib_size=library_size,
            fold_sampling=fs,
            skew=skew,
            p_grow=0.67,
            return_correct=True,
            seed=42,
        )
        coverage_means.append(round(float(np.mean(result) / library_size * 100), 2))
        coverage_stds.append(round(float(np.std(result) / library_size * 100), 2))
    return {
        "fold_samplings": fold_samplings,
        "coverage_means": coverage_means,
        "coverage_stds": coverage_stds,
    }


def _save_demux_results(results: dict, output_dir: Path, project: Optional[dict] = None):
    """Save demultiplexing results to JSON summary and CSV well assignments.

    Output files:
        - demux_summary.json: aggregate statistics
        - well_assignments.csv: per-well data (plate, well, reads, variant,
          consensus_fraction)

    Args:
        results: Results dict from the pipeline.
        output_dir: Output directory.
        project: Project state dict (used to extract library_size and skew for
            the recovery curve simulation).
    """
    # Save summary JSON
    summary = {
        "input_reads": results.get("input_reads", 0),
        "aligned_reads": results.get("aligned_reads", 0),
        "demuxed_reads": results.get("demuxed_reads", 0),
        "assigned_reads": results["assigned_reads"],
        "wells_with_data": results["wells_with_data"],
        "wells_passing": results["wells_passing"],
    }
    for key in ("seq_len_min", "seq_len_max", "seq_len_median"):
        if key in results:
            summary[key] = results[key]
    if "read_len_hist" in results:
        summary["read_len_hist"] = results["read_len_hist"]
    if "streakout" in results:
        summary["streakout"] = results["streakout"]
    if "flank_5p_len" in results:
        summary["flank_5p_len"] = results["flank_5p_len"]
        summary["flank_3p_len"] = results["flank_3p_len"]

    # Pre-compute recovery curve if library_size is known
    if project:
        library_size = project.get("library_size")
        skew = float(project.get("skew", 4.0))
        if library_size and int(library_size) > 0:
            console.print("[muted]Running coverage simulations...[/muted]")
            curve = _compute_recovery_curve(int(library_size), skew)
            if curve:
                summary["recovery_curve"] = curve

    with open(output_dir / "demux_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save well assignments CSV
    # Determine if flanking data is present
    has_flanks = any(
        "flank_check" in data
        for data in results["well_assignments"].values()
    )
    has_protein = any(
        "protein_check" in data
        for data in results["well_assignments"].values()
    )
    has_confidence = any(
        "assignment_confidence" in data
        for data in results["well_assignments"].values()
    )
    has_flagged = any(
        "n_flagged_positions" in data
        for data in results["well_assignments"].values()
    )
    header = ["plate", "well", "reads", "variant", "consensus_fraction", "cons_check"]
    if has_flanks:
        header.append("flank_check")
    if has_protein:
        header.append("protein_check")
    if has_confidence:
        header.append("assignment_confidence")
    if has_flagged:
        header.extend(["n_flagged_positions", "max_mismatch_frac"])

    with open(output_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for well_id, data in results["well_assignments"].items():
            row = [
                data["plate"],
                data["well"],
                data["reads"],
                data["variant"],
                data["consensus_fraction"],
                data.get("cons_check", ""),
            ]
            if has_flanks:
                row.append(data.get("flank_check", ""))
            if has_protein:
                row.append(data.get("protein_check", ""))
            if has_confidence:
                row.append(data.get("assignment_confidence", ""))
            if has_flagged:
                row.append(data.get("n_flagged_positions", ""))
                row.append(data.get("max_mismatch_frac", ""))
            writer.writerow(row)
