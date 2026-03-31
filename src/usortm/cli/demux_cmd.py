"""Demultiplex sequencing data for a uSort-M project.

Orchestrates the LevSeq demultiplexing pipeline: Dorado barcode demux,
reference alignment, consensus generation, and variant calling.
"""

from typing import Optional
from pathlib import Path
import csv
import gzip
import json
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
    fastq: Path = typer.Option(
        ...,
        "--fastq", "-f",
        help="Path to FASTQ file or directory containing FASTQ files.",
        exists=True,
        file_okay=True,
        dir_okay=True,
    ),
    barcodes: Optional[Path] = typer.Option(
        None,
        "--barcodes", "-b",
        help="CSV file mapping wells to barcodes (overrides project default).",
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help="Reference FASTA for alignment (improves variant calling).",
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
    \u2022 Reference FASTA for variant calling (recommended)

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

        # Auto-use round variants as library reference when not explicitly provided
        if library_csv is None and reference is None:
            round_variants = round_dir / "variants.csv"
            if round_variants.exists():
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

    # If directory, concatenate all FASTQs into a single file
    if fastq.is_dir():
        fastq = _concat_fastq_dir(fastq, demux_output)

    # Run the pipeline with progress updates
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting pipeline...", total=None)

        def on_progress(msg: str):
            """Update the spinner text as the pipeline progresses."""
            progress.update(task, description=msg)

        results = _run_demux(
            fastq=fastq,
            output_dir=demux_output,
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
        )

        progress.update(task, description="Done!", completed=True)

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
    assignments_csv = demux_output / "well_assignments.csv"
    if assignments_csv.exists():
        with open(assignments_csv) as _af:
            for row in csv.DictReader(_af):
                if (
                    row.get("cons_check", "") in ("Other Error", "Error")
                    and int(row.get("reads", 0)) >= 20
                ):
                    well_key = f"{row['plate']}_{row['well']}"
                    if well_key not in streakout_wells:
                        mutation_wells.add(well_key)

    # Generate interactive plate map (Bokeh is an optional dependency)
    try:
        import pandas as pd
        from usortm.demux.viz import save_plate_map_html

        read_df_path = demux_output / "read_df.csv"
        if read_df_path.exists():
            read_df = pd.read_csv(read_df_path)
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
        "fastq": str(fastq.absolute()),
        "input_reads": results.get("input_reads", 0),
        "aligned_reads": results.get("aligned_reads", 0),
        "demuxed_reads": results.get("demuxed_reads", 0),
        "assigned_reads": results["assigned_reads"],
        "wells_with_data": results["wells_with_data"],
    }

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
    header = ["plate", "well", "reads", "variant", "consensus_fraction", "cons_check"]
    if has_flanks:
        header.append("flank_check")

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
            writer.writerow(row)
