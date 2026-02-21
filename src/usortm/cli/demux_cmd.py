"""Demultiplex sequencing data for a uSort-M project.

Orchestrates the LevSeq demultiplexing pipeline: Dorado barcode demux,
reference alignment, consensus generation, and variant calling.
"""

from typing import Optional
from pathlib import Path
import csv
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
        help="Path to FASTQ file with sequencing data.",
        exists=True,
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
    mask_config_file: Optional[Path] = typer.Option(
        None,
        "--mask-config",
        help=(
            "TOML file with custom barcode mask (flanking) sequences. "
            "Sections [fbc] and [rbc] with keys mask1_front, mask1_rear, "
            "mask2_front, mask2_rear. Defaults to cutinase backbone masks."
        ),
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
        ref_fasta_path = project_dir / "demux_output" / "library_reference.fasta"
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
        ref_fasta_path = project_dir / "demux_output" / "library_reference.fasta"
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
        mask_config = _load_mask_config(mask_config_file)
        console.print(f"[green]\u2713[/green] Loaded mask config from {mask_config_file}")
    else:
        # Check for default mask config in project directory
        default_mask = project_dir / "mask_config.toml"
        if default_mask.exists():
            mask_config = _load_mask_config(default_mask)
            console.print(f"[green]\u2713[/green] Using project mask config ({default_mask})")

    # Create output directory
    demux_output = project_dir / "demux_output"
    demux_output.mkdir(exist_ok=True)

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
            project_params=project,
            progress_callback=on_progress,
            mask_config=mask_config,
            subsample=subsample,
        )

        progress.update(task, description="Done!", completed=True)

    # Save results
    _save_demux_results(results, demux_output)

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
                plate_map_path = demux_output / "plate_map.html"
                save_plate_map_html(
                    read_df, str(plate_map_path),
                    title="Demux Plate Map",
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
    project["workflow_steps"]["demux"] = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "fastq": str(fastq.absolute()),
        "input_reads": results.get("input_reads", 0),
        "aligned_reads": results.get("aligned_reads", 0),
        "demuxed_reads": results.get("demuxed_reads", 0),
        "assigned_reads": results["assigned_reads"],
        "wells_with_data": results["wells_with_data"],
    }

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

    input_reads = results.get("input_reads", 0)
    aligned_reads = results.get("aligned_reads", 0)
    demuxed_reads = results.get("demuxed_reads", 0)
    assigned_reads = results.get("assigned_reads", 0)

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

    console.print("[green]\u2713[/green] Demultiplexing complete!")
    console.print(f"  Results saved to: {demux_output}/")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(
        f"  [cyan]usortm pick {project_dir}/[/cyan]  "
        "\u2192 Generate hit-picking list"
    )
    console.print()


def _load_mask_config(mask_file: Path) -> dict:
    """Load barcode mask sequences from a TOML file.

    Expected format::

        [fbc]
        mask1_front = "..."
        mask1_rear  = "..."
        mask2_front = "..."
        mask2_rear  = "..."

        [rbc]
        mask1_front = "..."
        mask1_rear  = "..."
        mask2_front = "..."
        mask2_rear  = "..."

    Returns:
        Dict with ``fbc`` and ``rbc`` sub-dicts.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(mask_file, "rb") as f:
        return tomllib.load(f)


def _run_demux(
    fastq: Path,
    output_dir: Path,
    reference: Optional[Path],
    min_reads: int,
    min_fraction: float,
    threads: int,
    project_params: dict = None,
    progress_callback=None,
    mask_config: dict = None,
    subsample: Optional[int] = None,
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
            progress_callback=progress_callback,
            mask_config=mask_config,
            subsample=subsample,
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


def _save_demux_results(results: dict, output_dir: Path):
    """Save demultiplexing results to JSON summary and CSV well assignments.

    Output files:
        - demux_summary.json: aggregate statistics
        - well_assignments.csv: per-well data (plate, well, reads, variant,
          consensus_fraction)

    Args:
        results: Results dict from the pipeline.
        output_dir: Output directory.
    """
    # Save summary JSON
    with open(output_dir / "demux_summary.json", "w") as f:
        json.dump({
            "input_reads": results.get("input_reads", 0),
            "aligned_reads": results.get("aligned_reads", 0),
            "demuxed_reads": results.get("demuxed_reads", 0),
            "assigned_reads": results["assigned_reads"],
            "wells_with_data": results["wells_with_data"],
            "wells_passing": results["wells_passing"],
        }, f, indent=2)

    # Save well assignments CSV
    with open(output_dir / "well_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plate", "well", "reads", "variant", "consensus_fraction"
        ])

        for well_id, data in results["well_assignments"].items():
            writer.writerow([
                data["plate"],
                data["well"],
                data["reads"],
                data["variant"],
                data["consensus_fraction"],
            ])
