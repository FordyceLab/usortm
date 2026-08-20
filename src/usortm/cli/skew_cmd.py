"""Measure library skew from sequencing reads and recommend a sorting depth.

Sits between `plan` and the wet lab: `plan` estimates skew from the
synthesis method before the library exists, and this command replaces that
estimate with a measurement once the amplified library has been sequenced.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from usortm.cli.theme import BORDER_STYLE, get_console
from usortm.paths import input_file

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"


def skew(
    fastq: Path = typer.Argument(
        ...,
        help="FASTQ of the amplified library (e.g. Plasmidsaurus premium PCR).",
        exists=True,
    ),
    project_dir: Optional[Path] = typer.Option(
        None,
        "--project", "-p",
        help="uSort-M project directory. Uses its variants.csv and, unless "
             "--no-update-plan is given, records the measurement in its project file.",
        exists=True,
    ),
    variants_file: Optional[Path] = typer.Option(
        None,
        "--variants", "-V",
        help="CSV of the starting variants (columns: Name, Sequence). "
             "Required when --project is not given.",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory [default: <project>/skew/ or ./usortm_skew/].",
    ),
    target_coverage: float = typer.Option(
        0.90,
        "--target-coverage",
        help="Fraction of the library to recover by sorting.",
    ),
    sorting_efficiency: float = typer.Option(
        0.67,
        "--sorting-efficiency",
        help="Fraction of sorted wells that grow.",
    ),
    basis: str = typer.Option(
        "empirical",
        "--basis",
        help="Recommend from the measured abundances ('empirical') or from a "
             "log-normal refit of them ('lognormal', matching 'usortm plan').",
    ),
    min_ref_cov: float = typer.Option(
        0.8,
        "--min-ref-cov",
        help="Minimum fraction of a variant an alignment must span to count.",
    ),
    margin: float = typer.Option(
        0.02,
        "--margin",
        help="Relative alignment-score lead the best variant must hold over "
             "the runner-up for a read to be counted.",
    ),
    threads: int = typer.Option(4, "--threads", "-t", help="minimap2 threads."),
    n_sims: int = typer.Option(
        100, "--n-sims", help="Simulations per fold-sampling evaluation."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Measure even when the variants are too similar to tell apart.",
    ),
    update_plan: bool = typer.Option(
        True,
        "--update-plan/--no-update-plan",
        help="Record the measurement in the project file (with --project).",
    ),
    html: bool = typer.Option(
        True, "--html/--no-html", help="Write an HTML summary with plots."
    ),
):
    """
    Measure library skew from reads and recommend a sorting depth.

    Sequence a little of the amplified library before sorting, and this turns
    those reads into a direct measurement of how evenly the library is
    distributed — replacing the skew that [cyan]plan[/cyan] had to assume
    from the synthesis method.

    Poisson counting noise at shallow depth makes a library look more skewed
    than it is, so the raw Q90/Q10 ratio is reported alongside a
    noise-corrected estimate, and the recommendation follows the corrected one.

    [bold]Example:[/bold]

        usortm skew library.fastq --project my_project/
    """
    if basis not in ("empirical", "lognormal"):
        console.print(
            f"[red]Error:[/red] --basis must be 'empirical' or 'lognormal', got '{basis}'"
        )
        raise typer.Exit(1)

    variants_csv = _resolve_variants(project_dir, variants_file)
    out_dir = _resolve_output(output, project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Library Skew",
        border_style=BORDER_STYLE,
    ))
    console.print()
    console.print(f"[green]✓[/green] Variants: [cyan]{variants_csv}[/cyan]")
    console.print(f"[green]✓[/green] Reads:    [cyan]{fastq}[/cyan]")
    console.print()

    from usortm.demux.deps import DependencyError
    from usortm.qc import (
        check_resolvability,
        count_variant_reads,
        measure_skew,
        recommend_sampling,
    )
    from usortm.qc.counting import count_fastq_reads
    from usortm.qc import LibraryProfile

    # --- Can these variants be told apart at all? ---
    try:
        with console.status("[muted]Checking variant separability...[/muted]"):
            resolvability = check_resolvability(variants_csv, threads=threads)
    except DependencyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    _report_resolvability(resolvability, force)

    # --- Count reads per variant ---
    try:
        total_reads = count_fastq_reads(fastq)
    except OSError:
        total_reads = None

    with Progress(
        TextColumn("[muted]{task.description}[/muted]"),
        BarColumn(complete_style=BORDER_STYLE),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Assigning reads", total=total_reads or 100)

        def on_progress(done, total):
            progress.update(task, completed=done, total=total or done or 100)

        try:
            counts = count_variant_reads(
                fastq, variants_csv, out_dir / "work",
                min_ref_cov=min_ref_cov, margin=margin, threads=threads,
                progress_callback=on_progress, total_reads=total_reads,
            )
        except DependencyError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        progress.update(task, completed=counts.total_reads, total=counts.total_reads)

    if counts.assigned_reads == 0:
        console.print()
        console.print(
            "[red]Error:[/red] no reads could be assigned to any variant.\n"
            "  Check that the FASTQ and the variant list describe the same library, "
            "and that reads span the variable region."
        )
        raise typer.Exit(1)

    _print_read_accounting(counts)

    # --- Deconvolve skew and search for a sorting depth ---
    stats = measure_skew(counts)

    with console.status("[muted]Simulating sorting depth...[/muted]") as status:
        def on_sim(iteration, fold, coverage):
            status.update(
                f"[muted]Simulating... {fold:.1f}× → {coverage:.1%} coverage[/muted]"
            )

        recommendation = recommend_sampling(
            stats,
            target_coverage=target_coverage,
            p_grow=sorting_efficiency,
            basis=basis,
            n_sims=n_sims,
            progress_callback=on_sim,
        )

    profile = LibraryProfile(
        counts=counts, stats=stats,
        recommendation=recommendation, resolvability=resolvability,
    )

    _print_abundance_histogram(counts, stats)
    _print_skew_table(stats)
    _print_recommendation(recommendation, stats, project_dir)

    # --- Write outputs ---
    written = _write_outputs(profile, out_dir, fastq, variants_csv, html)
    console.print("[green]✓[/green] Wrote:")
    for path in written:
        console.print(f"  • {path}")
    console.print()

    if project_dir is not None and update_plan:
        _update_project_state(project_dir, profile, fastq)
        console.print(
            f"[green]✓[/green] Recorded in "
            f"[cyan]{project_dir / PROJECT_STATE_FILE}[/cyan] "
            f"under [cyan]measured_skew[/cyan]"
        )
        console.print()

    _print_next_steps(recommendation, project_dir)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _resolve_variants(project_dir, variants_file) -> Path:
    """Locate the variant CSV from the explicit flag or the project."""
    if variants_file is not None:
        return variants_file
    if project_dir is None:
        console.print(
            "[red]Error:[/red] provide either --project or --variants."
        )
        raise typer.Exit(1)

    candidate = input_file(project_dir, "variants.csv")
    if not candidate.exists():
        console.print(
            f"[red]Error:[/red] no variants.csv in {project_dir}. "
            "Run 'usortm plan' first, or pass --variants."
        )
        raise typer.Exit(1)
    return candidate


def _resolve_output(output, project_dir) -> Path:
    if output is not None:
        return output
    if project_dir is not None:
        return project_dir / "skew"
    return Path("usortm_skew")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _report_resolvability(resolvability, force: bool):
    """Print the separability verdict, stopping when it is hopeless."""
    if resolvability.duplicate_groups:
        n_dup = len(resolvability.duplicate_groups)
        console.print(
            f"[yellow]⚠[/yellow] {n_dup} group(s) of variants share an identical "
            "sequence; reads cannot be attributed among them."
        )

    if resolvability.verdict == "clean":
        console.print(
            f"[green]✓[/green] Variants are separable "
            f"(nearest neighbour {resolvability.median_nn_distance:.0f} bp apart)"
        )
        console.print()
        return

    if resolvability.verdict == "marginal":
        console.print(
            f"[yellow]⚠[/yellow] {resolvability.n_below_threshold} variant(s) sit "
            f"within {resolvability.warn_below} bp of another; their reads will "
            "mostly land in the ambiguous pile."
        )
        console.print()
        return

    console.print()
    console.print(Panel(
        f"Variants are a median of [cyan]{resolvability.median_nn_distance:.0f} bp[/cyan] "
        f"apart (minimum {resolvability.min_distance} bp).\n\n"
        "At nanopore error rates individual reads cannot be attributed to "
        "variants this similar, so per-variant counts — and any skew computed "
        "from them — would be meaningless rather than merely noisy.\n\n"
        "[muted]Pass --force to measure anyway, understanding the abundances "
        "will be smeared across near-identical variants.[/muted]",
        title="[yellow]Library is not separable read-by-read[/yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))
    console.print()
    if not force:
        raise typer.Exit(1)


def _print_read_accounting(counts):
    """Show where every read went."""
    table = Table(title="Read Accounting", box=box.ROUNDED,
                  show_header=True, header_style="bold cyan")
    table.add_column("Category", style="muted")
    table.add_column("Reads", justify="right")
    table.add_column("Share", justify="right")

    total = max(1, counts.total_reads)
    rows = [
        ("Assigned to a variant", counts.assigned_reads, "green"),
        ("Ambiguous (no clear best match)", counts.ambiguous, "yellow"),
        ("Partial coverage of variant", counts.low_cov, "yellow"),
        ("Unmapped", counts.unmapped, "muted"),
    ]
    for label, value, style in rows:
        table.add_row(label, f"[{style}]{value:,}[/{style}]", f"{value / total:.1%}")
    table.add_row("[bold]Total[/bold]", f"[bold]{counts.total_reads:,}[/bold]", "")

    console.print()
    console.print(table)
    console.print()


def _print_skew_table(stats):
    """Show the measured distribution, raw next to corrected."""
    table = Table(title="Measured Library Skew", box=box.ROUNDED,
                  show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="muted")
    table.add_column("Value", justify="right")

    observed = (
        f"{stats.q90_q10_observed:.1f}×"
        if stats.q90_q10_observed is not None
        else "undefined (bottom decile saw no reads)"
    )
    ci_low, ci_high = stats.q90_q10_ci
    corrected = f"[cyan]{stats.q90_q10_corrected:.1f}×[/cyan]"
    if ci_low == ci_low and ci_high == ci_high:  # not NaN
        corrected += f" [muted](95% CI {ci_low:.1f}–{ci_high:.1f})[/muted]"

    table.add_row("Depth", f"{stats.mean_depth:.1f} reads/variant")
    table.add_row("Q90/Q10, raw", observed)
    table.add_row("Q90/Q10, noise-corrected", corrected)
    table.add_row("Effective library size", f"{stats.effective_library_size:,.0f}")
    table.add_row("Gini coefficient", f"{stats.gini:.2f}")
    table.add_row("Undetected variants", f"{stats.n_undetected:,}")
    table.add_row("Estimated dropout", f"{stats.dropout_fraction:.1%}")

    console.print(table)

    if stats.q90_q10_observed is not None:
        inflation = stats.q90_q10_observed / max(stats.q90_q10_corrected, 1e-9)
        if inflation > 1.15:
            console.print(
                f"  [muted]Counting noise inflates the raw ratio "
                f"{inflation:.1f}× at this depth.[/muted]"
            )
    if not stats.depth_sufficient:
        console.print(
            f"  [yellow]⚠[/yellow] Only {stats.mean_depth:.1f} reads per variant — "
            "treat the corrected skew as approximate."
        )
    if stats.beyond_validated_range:
        console.print(
            f"  [yellow]⚠[/yellow] Above ~10× the fit reads low (≈0.85× of truth "
            "at 16×), because too much of the library falls below one expected "
            "read. Treat this skew and the sorting depth below as lower bounds."
        )
    console.print()


_EIGHTHS = " ▏▎▍▌▋▊▉█"


def _bar(value: float, vmax: float, width: int = 22) -> str:
    """Horizontal bar with eighth-block resolution."""
    if vmax <= 0:
        return ""
    filled = max(0.0, value) / vmax * width
    whole = int(filled)
    bar = "█" * min(whole, width)
    if whole < width:
        eighth = int((filled - whole) * 8)
        if eighth > 0:
            bar += _EIGHTHS[eighth]
    return bar


def _integer_bin_label(log_lo: float, log_hi: float) -> str:
    """Label a log10 bin by the integer read counts it actually covers.

    Read counts are integers, so formatting the raw powers gives
    meaningless labels like "1-1" and "2-2" at low depth. This names the
    integers k with log_lo <= log10(k) < log_hi instead.
    """
    lo = int(math.ceil(10 ** log_lo - 1e-9))
    hi = int(math.ceil(10 ** log_hi - 1e-9)) - 1
    lo = max(lo, 1)
    if hi < lo:
        return "—"
    return f"{lo:,}" if hi == lo else f"{lo:,}–{hi:,}"


def _print_abundance_histogram(counts, stats, n_bins: int = 12):
    """Log-abundance histogram in the terminal.

    A uniform library reads as a tight, symmetric bell. The `fit` column is
    what the model expects to observe at this depth, counting noise
    included, so a column that tracks the bars means the fit describes the
    library.
    """
    from usortm.qc.skew import log10_histogram

    try:
        edges, observed, predicted, _ = log10_histogram(counts, stats, n_bins=n_bins)
    except ValueError:
        return

    table = Table(
        title="Abundance Distribution (log₁₀ reads per variant)",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
    )
    table.add_column("Reads", justify="right", style="muted")
    table.add_column("Variants", justify="left")
    table.add_column("n", justify="right")
    table.add_column("fit", justify="right", style="muted")

    vmax = max(observed.max(), predicted.max())
    for i in range(len(observed)):
        label = _integer_bin_label(edges[i], edges[i + 1])
        if label == "—" and observed[i] == 0 and predicted[i] < 0.5:
            continue  # empty bin covering no integer count; nothing to say
        table.add_row(
            label,
            f"[brand.plain]{_bar(observed[i], vmax)}[/brand.plain]",
            f"{observed[i]:,}",
            f"{predicted[i]:.0f}",
        )

    console.print(table)
    sd_log10 = stats.sigma_log / math.log(10)
    console.print(
        f"  [muted]Underlying width σ = {sd_log10:.2f} log₁₀ after removing "
        f"counting noise, i.e. {stats.q90_q10_corrected:.1f}× Q90/Q10.[/muted]",
        highlight=False,
    )
    if stats.n_undetected:
        plural = "s" if stats.n_undetected != 1 else ""
        console.print(
            f"  [muted]{stats.n_undetected:,} variant{plural} with zero reads "
            "not shown — they have no log abundance.[/muted]",
            highlight=False,
        )
    console.print()


def _print_recommendation(rec, stats, project_dir):
    """Show the recommended depth, against the plan where one exists."""
    table = Table(title="Recommended Sorting Depth", box=box.ROUNDED,
                  show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="muted")
    table.add_column("Value", justify="right")

    table.add_row("Fold sampling", f"[cyan]{rec.fold_sampling:g}×[/cyan]")
    table.add_row("Wells to sort", f"[cyan]{rec.n_wells:,}[/cyan]")
    table.add_row("384-well plates", f"{rec.n_plates}")
    table.add_row("Predicted coverage", f"[green]{rec.expected_coverage:.1%}[/green]")
    table.add_row("Target coverage", f"{rec.target_coverage:.0%}")
    if stats.dropout_fraction > 0:
        table.add_row(
            "Coverage ceiling",
            f"{rec.coverage_ceiling:.1%} [muted](set by dropouts)[/muted]",
        )
    console.print(table)

    if not rec.target_reachable:
        console.print(
            f"  [yellow]⚠[/yellow] {rec.target_coverage:.0%} is above the "
            f"{rec.coverage_ceiling:.0%} ceiling set by variants missing from the "
            "library. Sorting cannot recover what was never synthesized — "
            "resynthesize the dropouts instead."
        )

    planned = _planned_values(project_dir)
    if planned:
        console.print()
        comparison = Table(title="Plan vs. Measurement", box=box.ROUNDED,
                           show_header=True, header_style="bold cyan")
        comparison.add_column("", style="muted")
        comparison.add_column("Planned", justify="right")
        comparison.add_column("Measured", justify="right")
        if planned.get("skew") is not None:
            comparison.add_row(
                "Skew (Q90/Q10)",
                f"{planned['skew']:.1f}×",
                f"[cyan]{stats.q90_q10_corrected:.1f}×[/cyan]",
            )
        if planned.get("fold_sampling") is not None:
            comparison.add_row(
                "Fold sampling",
                f"{planned['fold_sampling']:g}×",
                f"[cyan]{rec.fold_sampling:g}×[/cyan]",
            )
        if planned.get("total_wells") is not None:
            comparison.add_row(
                "Wells", f"{planned['total_wells']:,}", f"[cyan]{rec.n_wells:,}[/cyan]"
            )
        console.print(comparison)
    console.print()


def _planned_values(project_dir) -> dict:
    """Planning-time skew and depth, when a project file is available."""
    if project_dir is None:
        return {}
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        return {}
    try:
        with open(state_file) as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "skew": state.get("skew"),
        "fold_sampling": state.get("fold_sampling"),
        "total_wells": state.get("total_wells"),
    }


def _print_next_steps(rec, project_dir):
    console.print("[bold]Next steps:[/bold]")
    console.print(
        f"  1. Sort [cyan]{rec.n_wells:,}[/cyan] wells "
        f"([cyan]{rec.n_plates}[/cyan] × 384-well plates)"
    )
    if project_dir is not None:
        console.print(
            f"  2. After sequencing, run: "
            f"[cyan]usortm demux {project_dir}/ --fastq <data.fastq>[/cyan]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def _write_outputs(profile, out_dir: Path, fastq, variants_csv, html: bool) -> list:
    """Write counts, report JSON and optional HTML. Returns paths written."""
    written = []

    counts_path = out_dir / "variant_counts.csv"
    _write_variant_counts(profile, counts_path)
    written.append(counts_path)

    report = profile.to_dict()
    report["measured"] = datetime.now().isoformat()
    report["fastq"] = str(Path(fastq).resolve())
    report["variants_file"] = str(Path(variants_csv).resolve())
    report["undetected_variants"] = profile.stats.undetected_names

    report_path = out_dir / "skew_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    written.append(report_path)

    if html:
        from usortm.qc.viz import write_skew_html

        html_path = out_dir / "skew_report.html"
        if write_skew_html(profile, html_path, title=Path(fastq).name):
            written.append(html_path)
        else:
            console.print(
                "  [muted]Skipped HTML report — install bokeh "
                "(pip install 'usortm[viz]') to enable it.[/muted]"
            )

    return written


def _write_variant_counts(profile, path: Path):
    """One row per variant: raw reads and the shrunk abundance estimate."""
    counts = profile.counts.counts
    shrunk = profile.stats.shrunk_abundance
    assigned = max(1, profile.counts.assigned_reads)

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    rank_by_name = {name: i + 1 for i, (name, _) in enumerate(ordered)}
    shrunk_by_name = dict(zip(counts.keys(), shrunk))

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Name", "reads", "observed_fraction",
            "estimated_fraction", "rank", "detected",
        ])
        for name, reads in ordered:
            writer.writerow([
                name,
                reads,
                f"{reads / assigned:.6f}",
                f"{shrunk_by_name[name]:.6f}",
                rank_by_name[name],
                "yes" if reads > 0 else "no",
            ])


def _update_project_state(project_dir: Path, profile, fastq):
    """Add a measured_skew block to the project file.

    Additive on purpose: the planning-time `skew` and `fold_sampling` stay
    untouched so the assumption and the measurement can be compared.
    """
    from usortm.qc.skew import ci_to_json

    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(
            f"[yellow]⚠[/yellow] No {PROJECT_STATE_FILE} in {project_dir}; "
            "skipping project update."
        )
        return

    with open(state_file) as fh:
        state = json.load(fh)

    stats, rec, counts = profile.stats, profile.recommendation, profile.counts
    state["measured_skew"] = {
        "measured": datetime.now().isoformat(),
        "fastq": str(Path(fastq).resolve()),
        "total_reads": counts.total_reads,
        "assigned_reads": counts.assigned_reads,
        "ambiguous_reads": counts.ambiguous,
        "unmapped_reads": counts.unmapped,
        "library_size": counts.library_size,
        "skew_observed": stats.q90_q10_observed,
        "skew_corrected": round(stats.q90_q10_corrected, 3),
        "skew_ci": ci_to_json(stats.q90_q10_ci),
        "sigma_log": round(stats.sigma_log, 4),
        "effective_library_size": round(stats.effective_library_size, 1),
        "gini": round(stats.gini, 4),
        "n_detected": stats.n_detected,
        "n_undetected": stats.n_undetected,
        "dropout_fraction": round(stats.dropout_fraction, 4),
        "coverage_ceiling": round(stats.coverage_ceiling, 4),
        "depth_sufficient": stats.depth_sufficient,
        "beyond_validated_range": stats.beyond_validated_range,
        "recommended_fold_sampling": rec.fold_sampling,
        "recommended_wells": rec.n_wells,
        "recommended_plates": rec.n_plates,
        "expected_coverage": round(rec.expected_coverage, 4),
        "target_coverage": rec.target_coverage,
        "basis": rec.basis,
    }

    with open(state_file, "w") as fh:
        json.dump(state, fh, indent=2)
