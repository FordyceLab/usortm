"""Generate final reports and plate maps from demultiplexing results."""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import csv
import json
from datetime import datetime
from collections import Counter

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"


def _norm_variant(name: str) -> str:
    """Normalize variant name: strip |cons_check suffix, unify separators."""
    return name.split("|")[0].replace(".", ";")


def _count_unique_variants(well_data: list) -> int:
    """Count unique variants, stripping ``|cons_check`` suffixes first."""
    return len(set(_norm_variant(w["variant"]) for w in well_data))


def report(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory (with demux results).",
        exists=True,
    ),
    format: str = typer.Option(
        "all",
        "--format", "-f",
        help="Output format: csv, html, json, or all",
    ),
    round_id: Optional[str] = typer.Option(
        None,
        "--round",
        help=(
            "Which round to report on: '1' (default), '2', '3', …, or 'merged'. "
            "Use 'merged' after running 'usortm merge'."
        ),
    ),
):
    """
    Generate final report and plate maps.

    Creates comprehensive reports including:
    • Interactive HTML summary
    • Plate maps (CSV)
    • Final variant mapping (CSV)
    • Missing variants list (CSV)

    [bold]Example:[/bold]

        usortm report my_project/ --format all
    """
    # Validate format
    valid_formats = ["csv", "html", "json", "all"]
    if format not in valid_formats:
        console.print(f"[red]Error:[/red] Invalid format '{format}'. Choose from: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project (missing {PROJECT_STATE_FILE})")
        console.print(f"Run 'usortm plan' first to create a project.")
        raise typer.Exit(1)

    with open(state_file) as f:
        project = json.load(f)

    # ------------------------------------------------------------------
    # Resolve round-specific context
    # ------------------------------------------------------------------
    round_id = (round_id or "1").strip().lower()

    if round_id == "merged":
        # Merged report: uses merged/well_assignments.csv (no demux_summary)
        merged_wa = project_dir / "merged" / "well_assignments.csv"
        if not merged_wa.exists():
            console.print("[red]Error:[/red] No merged results found.")
            console.print("Run 'usortm merge' first to combine rounds.")
            raise typer.Exit(1)
        well_assignments_file = merged_wa
        demux_summary: dict = {}   # no single pipeline summary for merged
        effective_project = project  # use full library_size from master state
        effective_project_dir = project_dir
        report_dir = project_dir / "merged" / "report"
        # For merged, strip the 'round' column added by merge (not expected by existing loaders)
        well_data = _load_well_assignments_merged(merged_wa)
        # Build per-round context for the HTML report
        merged_context = _build_merged_html_context(project, project_dir, well_data)
    elif round_id != "1":
        # Round N > 1
        try:
            round_num = int(round_id)
        except ValueError:
            console.print(f"[red]Error:[/red] Unknown --round value '{round_id}'. Use '1', '2', … or 'merged'.")
            raise typer.Exit(1)
        round_dir = project_dir / "rounds" / str(round_num)
        round_state_file = round_dir / "usortm_round.json"
        if not round_state_file.exists():
            console.print(f"[red]Error:[/red] Round {round_num} has not been planned.")
            raise typer.Exit(1)
        with open(round_state_file) as f:
            effective_project = json.load(f)
        demux_output = round_dir / "demux_output"
        well_assignments_file = demux_output / "well_assignments.csv"
        demux_summary_file = demux_output / "demux_summary.json"
        if not well_assignments_file.exists():
            console.print(f"[red]Error:[/red] Round {round_num} demux results not found.")
            console.print(f"Run: [cyan]usortm demux {project_dir}/ --round {round_num}[/cyan]")
            raise typer.Exit(1)
        demux_summary = json.load(open(demux_summary_file)) if demux_summary_file.exists() else {}
        well_data = _load_well_assignments(well_assignments_file)
        effective_project_dir = round_dir
        report_dir = round_dir / "report"
        merged_context = None
    else:
        # Round 1 (default)
        if "workflow_steps" not in project or not project["workflow_steps"].get("demux", {}).get("completed"):
            console.print("[red]Error:[/red] No demultiplexing results found.")
            console.print("Run 'usortm demux' first to process sequencing data.")
            raise typer.Exit(1)
        demux_output = project_dir / "demux_output"
        well_assignments_file = demux_output / "well_assignments.csv"
        demux_summary_file = demux_output / "demux_summary.json"
        if not well_assignments_file.exists() or not demux_summary_file.exists():
            console.print("[red]Error:[/red] Demux results incomplete")
            raise typer.Exit(1)
        well_data = _load_well_assignments(well_assignments_file)
        with open(demux_summary_file) as f:
            demux_summary = json.load(f)
        effective_project = project
        effective_project_dir = project_dir
        report_dir = project_dir / "report"
        merged_context = None

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Reporting",
        border_style=BORDER_STYLE,
    ))
    console.print()

    console.print(f"[green]\u2713[/green] Loaded {'merged' if round_id == 'merged' else f'round {round_id}'} results ({len(well_data)} wells with data)")

    # Create report directory
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate reports based on format
    generated_files = []

    if format in ["csv", "all"]:
        # Generate CSV reports
        plate_maps_file = report_dir / "plate_maps.csv"
        _save_plate_maps(well_data, plate_maps_file)
        generated_files.append(plate_maps_file)

        final_mapping_file = report_dir / "final_mapping.csv"
        _save_final_mapping(well_data, final_mapping_file)
        generated_files.append(final_mapping_file)

        missing_variants_file = report_dir / "missing_variants.csv"
        _save_missing_variants(effective_project, well_data, missing_variants_file, effective_project_dir)
        generated_files.append(missing_variants_file)

        library_recovery_file = report_dir / "library_recovery.csv"
        _save_library_recovery(effective_project, well_data, library_recovery_file, effective_project_dir)
        generated_files.append(library_recovery_file)

    if format in ["json", "all"]:
        # Generate JSON report
        json_file = report_dir / "report.json"
        _save_json_report(effective_project, demux_summary, well_data, json_file)
        generated_files.append(json_file)

    if format in ["html", "all"]:
        # Generate HTML report
        html_file = report_dir / "summary.html"
        _save_html_report(effective_project, demux_summary, well_data, html_file, effective_project_dir,
                          merged_context=merged_context)
        generated_files.append(html_file)

    # Display summary
    console.print()
    console.print("[green]\u2713[/green] Reports generated!")
    console.print()

    for file_path in generated_files:
        try:
            console.print(f"  \u2022 {file_path.relative_to(project_dir)}")
        except ValueError:
            console.print(f"  \u2022 {file_path}")

    console.print()

    # Display quick stats
    stats_table = Table(
        title="Workflow Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    stats_table.add_column("Metric", style="muted")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Library size", f"{effective_project.get('library_size', 'N/A')}")
    if demux_summary:
        stats_table.add_row("Input reads", f"{demux_summary.get('input_reads', demux_summary.get('total_reads', 0)):,}")
        stats_table.add_row("Wells with data", f"{demux_summary.get('wells_with_data', 0):,}")

    unique_variants = _count_unique_variants(well_data)
    stats_table.add_row("Unique variants", f"{unique_variants}")

    library_size = effective_project.get("library_size", 0)
    if library_size and library_size > 0:
        coverage_pct = min((unique_variants / library_size) * 100, 100.0)
        stats_table.add_row("Library coverage", f"{coverage_pct:.1f}%")

    console.print(stats_table)
    console.print()

    # Display Library Recovery tiers
    if library_size and library_size > 0:
        bins = _compute_quality_bins(well_data, library_size)
        tiers = bins["recovery_tiers"]

        tier_table = Table(
            title="Library Recovery",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        tier_table.add_column("Tier", style="bold")
        tier_table.add_column("Criteria", style="muted")
        tier_table.add_column("Count", justify="right")
        tier_table.add_column("% Library", justify="right")

        tier_table.add_row(
            "A", "\u226590% cons, \u2265100 reads",
            str(tiers["A"]["count"]), f"{tiers['A']['pct']:.1f}%"
        )
        tier_table.add_row(
            "B", "\u226590% cons, \u226550 reads",
            str(tiers["B"]["count"]), f"{tiers['B']['pct']:.1f}%"
        )
        tier_table.add_row(
            "C", "\u226590% cons, \u226520 reads",
            str(tiers["C"]["count"]), f"{tiers['C']['pct']:.1f}%"
        )

        console.print(tier_table)
        console.print()

    # ------------------------------------------------------------------
    # Build and save shareable zip
    # ------------------------------------------------------------------
    if round_id == "merged":
        pick_dir = project_dir / "merged" / "pick"
    elif round_id != "1":
        pick_dir = project_dir / "rounds" / round_id / "pick"
    else:
        pick_dir = project_dir / "pick"

    zip_label = "merged" if round_id == "merged" else f"round{round_id}"
    zip_path = project_dir / f"usortm_report_{zip_label}.zip"
    _make_report_zip(zip_path, generated_files, pick_dir, project_dir, round_id)
    console.print(f"[green]\u2713[/green] Shareable zip: {zip_path.name}")
    console.print()


def _make_report_zip(
    zip_path: Path,
    report_files: list,
    pick_dir: Path,
    project_dir: Path,
    round_id: str,
) -> None:
    """Bundle report files, pick list, and pileup HTMLs into a shareable zip.

    Internal zip structure::

        report/
            summary.html
            plate_maps.csv
            ...
        pick/
            hitlist_integra_assist*.csv
            README.txt
        pileup/
            well_*.html
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Report files (CSV, JSON, HTML)
        for f in report_files:
            if f.exists():
                zf.write(f, f"report/{f.name}")

        # Integra ASSIST pick list
        integra_dir = pick_dir / "Integra ASSIST Input"
        if integra_dir.exists():
            for f in integra_dir.iterdir():
                if f.is_file():
                    zf.write(f, f"pick/{f.name}")

        # Pileup HTMLs — collect from all relevant round pick dirs,
        # prefixing filenames with round number to avoid collisions.
        if round_id == "merged":
            r1_pileup = project_dir / "pick" / "pileup"
            if r1_pileup.exists():
                for f in sorted(r1_pileup.glob("*.html")):
                    zf.write(f, f"pileup/R1_{f.name}")
            rounds_root = project_dir / "rounds"
            if rounds_root.exists():
                for rdir in sorted(rounds_root.iterdir()):
                    try:
                        rnum = int(rdir.name)
                    except ValueError:
                        continue
                    rp = rdir / "pick" / "pileup"
                    if rp.exists():
                        for f in sorted(rp.glob("*.html")):
                            zf.write(f, f"pileup/R{rnum}_{f.name}")
        else:
            single_pileup = pick_dir / "pileup"
            if single_pileup.exists():
                for f in sorted(single_pileup.glob("*.html")):
                    zf.write(f, f"pileup/{f.name}")


def _load_well_assignments(assignments_file: Path) -> list:
    """Load well assignments from a per-round demux output CSV."""
    well_data = []
    with open(assignments_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            well_data.append({
                "plate": row["plate"],
                "well": row["well"],
                "variant": row["variant"],
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
                "cons_check": row.get("cons_check", ""),
                "flank_check": row.get("flank_check", ""),
            })
    return well_data


def _build_merged_html_context(project: dict, project_dir: Path, merged_well_data: list) -> dict:
    """Build per-round context dict for the merged HTML report.

    Returns a dict with:
      round_nums         - sorted list of round numbers
      total_input_reads  - sum of total_reads from each round's demux_summary
      round_well_data    - {round_num: [wells]} split from merged_well_data
      round_library_sizes- {round_num: int}
      round_demux_dirs   - {round_num: Path}  for plate maps
    """
    # Determine available rounds from project state
    round_library_sizes: dict[int, int] = {}
    round_demux_dirs: dict[int, Path] = {}
    total_input_reads = 0

    # Round 1
    if (project_dir / "demux_output" / "well_assignments.csv").exists():
        round_library_sizes[1] = project.get("library_size", 0)
        round_demux_dirs[1] = project_dir / "demux_output"
        dsf = project_dir / "demux_output" / "demux_summary.json"
        if dsf.exists():
            with open(dsf) as f:
                ds = json.load(f)
            total_input_reads += ds.get("input_reads", ds.get("total_reads", 0))

    # Rounds N > 1
    for rnum_str, rinfo in project.get("rounds", {}).items():
        rnum = int(rnum_str)
        demux_output = project_dir / rinfo.get("demux_output", f"rounds/{rnum}/demux_output")
        if (demux_output / "well_assignments.csv").exists():
            round_library_sizes[rnum] = rinfo.get("library_size", 0)
            round_demux_dirs[rnum] = demux_output
            dsf = demux_output / "demux_summary.json"
            if dsf.exists():
                with open(dsf) as f:
                    ds = json.load(f)
                total_input_reads += ds.get("input_reads", ds.get("total_reads", 0))

    round_nums = sorted(round_demux_dirs.keys())

    # Split merged_well_data by round (plate IDs are "R{n}_{plate}")
    round_well_data: dict[int, list] = {}
    for w in merged_well_data:
        pid = w["plate"]  # e.g. "R1.3" after label substitution, but raw data still "R1_3"
        # Plate prefix may already have been transformed to "R1.3" by the loader,
        # so split on both "." and "_"
        prefix = pid.split(".")[0].split("_")[0]
        try:
            rnum = int(prefix.lstrip("Rr"))
        except ValueError:
            rnum = 1
        round_well_data.setdefault(rnum, []).append(w)

    # Collect per-round demux summaries (for recovery curves, read-len histograms, etc.)
    round_demux_summaries: dict[int, dict] = {}
    for rnum, rdir in round_demux_dirs.items():
        dsf = rdir / "demux_summary.json"
        if dsf.exists():
            with open(dsf) as f:
                round_demux_summaries[rnum] = json.load(f)

    return {
        "round_nums": round_nums,
        "total_input_reads": total_input_reads,
        "round_well_data": round_well_data,
        "round_library_sizes": round_library_sizes,
        "round_demux_dirs": round_demux_dirs,
        "round_demux_summaries": round_demux_summaries,
    }


def _load_well_assignments_merged(assignments_file: Path,
                                   min_plate_reads: int = 20) -> list:
    """Load well assignments from the merged well_assignments.csv.

    The merged CSV has an extra leading 'round' column; this function
    strips it so the returned dicts match the standard format.

    Plates whose total read count is below ``min_plate_reads`` are dropped —
    they are almost certainly barcode-switching artefacts from a neighbouring
    real plate rather than genuine data.
    """
    raw: list[dict] = []
    with open(assignments_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw.append({
                "plate": f"R{row['round']}_{row['plate']}",  # prefix with round
                "well": row["well"],
                "variant": row["variant"],
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
                "cons_check": row.get("cons_check", ""),
                "flank_check": row.get("flank_check", ""),
            })

    # Compute total reads per prefixed plate and drop ghost plates
    plate_totals: dict[str, int] = {}
    for w in raw:
        plate_totals[w["plate"]] = plate_totals.get(w["plate"], 0) + w["reads"]

    return [w for w in raw if plate_totals[w["plate"]] >= min_plate_reads]


def _save_plate_maps(well_data: list, output_file: Path):
    """Save plate maps with well-to-variant assignments."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "variant", "reads", "consensus_fraction"])

        for well in well_data:
            writer.writerow([
                well["plate"],
                well["well"],
                well["variant"],
                well["reads"],
                well["consensus_fraction"],
            ])


def _save_final_mapping(well_data: list, output_file: Path):
    """Save final variant-to-well mapping."""
    # Group wells by base variant name (strip legacy |cons_check suffix)
    variant_map = {}
    for well in well_data:
        variant = well["variant"].split("|")[0]
        if variant not in variant_map:
            variant_map[variant] = []
        variant_map[variant].append(well)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "num_wells", "best_plate", "best_well", "best_reads"])

        for variant, wells in sorted(variant_map.items()):
            # Find best well (highest read count)
            best_well = max(wells, key=lambda x: x["reads"])

            writer.writerow([
                variant,
                len(wells),
                best_well["plate"],
                best_well["well"],
                best_well["reads"],
            ])


_TIER_THRESHOLDS: dict[str, dict] = {
    "A": {"min_reads": 100, "min_consensus": 0.9},
    "B": {"min_reads": 50,  "min_consensus": 0.9},
    "C": {"min_reads": 20,  "min_consensus": 0.9},
}
_TIER_ORDER = ["A", "B", "C"]  # highest to lowest


def _resolve_library_variants(project: dict, project_dir: Path = None) -> list[str]:
    """Return ordered list of variant names from the library CSV.

    Returns an empty list if the file cannot be found or read.
    """
    library_file = project.get("library_file") or project.get("variants_file")
    candidates = []
    if library_file:
        candidates.append(Path(library_file))
    if project_dir:
        candidates.append(Path(project_dir) / "variants.csv")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            names: list[str] = []
            with open(candidate, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (
                        row.get("Name")
                        or row.get("name")
                        or row.get("variant")
                        or row.get("variant_name")
                    )
                    if name:
                        names.append(name)
            return names
        except Exception:
            pass
    return []


def _best_tier(reads: int, cons_frac: float) -> str:
    """Return the highest tier letter the well qualifies for, or ''."""
    for t in _TIER_ORDER:
        th = _TIER_THRESHOLDS[t]
        if reads >= th["min_reads"] and cons_frac > th["min_consensus"]:
            return t
    return ""


def _classify_variants(
    library_names: list[str],
    well_data: list,
    pick_tier: str,
) -> list[dict]:
    """Classify every library variant as recovered / passed / missing.

    Args:
        library_names: Ordered variant names from the library CSV.
        well_data: List of well dicts with 'variant', 'reads',
            'consensus_fraction' keys.
        pick_tier: The tier used for picking (e.g. 'A', 'B', 'C').

    Returns:
        List of dicts with keys: name, tier, status.
        ``tier`` is the best tier achieved ('' if none).
        ``status`` is one of:
          - 'recovered'  — meets pick_tier threshold
          - 'passed'     — has data but below pick_tier threshold
          - 'missing'    — no wells meet any tier threshold
    """
    pick_tier = (pick_tier or "A").upper()
    pick_tier_rank = _TIER_ORDER.index(pick_tier) if pick_tier in _TIER_ORDER else 0

    # Best tier per variant across all wells (exclude mutation and flank-error wells)
    has_flank_data = any(w.get("flank_check") for w in well_data)
    best: dict[str, str] = {}
    for w in well_data:
        if w.get("cons_check", "") in ("Other Error", "Error"):
            continue
        if has_flank_data and w.get("flank_check", "") != "OK":
            continue
        name = w["variant"].split("|")[0]
        t = _best_tier(w["reads"], w["consensus_fraction"])
        if t:
            prev = best.get(name, "")
            if not prev or _TIER_ORDER.index(t) < _TIER_ORDER.index(prev):
                best[name] = t

    rows: list[dict] = []
    for name in library_names:
        t = best.get(name, "")
        if not t:
            status = "missing"
        elif _TIER_ORDER.index(t) <= pick_tier_rank:
            status = "recovered"
        else:
            status = "passed"
        rows.append({"name": name, "tier": t, "status": status})
    return rows


def _save_missing_variants(
    project: dict,
    well_data: list,
    output_file: Path,
    project_dir: Path = None,
):
    """Save list of variants not recovered at the pick tier threshold."""
    library_names = _resolve_library_variants(project, project_dir)
    pick_tier = (
        project.get("workflow_steps", {}).get("pick", {}).get("tier") or "A"
    ).upper()

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "status"])

        if not library_names:
            writer.writerow(["N/A", "No library file found in project"])
            return

        rows = _classify_variants(library_names, well_data, pick_tier)
        for r in rows:
            if r["status"] == "missing":
                writer.writerow([r["name"], "missing"])


def _save_library_recovery(
    project: dict,
    well_data: list,
    output_file: Path,
    project_dir: Path = None,
):
    """Save per-variant recovery table with tier and status columns.

    Columns: Name, Tier, Status
      - Tier   — best picking tier achieved (A/B/C), blank if none
      - Status — recovered | passed | missing
        recovered: meets the pick tier threshold
        passed:    has data but below the pick tier threshold
        missing:   no wells meet any tier threshold
    """
    library_names = _resolve_library_variants(project, project_dir)
    pick_tier = (
        project.get("workflow_steps", {}).get("pick", {}).get("tier") or "A"
    ).upper()

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Tier", "Status"])

        if not library_names:
            writer.writerow(["N/A", "", "No library file found in project"])
            return

        rows = _classify_variants(library_names, well_data, pick_tier)
        for r in rows:
            writer.writerow([r["name"], r["tier"], r["status"]])


def _save_json_report(project: dict, demux_summary: dict, well_data: list, output_file: Path):
    """Save comprehensive JSON report."""
    # Calculate statistics — strip |cons_check suffix before counting
    unique_variants = _count_unique_variants(well_data)
    stripped = [w["variant"].split("|")[0] for w in well_data]
    variant_counts = Counter(stripped)

    read_counts = [w["reads"] for w in well_data]
    avg_reads = sum(read_counts) / len(read_counts) if read_counts else 0

    library_size = project.get("library_size", 0)
    coverage_pct = round(
        min((unique_variants / library_size) * 100, 100.0), 1
    ) if library_size else None

    # Compute quality bins / recovery tiers
    bins_data = _compute_quality_bins(well_data, library_size) if library_size else None

    report = {
        "generated": datetime.now().isoformat(),
        "project": {
            "library_size": library_size,
            "seq_length": project.get("seq_length"),
            "fold_sampling": project.get("fold_sampling"),
        },
        "demux": demux_summary,
        "variants": {
            "unique": unique_variants,
            "total_wells": len(well_data),
            "avg_reads_per_well": round(avg_reads, 1),
            "variants_with_multiple_wells": sum(1 for count in variant_counts.values() if count > 1),
        },
        "coverage": {
            "library_size": library_size,
            "recovered": unique_variants,
            "percent": coverage_pct,
        },
    }

    if bins_data:
        report["quality_bins"] = bins_data["quality_bins"]
        report["recovery_tiers"] = bins_data["recovery_tiers"]

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)


def _compute_quality_bins(well_data: list, library_size: int) -> dict:
    """Classify variants into quality tiers, mirroring the pick command's logic.

    A variant qualifies for a tier if it has **at least one well** meeting the
    threshold — matching how ``usortm pick --tier A/B/C`` counts recovered variants.

    Recovery tiers are cumulative:
    - **Tier A:** ≥100 reads AND >90% consensus
    - **Tier B:** ≥50 reads AND >90% consensus  (includes Tier A)
    - **Tier C:** ≥20 reads AND >90% consensus  (includes Tier A + B)
    """
    has_flank_data = any(w.get("flank_check") for w in well_data)

    def _qualifying_variants(min_reads: int) -> set[str]:
        return {
            _norm_variant(w["variant"]) for w in well_data
            if w["reads"] >= min_reads
            and w["consensus_fraction"] > 0.9
            and w.get("cons_check", "") not in ("Other Error", "Error")
            and (not has_flank_data or w.get("flank_check", "") == "OK")
        }

    tier_a_set = _qualifying_variants(100)
    tier_b_set = _qualifying_variants(50)
    tier_c_set = _qualifying_variants(20)

    tier_a = len(tier_a_set)
    tier_b = len(tier_b_set)
    tier_c = len(tier_c_set)

    # Non-cumulative bins (for internal use / future histograms)
    bin1 = tier_a
    bin2 = len(tier_b_set - tier_a_set)
    bin3 = len(tier_c_set - tier_b_set)
    unbinned = _count_unique_variants(well_data) - tier_c

    def _pct(n: int) -> float:
        return round(n / library_size * 100, 1) if library_size else 0.0

    return {
        "quality_bins": {"bin1": bin1, "bin2": bin2, "bin3": bin3, "unbinned": unbinned},
        "recovery_tiers": {
            "A": {"count": tier_a, "pct": _pct(tier_a)},
            "B": {"count": tier_b, "pct": _pct(tier_b)},
            "C": {"count": tier_c, "pct": _pct(tier_c)},
        },
    }


def _style_figure(fig):
    """Apply consistent dashboard styling to a Bokeh figure."""
    fig.background_fill_color = None  # transparent — inherits card bg
    fig.border_fill_color = None
    fig.outline_line_color = None
    fig.toolbar_location = None  # clean look, no toolbar clutter
    fig.axis.axis_label_text_font_size = "12px"
    fig.axis.major_label_text_font_size = "11px"
    fig.axis.axis_label_text_color = "#6b7280"
    fig.axis.major_label_text_color = "#6b7280"
    fig.axis.axis_line_color = "#e5e7eb"
    fig.grid.grid_line_color = "#e5e7eb"
    fig.grid.grid_line_alpha = 0.5


def _cmap_hex(t: float) -> str:
    """Sample the custom white→yellow→green colormap at t in [0, 1].

    Mirrors the color stops used by ``get_custom_cmap()`` in demux/viz.py.
    """
    t = max(0.0, min(1.0, t))

    def _lerp(stops: list, x: float) -> float:
        for i in range(len(stops) - 1):
            x0, v0 = stops[i]
            x1, v1 = stops[i + 1]
            if x0 <= x <= x1:
                f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                return v0 + f * (v1 - v0)
        return stops[-1][1]

    r = _lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 1.0), (0.40, 0.5), (1.0, 0.0)], t)
    g = _lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 0.95), (0.40, 0.98), (1.0, 0.39)], t)
    b = _lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 0.35), (0.40, 0.6), (1.0, 0.0)], t)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _make_read_depth_bokeh(read_counts: list):
    """Build a Bokeh histogram of per-well read depths with tier threshold lines."""
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import ColumnDataSource, HoverTool, Span, Label

    if not read_counts:
        return None

    max_val = max(read_counts)
    n_bins = 25
    bin_size = max(1, (max_val + n_bins) // n_bins)

    bins = [0] * n_bins
    for r in read_counts:
        idx = min(int(r / bin_size), n_bins - 1)
        bins[idx] += 1

    plate_map_color_high = 200.0
    lefts = [i * bin_size for i in range(n_bins)]
    rights = [(i + 1) * bin_size for i in range(n_bins)]
    colors = [_cmap_hex(min(((i + 0.5) * bin_size) / plate_map_color_high, 1.0))
              for i in range(n_bins)]

    source = ColumnDataSource(data=dict(
        left=lefts, right=rights, top=bins, bottom=[0] * n_bins, color=colors,
        bin_label=[f"{l}\u2013{r}" for l, r in zip(lefts, rights)],
        count=bins,
    ))

    fig = bokeh_figure(
        width=420, height=280,
        x_axis_label="Reads per well", y_axis_label="Wells",
        sizing_mode="stretch_width",
    )
    fig.quad(left="left", right="right", top="top", bottom="bottom",
             source=source, fill_color="color", line_color="#aaaaaa", line_width=0.3)

    fig.add_tools(HoverTool(tooltips=[("Range", "@bin_label"), ("Wells", "@count")]))

    # Tier threshold lines
    for threshold, tier_label in [(20, "C"), (50, "B"), (100, "A")]:
        if threshold <= max_val:
            fig.add_layout(Span(
                location=threshold, dimension="height",
                line_color="#6b7280", line_width=1, line_dash="dashed", line_alpha=0.5,
            ))
            fig.add_layout(Label(
                x=threshold, y=max(bins) * 0.95,
                text=f" Tier {tier_label}",
                text_font_size="10px", text_color="#6b7280", text_alpha=0.7,
            ))

    _style_figure(fig)
    return fig


def _make_plate_bar_bokeh(plate_reads: dict):
    """Build a Bokeh vertical bar chart for per-plate read counts."""
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import ColumnDataSource, HoverTool, LabelSet

    if not plate_reads:
        return None

    def _fmt(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:,}"

    def _plate_sort_key(item):
        pid = str(item[0])
        try:
            return (0, int(pid), "")
        except ValueError:
            # "R{round}_{plate}" format from merged reports
            parts = pid.split("_", 1)
            try:
                round_n = int(parts[0].lstrip("Rr")) if parts[0] else 0
                plate_n = int(parts[1]) if len(parts) > 1 else 0
                return (round_n, plate_n, pid)
            except (ValueError, IndexError):
                return (0, 0, pid)

    sorted_plates = sorted(plate_reads.items(), key=_plate_sort_key)
    plates = [str(p).replace("_", ".") for p, _ in sorted_plates]
    reads = [r for _, r in sorted_plates]
    max_reads = max(reads) or 1
    colors = [_cmap_hex(r / max_reads) for r in reads]
    labels = [_fmt(r) for r in reads]

    source = ColumnDataSource(data=dict(
        plates=plates, reads=reads, color=colors, label=labels,
    ))

    fig = bokeh_figure(
        x_range=plates,
        width=420, height=280,
        x_axis_label="Plate", y_axis_label="Reads",
        sizing_mode="stretch_width",
    )
    fig.vbar(x="plates", top="reads", width=0.65, source=source,
             fill_color="color", line_color="#aaaaaa", line_width=0.3)

    fig.add_layout(LabelSet(
        x="plates", y="reads", text="label", source=source,
        text_font_size="11px", text_color="#6b7280",
        text_align="center", y_offset=4,
    ))

    fig.add_tools(HoverTool(tooltips=[("Plate", "@plates"), ("Reads", "@reads{,}")]))

    _style_figure(fig)
    fig.xgrid.grid_line_color = None
    return fig


def _make_tier_donut_bokeh(tiers: dict, library_size: int, streakout_count: int = 0):
    """Build a Bokeh donut chart showing tier + streakout + untiered breakdown."""
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import ColumnDataSource, HoverTool, Label
    import math

    if not tiers or not library_size:
        return None

    tier_a = tiers["A"]["count"]
    tier_b = tiers["B"]["count"]
    tier_c = tiers["C"]["count"]

    count_a = tier_a
    count_b = max(0, tier_b - tier_a)
    count_c = max(0, tier_c - tier_b)
    count_s = max(0, min(streakout_count, library_size - tier_c))
    count_u = max(0, library_size - tier_c - count_s)

    total = count_a + count_b + count_c + count_s + count_u
    if total == 0:
        return None

    segments = [
        (count_a, _cmap_hex(0.75), "Tier A (\u2265100 reads)"),
        (count_b, _cmap_hex(0.35), "Tier B (50\u201399 reads)"),
        (count_c, _cmap_hex(0.20), "Tier C (20\u201349 reads)"),
        (count_s, "#3B82F6",        "Streakout recoverable"),
        (count_u, "#d1d5db",        "Unrecovered"),
    ]

    starts, ends, colors, labels, counts, pcts = [], [], [], [], [], []
    angle = math.pi / 2  # start at 12 o'clock
    for count, color, label in segments:
        if count == 0:
            continue
        frac = count / total
        end = angle - frac * 2 * math.pi
        starts.append(angle)
        ends.append(end)
        colors.append(color)
        labels.append(label)
        counts.append(count)
        pcts.append(f"{frac * 100:.1f}%")
        angle = end

    source = ColumnDataSource(data=dict(
        start=starts, end=ends, color=colors,
        label=labels, count=counts, pct=pcts,
    ))

    fig = bokeh_figure(
        width=200, height=200,
        x_range=(-1.3, 1.3), y_range=(-1.3, 1.3),
        sizing_mode="fixed",
        match_aspect=True,
    )
    fig.annular_wedge(
        x=0, y=0, inner_radius=0.55, outer_radius=1.0,
        start_angle="start", end_angle="end",
        fill_color="color", line_color="white", line_width=1.5,
        source=source, direction="clock",
    )

    # Center text: tier C count (streakout shown as separate segment)
    center_count = tier_c
    fig.add_layout(Label(
        x=0, y=0.08, text=str(center_count),
        text_font_size="18px", text_font_style="bold",
        text_color="#6b7280", text_align="center", text_baseline="middle",
    ))
    fig.add_layout(Label(
        x=0, y=-0.18, text="recovered",
        text_font_size="9px", text_color="#6b7280",
        text_align="center", text_baseline="middle",
    ))

    fig.add_tools(HoverTool(tooltips=[
        ("", "@label"), ("Count", "@count{,}"), ("", "@pct"),
    ]))

    _style_figure(fig)
    fig.axis.visible = False
    fig.grid.visible = False
    return fig


def _make_read_length_bokeh(hist_data: dict):
    """Build a Bokeh histogram of input read lengths with peak/median markers.

    Args:
        hist_data: Dict with bin_size, counts (list of 50 ints), median, n_reads.
    """
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import ColumnDataSource, HoverTool, Label

    if not hist_data or not hist_data.get("counts"):
        return None

    counts = hist_data["counts"]
    bin_size = hist_data.get("bin_size", 1)
    median_bp = hist_data.get("median", 0)
    n_bins = len(counts)
    max_count = max(counts) if any(counts) else 1

    ref_len = 500.0
    lefts = [i * bin_size for i in range(n_bins)]
    rights = [(i + 1) * bin_size for i in range(n_bins)]
    colors = [_cmap_hex(min(((i + 0.5) * bin_size) / ref_len, 1.0))
              for i in range(n_bins)]

    source = ColumnDataSource(data=dict(
        left=lefts, right=rights, top=counts, bottom=[0] * n_bins, color=colors,
        bin_label=[f"{l}\u2013{r} bp" for l, r in zip(lefts, rights)],
        count=counts,
    ))

    fig = bokeh_figure(
        width=420, height=280,
        x_axis_label="Read Length (bp)", y_axis_label="Reads",
        sizing_mode="stretch_width",
    )
    fig.quad(left="left", right="right", top="top", bottom="bottom",
             source=source, fill_color="color", line_color="#aaaaaa", line_width=0.3)

    fig.add_tools(HoverTool(tooltips=[("Range", "@bin_label"), ("Count", "@count{,}")]))

    # Peak annotations
    threshold_5pct = max_count * 0.05
    peaks: list[tuple[int, int]] = []
    for i in range(1, n_bins - 1):
        if (counts[i] > counts[i - 1] and
                counts[i] > counts[i + 1] and
                counts[i] - counts[i - 1] > threshold_5pct and
                counts[i] - counts[i + 1] > threshold_5pct):
            peaks.append((i, counts[i]))

    median_bin = min(int(median_bp / bin_size), n_bins - 1) if median_bp > 0 else -100
    peaks_sorted = sorted(peaks, key=lambda p: p[1], reverse=True)[:2]
    peaks_to_annotate = [
        (b, c) for b, c in peaks_sorted if abs(b - median_bin) > 3
    ]
    for peak_bin, peak_count in peaks_to_annotate:
        peak_bp = int((peak_bin + 0.5) * bin_size)
        fig.scatter(
            x=[peak_bp], y=[peak_count * 1.06], size=8,
            marker="inverted_triangle", color="#6b7280", alpha=0.6,
        )
        fig.add_layout(Label(
            x=peak_bp, y=peak_count * 1.10, text=f"{peak_bp}bp",
            text_font_size="9px", text_color="#6b7280", text_alpha=0.7,
            text_align="center",
        ))

    # Median marker
    if median_bp > 0:
        med_count = counts[min(int(median_bp / bin_size), n_bins - 1)]
        fig.scatter(
            x=[median_bp], y=[med_count * 1.06], size=8,
            marker="inverted_triangle", color="#ef4444", alpha=0.85,
        )
        fig.add_layout(Label(
            x=median_bp, y=med_count * 1.10, text=f"med {median_bp}bp",
            text_font_size="9px", text_color="#ef4444", text_align="center",
        ))

    _style_figure(fig)
    return fig


def _compute_read_len_hist(fastq_path: str) -> dict:
    """Compute a 50-bin read-length histogram from a FASTQ file (stdlib only).

    Used as a fallback when the pipeline did not store ``read_len_hist`` in
    ``demux_summary.json`` (e.g. projects demuxed before this feature was added).

    Returns a dict with keys bin_size, counts, median, n_reads, or {} on error.
    """
    import gzip as _gz
    import statistics as _stats

    open_fn = _gz.open if str(fastq_path).endswith(".gz") else open
    lengths: list[int] = []
    try:
        with open_fn(fastq_path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % 4 == 1:
                    lengths.append(len(line.rstrip()))
    except Exception:
        return {}
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
        "median": int(_stats.median(lengths)),
        "n_reads": len(lengths),
    }


def _make_recovery_curve_bokeh(
    curve_data: dict,
    true_sampling: Optional[float],
    tier_c_pct: Optional[float],
    round_n: int = 1,
    streakout_pct: Optional[float] = None,
    merged_point: Optional[dict] = None,
):
    """Build a Bokeh recovery curve with confidence ribbon.

    Args:
        curve_data:    Dict with fold_samplings, coverage_means, coverage_stds.
        true_sampling: Actual fold sampling (x-position of the real point).
        tier_c_pct:    Actual Tier-C coverage % (y-position), or None.
        round_n:       Sort round number (shown in corner label).
        streakout_pct: Coverage % including streak-out recoverable variants.
        merged_point:  Optional dict with ``sampling``, ``tier_c_pct``, and
                       ``streakout_pct`` for the merged result. When given, the
                       merged points are drawn and connected to the round-1 points.
    """
    from bokeh.plotting import figure as bokeh_figure
    from bokeh.models import (
        ColumnDataSource, HoverTool, Band, Label, Span,
        Legend, LegendItem,
    )

    fold_samplings = curve_data.get("fold_samplings", [])
    coverage_means = curve_data.get("coverage_means", [])
    coverage_stds = curve_data.get("coverage_stds", [])

    if not fold_samplings or not coverage_means:
        return None

    # Prepend origin so the curve starts at (0, 0)
    fs = [0.0] + list(fold_samplings)
    means = [0.0] + list(coverage_means)
    stds = [0.0] + list(coverage_stds)

    upper = [min(m + s, 100) for m, s in zip(means, stds)]
    lower = [max(m - s, 0) for m, s in zip(means, stds)]

    source = ColumnDataSource(data=dict(
        x=fs, y=means, upper=upper, lower=lower,
    ))

    fig = bokeh_figure(
        width=420, height=300,
        x_axis_label="Fold Sampling", y_axis_label="% Recovered",
        sizing_mode="stretch_width",
    )

    # Confidence ribbon
    fig.add_layout(Band(
        base="x", lower="lower", upper="upper", source=source,
        fill_color="#2563eb", fill_alpha=0.15, line_color=None,
    ))

    # Mean curve
    line_r = fig.line("x", "y", source=source,
                      line_color="#2563eb", line_width=2)

    fig.add_tools(HoverTool(
        renderers=[line_r],
        tooltips=[("Fold sampling", "@x{0.1f}"), ("Coverage", "@y{0.1f}%")],
    ))

    legend_items = [LegendItem(label="Simulated mean \u00b1 1\u03c3", renderers=[line_r])]

    # Observed data point(s) — render streakout first so green dot is on top
    if true_sampling is not None and tier_c_pct is not None:
        # --- Connector lines from round 1 → merged (drawn first, behind dots) ---
        if merged_point:
            m_sampling = merged_point.get("sampling")
            m_tc = merged_point.get("tier_c_pct")
            m_so = merged_point.get("streakout_pct")
            # Connect round-1 observed → merged observed (green dashed)
            if m_sampling is not None and m_tc is not None:
                fig.segment(
                    x0=[true_sampling], y0=[tier_c_pct],
                    x1=[m_sampling], y1=[m_tc],
                    line_color="#22c55e", line_width=1.5, line_dash="dashed",
                )
            # Connect round-1 streakout → merged streakout (blue dashed)
            if (streakout_pct is not None and m_sampling is not None
                    and m_so is not None and m_so > (m_tc or 0)):
                fig.segment(
                    x0=[true_sampling], y0=[streakout_pct],
                    x1=[m_sampling], y1=[m_so],
                    line_color="#2563eb", line_width=1.5, line_dash="dashed",
                )

        # --- Round-1 vertical streakout connector ---
        _r1_alpha = 0.4 if merged_point else 1.0
        if streakout_pct is not None and streakout_pct > tier_c_pct:
            so_r = fig.scatter(
                [true_sampling], [streakout_pct], size=10, color="#2563eb",
                alpha=_r1_alpha,
            )
            fig.segment(
                x0=[true_sampling], y0=[tier_c_pct],
                x1=[true_sampling], y1=[streakout_pct],
                line_color="#2563eb", line_width=1.5, line_dash="dashed",
                line_alpha=_r1_alpha,
            )

        # --- Round-1 observed dot (green, on top) ---
        obs_r = fig.scatter(
            [true_sampling], [tier_c_pct], size=10, color="#22c55e",
            alpha=_r1_alpha,
        )
        _r1_label = "R1 observed" if merged_point else "Observed"
        legend_items.append(LegendItem(
            label=f"{_r1_label} ({tier_c_pct:.1f}%)", renderers=[obs_r],
        ))

        if streakout_pct is not None and streakout_pct > tier_c_pct:
            legend_items.append(LegendItem(
                label=f"+ streak-out ({streakout_pct:.1f}%)", renderers=[so_r],
            ))

        # --- Merged points ---
        _merged_legend_start = len(legend_items)
        if merged_point:
            m_sampling = merged_point.get("sampling")
            m_tc = merged_point.get("tier_c_pct")
            m_so = merged_point.get("streakout_pct")

            if m_sampling is not None and m_tc is not None:
                # Merged streakout dot + vertical connector (behind green)
                if m_so is not None and m_so > m_tc:
                    m_so_r = fig.scatter(
                        [m_sampling], [m_so], size=10, color="#2563eb",
                    )
                    fig.segment(
                        x0=[m_sampling], y0=[m_tc],
                        x1=[m_sampling], y1=[m_so],
                        line_color="#2563eb", line_width=1.5, line_dash="dashed",
                    )

                # Merged observed dot (green circle, on top)
                m_obs_r = fig.scatter(
                    [m_sampling], [m_tc], size=10, color="#22c55e",
                )
                legend_items.append(LegendItem(
                    label=f"Merged ({m_tc:.1f}%)", renderers=[m_obs_r],
                ))

                if m_so is not None and m_so > m_tc:
                    legend_items.append(LegendItem(
                        label=f"+ streak-out ({m_so:.1f}%)", renderers=[m_so_r],
                    ))

    elif true_sampling is not None:
        fig.add_layout(Span(
            location=true_sampling, dimension="height",
            line_color="#6b7280", line_width=1.5, line_dash="dashed",
            line_alpha=0.6,
        ))

    # Round label
    _label_text = "Merged" if merged_point else f"Round {round_n}"
    fig.add_layout(Label(
        x=70, y=10, text=_label_text,
        text_font_size="11px", text_color="#6b7280",
        x_units="screen", y_units="screen",
    ))

    # Row 1: simulation line
    # Row 2: R1 observed + streak-out
    # Row 3: Merged observed + streak-out
    _legend_kwargs = dict(
        location="bottom_center", orientation="horizontal",
        label_text_font_size="10px", label_text_color="#6b7280",
        border_line_color=None, background_fill_alpha=0,
        margin=0, padding=2,
    )
    fig.add_layout(Legend(items=legend_items[:1], **_legend_kwargs), "below")
    if merged_point and len(legend_items) > 1:
        r1_items = legend_items[1:_merged_legend_start]
        merged_items = legend_items[_merged_legend_start:]
        if r1_items:
            fig.add_layout(Legend(items=r1_items, **_legend_kwargs), "below")
        if merged_items:
            fig.add_layout(Legend(items=merged_items, **_legend_kwargs), "below")
    elif len(legend_items) > 1:
        fig.add_layout(Legend(items=legend_items[1:], **_legend_kwargs), "below")

    _style_figure(fig)
    fig.y_range.start = 0
    fig.y_range.end = 105
    return fig


def _save_html_report(project: dict, demux_summary: dict, well_data: list,
                      output_file: Path, project_dir: Path = None,
                      merged_context: dict = None):
    """Save interactive HTML summary report with embedded plate maps."""
    import html as _html
    import os as _os

    # Calculate statistics — strip |cons_check suffix before counting
    unique_variants = _count_unique_variants(well_data)

    read_counts = [w["reads"] for w in well_data]
    avg_reads = sum(read_counts) / len(read_counts) if read_counts else 0
    max_reads = max(read_counts) if read_counts else 0
    wells_gt_20 = sum(1 for r in read_counts if r > 20)

    library_size = project.get("library_size", 0)
    coverage_pct = min((unique_variants / library_size) * 100, 100.0) if library_size else 0
    true_sampling = (wells_gt_20 / library_size) if library_size else None
    true_sampling_display = f"{true_sampling:.1f} fold" if true_sampling is not None else "N/A"
    true_sampling_note = f"{wells_gt_20:,} wells with &gt;20 reads"

    # Merged-report overrides for header chips
    round_label = "Merged" if merged_context else str(project.get("round", 1))
    input_reads = (
        merged_context["total_input_reads"] if merged_context
        else demux_summary.get("input_reads", demux_summary.get("total_reads", 0))
    )

    # Quality bins / recovery tiers
    bins_data = _compute_quality_bins(well_data, library_size) if library_size else None
    tiers = bins_data["recovery_tiers"] if bins_data else None
    qbins = bins_data["quality_bins"] if bins_data else None

    # Recovery curve (pre-computed during demux, cached in demux_summary).
    # For merged reports, build one curve per round from round_demux_summaries.
    recovery_fig = None

    def _build_recovery_fig(ds: dict, wd: list, lib_size: int, round_n: int):
        """Build a recovery curve figure from one round's data."""
        rc_data = ds.get("recovery_curve")
        if not rc_data or not lib_size:
            return None
        rbins = _compute_quality_bins(wd, lib_size)
        rt = rbins["recovery_tiers"] if rbins else None
        tc_pct = rt["C"]["pct"] if rt else None
        # Fold sampling for this round
        rc_reads = [w["reads"] for w in wd]
        rc_wells_gt_20 = sum(1 for r in rc_reads if r > 20)
        rc_sampling = (rc_wells_gt_20 / lib_size) if lib_size else None
        # Streakout coverage
        so_pct = None
        so_data = ds.get("streakout", {})
        so_vars = so_data.get("recoverable_variants", [])
        if so_vars and rt and lib_size:
            tc_count = rt["C"]["count"]
            tc_variants = {
                _norm_variant(w["variant"]) for w in wd
                if w["reads"] >= 20 and w["consensus_fraction"] > 0.9
            }
            new_v = len({_norm_variant(v) for v in so_vars} - tc_variants)
            so_pct = min((tc_count + new_v) / lib_size * 100, 100.0)
        return _make_recovery_curve_bokeh(
            rc_data, rc_sampling, tc_pct, round_n=round_n, streakout_pct=so_pct,
        )

    if merged_context:
        # Build single recovery curve: round 1 simulation + both R1 and merged points
        # Use round 1's simulation curve data as the base
        r1_num = merged_context["round_nums"][0]
        r1_ds = merged_context.get("round_demux_summaries", {}).get(r1_num, {})
        r1_wd = merged_context["round_well_data"].get(r1_num, [])
        r1_lib = merged_context["round_library_sizes"].get(r1_num, 0)
        r1_curve = r1_ds.get("recovery_curve")

        if r1_curve and r1_lib:
            # Round 1 observed stats
            r1_bins = _compute_quality_bins(r1_wd, r1_lib)
            r1_tiers = r1_bins["recovery_tiers"] if r1_bins else None
            r1_tc_pct = r1_tiers["C"]["pct"] if r1_tiers else None
            r1_reads = [w["reads"] for w in r1_wd]
            r1_wells_gt_20 = sum(1 for r in r1_reads if r > 20)
            r1_sampling = (r1_wells_gt_20 / r1_lib) if r1_lib else None

            # Round 1 streakout
            r1_so_pct = None
            r1_so_data = r1_ds.get("streakout", {})
            r1_so_vars = r1_so_data.get("recoverable_variants", [])
            if r1_so_vars and r1_tiers and r1_lib:
                r1_tc_count = r1_tiers["C"]["count"]
                r1_tc_variants = {
                    _norm_variant(w["variant"]) for w in r1_wd
                    if w["reads"] >= 20 and w["consensus_fraction"] > 0.9
                }
                r1_new_v = len({_norm_variant(v) for v in r1_so_vars} - r1_tc_variants)
                r1_so_pct = min((r1_tc_count + r1_new_v) / r1_lib * 100, 100.0)

            # Merged observed stats (use the already-computed merged tiers/sampling)
            merged_tc_pct = tiers["C"]["pct"] if tiers else None
            merged_so_pct = None
            # Use actual streakout picks from project state for consistency
            if tiers and library_size:
                merged_tc_count = tiers["C"]["count"]
                _m_pick = project.get("merged", {}) if merged_context else project.get("workflow_steps", {}).get("pick", {})
                _m_so_n = _m_pick.get("streakout_variants", 0)
                if _m_so_n:
                    merged_so_pct = min((merged_tc_count + _m_so_n) / library_size * 100, 100.0)

            recovery_fig = _make_recovery_curve_bokeh(
                r1_curve, r1_sampling, r1_tc_pct,
                round_n=r1_num, streakout_pct=r1_so_pct,
                merged_point={
                    "sampling": true_sampling,
                    "tier_c_pct": merged_tc_pct,
                    "streakout_pct": merged_so_pct,
                },
            )
    else:
        recovery_curve_data = demux_summary.get("recovery_curve")
        if recovery_curve_data:
            tier_c_pct = tiers["C"]["pct"] if tiers else None
            streakout_pct = None
            streakout_data_rc = demux_summary.get("streakout", {})
            so_variants = streakout_data_rc.get("recoverable_variants", [])
            if so_variants and tiers and library_size:
                tier_c_count = tiers["C"]["count"]
                tier_c_variants = {
                    _norm_variant(w["variant"]) for w in well_data
                    if w["reads"] >= 20 and w["consensus_fraction"] > 0.9
                }
                new_variants = len({_norm_variant(v) for v in so_variants} - tier_c_variants)
                streakout_pct = min((tier_c_count + new_variants) / library_size * 100, 100.0)
            recovery_fig = _make_recovery_curve_bokeh(
                recovery_curve_data, true_sampling, tier_c_pct,
                round_n=project.get("round", 1),
                streakout_pct=streakout_pct,
            )

    # Per-plate read totals for bar chart — only show plates with ≥250 reads
    _MIN_PLATE_READS = 250
    plate_reads: dict[str, int] = {}
    for w in well_data:
        p = w["plate"]
        plate_reads[p] = plate_reads.get(p, 0) + w["reads"]
    plate_reads = {p: r for p, r in plate_reads.items() if r >= _MIN_PLATE_READS}

    # Read length histogram — backfill from FASTQ if not cached in demux_summary
    if not demux_summary.get("read_len_hist") and project_dir:
        _fq_candidates = [
            project_dir / "demux_output" / "subsampled.fastq",
            project_dir / "demux_output" / "combined.fastq",
        ]
        for _fq in _fq_candidates:
            if _fq.exists():
                try:
                    _hist = _compute_read_len_hist(str(_fq))
                    if _hist:
                        demux_summary["read_len_hist"] = _hist
                        _dsf = project_dir / "demux_output" / "demux_summary.json"
                        with open(_dsf, "w") as _f:
                            json.dump(demux_summary, _f, indent=2)
                except Exception:
                    pass
                break

    # Resolve pick state early — needed for the donut chart streakout segment
    if merged_context:
        pick_state = project.get("merged", {})
    else:
        pick_state = project.get("workflow_steps", {}).get("pick", {})

    # Build Bokeh figures for all charts
    depth_fig = _make_read_depth_bokeh(read_counts)
    plate_fig = _make_plate_bar_bokeh(plate_reads)
    read_len_fig = _make_read_length_bokeh(demux_summary.get("read_len_hist") or {})
    _streakout_count = pick_state.get("streakout_variants", 0) if pick_state else 0
    donut_fig = _make_tier_donut_bokeh(tiers, library_size, _streakout_count) if tiers else None

    # Collect non-None figures and generate Bokeh components
    from bokeh.embed import components as bokeh_components
    from bokeh.resources import INLINE as BOKEH_INLINE

    _chart_figs: dict = {}
    for _key, _fig in [("depth", depth_fig), ("plate", plate_fig),
                        ("read_len", read_len_fig), ("donut", donut_fig),
                        ("recovery", recovery_fig)]:
        if _fig is not None:
            _chart_figs[_key] = _fig
    if _chart_figs:
        _keys = list(_chart_figs.keys())
        _bokeh_script, _bokeh_div_list = bokeh_components(
            [_chart_figs[k] for k in _keys]
        )
        _bokeh_divs = dict(zip(_keys, _bokeh_div_list))
    else:
        _bokeh_script = ""
        _bokeh_divs = {}

    _bokeh_js = BOKEH_INLINE.render_js() if _chart_figs else ""
    _bokeh_css = BOKEH_INLINE.render_css() if _chart_figs else ""

    # Build chart HTML from Bokeh divs
    read_depth_div = _bokeh_divs.get("depth", "")
    plate_bar_div = _bokeh_divs.get("plate", "")
    read_len_div = _bokeh_divs.get("read_len", "")
    donut_div = _bokeh_divs.get("donut", "")
    recovery_div = _bokeh_divs.get("recovery", "")

    if recovery_div:
        recovery_curve_html = (
            f'<div class="chart-card"><h3>Recovery Curve</h3>'
            f'{recovery_div}'
            f'</div>'
        )
    else:
        recovery_curve_html = ""

    if read_len_div:
        _n_reads = (demux_summary.get("read_len_hist") or {}).get("n_reads", 0)
        _n_reads_note = f" ({_n_reads:,} reads)" if _n_reads else ""
        read_len_col_html = (
            f'        <div class="chart-card">\n'
            f'            <h3>Read Length Distribution{_n_reads_note}</h3>\n'
            f'            <p class="note" style="margin:0 0 0.5rem;">Peaks labeled &#9660;. '
            f'Median in red &#9660;.</p>\n'
            f'            <div class="read-len-chart">{read_len_div}</div>\n'
            f'        </div>'
        )
    else:
        read_len_col_html = ""

    # Sequence length display — use measured range from demux, fall back to plan value
    sl_min = demux_summary.get("seq_len_min")
    sl_max = demux_summary.get("seq_len_max")
    if sl_min is not None and sl_max is not None:
        if sl_min == sl_max:
            seq_len_display = f"{sl_min} bp"
        else:
            seq_len_display = f"{sl_min}\u2013{sl_max} bp"
    else:
        seq_len_display = f"{project.get('seq_length', 'N/A')} bp"

    # Unified Library Recovery section (merges coverage + quality tiers)
    # pick_state was already resolved above (merged vs round-1); don't overwrite
    if not merged_context:
        pick_state = project.get("workflow_steps", {}).get("pick", {})
    selected_tier = (
        str(pick_state.get("tier", "")).upper()
        if pick_state.get("completed") and pick_state.get("tier")
        else ""
    )
    if selected_tier not in {"A", "B", "C"}:
        selected_tier = ""

    def _tier_box(tier_key: str, label: str, count: int, pct: float) -> str:
        selected = tier_key == selected_tier
        selected_class = " selected-tier" if selected else ""
        selected_badge = (
            '<div class="selected-tier-label">Selected tier</div>'
            if selected else ""
        )
        return (
            f'<div class="stat-box{selected_class}">'
            f'<div class="stat-label">{label}</div>'
            f'{selected_badge}'
            f'<div class="stat-value success">{count}</div>'
            f'<div class="stat-sub">{pct:.1f}% of library</div>'
            f'</div>'
        )

    tier_boxes = ""
    if tiers:
        tier_boxes = (
            f"\n        {_tier_box('A', 'Tier A (≥100 reads)', tiers['A']['count'], tiers['A']['pct'])}"
            f"\n        {_tier_box('B', 'Tier B (≥50 reads)', tiers['B']['count'], tiers['B']['pct'])}"
            f"\n        {_tier_box('C', 'Tier C (≥20 reads)', tiers['C']['count'], tiers['C']['pct'])}"
        )

    tier_note = (
        '<p class="note">All tiers require &gt;90% consensus and no mutations. Only unique variants '
        '(best well per variant). Tiers are cumulative (B includes A, C includes B).</p>'
        if tiers else ""
    )

    # Embed plate maps via srcdoc (inline HTML) so they render correctly
    # when summary.html is opened as a local file:// URL.  Browsers block
    # cross-origin iframe src for file:// origins.
    plate_map_section = ""
    pick_plate_iframe = ""

    def _embed_srcdoc(html_path: Path, height: int = 620) -> str:
        """Embed a local HTML file via srcdoc so it renders on file:// pages.

        Browsers block <iframe src="file://..."> on file:// parent pages (Safari,
        hardened Chrome). srcdoc embeds the content inline, bypassing that restriction.
        Relative window.open() URLs in Bokeh data (streakout links) are replaced with
        absolute file:// paths before embedding so tap-tool navigation still works.
        """
        content = html_path.read_text(encoding="utf-8")

        # Absolutize relative URLs baked into Bokeh JS data so tap-tool
        # navigation works when the content is embedded via srcdoc (the
        # srcdoc iframe inherits the parent page's base URL, not the
        # original file's directory).
        base_uri = html_path.parent.resolve().as_uri()
        for rel_prefix in ("streakout/", "pileup/", "mutation/pileup/"):
            content = content.replace(f'"{rel_prefix}', f'"{base_uri}/{rel_prefix}')

        # Escape for double-quoted HTML attribute value
        content = content.replace("&", "&amp;").replace('"', "&quot;")

        # Compute correct relative path from the report file to the embedded file,
        # regardless of how deep the report is nested.
        return (
            f'<iframe srcdoc="{content}" width="100%" height="{height}" '
            f'style="border:none;"></iframe>'
        )

    if merged_context:
        # Show each round's existing demux plate map (with mutation/streakout labels).
        # Use plate_map.html (generated by demux with overlay data) — the
        # plate_map_filtered.html variant may lack mutation/streakout overlays.
        round_map_parts = []
        for rnum in merged_context["round_nums"]:
            rdir = merged_context["round_demux_dirs"].get(rnum)
            if not rdir:
                continue
            pm = rdir / "plate_map.html"
            if pm.exists():
                round_map_parts.append(
                    f"<h3>Round {rnum} Demux Plate Map</h3>"
                    f"<p>Interactive plate map showing per-well read depth and variant composition.</p>"
                    f"{_embed_srcdoc(pm, 620)}"
                )
        if round_map_parts:
            plate_map_section = (
                "\n    <h2>Demux Plate Maps</h2>\n    "
                + "\n    ".join(round_map_parts)
                + "\n"
            )
        # Merged pick plate map (generated by merge command)
        merged_pick_map = project_dir / "merged" / "pick" / "pick_plate_map.html"
        if merged_pick_map.exists():
            pick_plate_iframe = _embed_srcdoc(merged_pick_map, 780)
    elif project_dir:
        demux_plate_map = project_dir / "demux_output" / "plate_map.html"
        if demux_plate_map.exists():
            plate_map_section = f"""
    <h2>Demux Plate Map</h2>
    <p>Interactive plate map showing per-well read depth and variant composition.</p>
    {_embed_srcdoc(demux_plate_map, 620)}
"""

        pick_plate_map = project_dir / "pick" / "pick_plate_map.html"
        if not pick_plate_map.exists():
            pick_plate_map = project_dir / "pick_plate_map.html"  # backward compat
        if pick_plate_map.exists():
            pick_plate_iframe = _embed_srcdoc(pick_plate_map, 780)

    # Pick summary stat box
    pick_stat_box = ""
    if pick_state.get("completed"):
        _tier_val = pick_state.get("tier", "") or ""
        _tier_label = f"Tier {_tier_val} Variants Picked" if _tier_val and _tier_val != "none" else "Unique Variants Picked"

        if merged_context and tiers:
            _streakout_n = pick_state.get("streakout_variants", 0)
            _total_picked = pick_state.get("unique_variants", 0)
            _regular_n = _total_picked - _streakout_n
        else:
            _streakout_n = pick_state.get("streakout_variants", 0)
            _regular_n = pick_state.get("unique_variants", "N/A")

        streakout_box = (
            f'<div class="stat-box" style="margin-top:0.6rem;border-left:3px solid #3B82F6;">'
            f'<div class="stat-label">Additional from Streakout</div>'
            f'<div class="stat-value" style="color:#3B82F6;">{_streakout_n}</div>'
            f'</div>'
            if _streakout_n else ""
        )
        pick_stat_box = (
            f'<div class="stat-box" style="margin-top:1rem;border-left:3px solid #22c55e;">'
            f'<div class="stat-label">{_tier_label}</div>'
            f'<div class="stat-value success">{_regular_n}</div>'
            f'</div>'
            f'{streakout_box}'
        )

    # Library Recovery section
    if merged_context and tiers:
        # Per-round table + merged summary row
        def _tier_cell(count: int, pct: float) -> str:
            return f'<td>{count} <span style="color:var(--muted);font-size:0.85rem;">({pct:.1f}%)</span></td>'

        round_rows = ""
        for rnum in merged_context["round_nums"]:
            rsize = merged_context["round_library_sizes"].get(rnum, 0)
            rwells = merged_context["round_well_data"].get(rnum, [])
            if rwells and rsize:
                rbins = _compute_quality_bins(rwells, rsize)
                rt = rbins["recovery_tiers"]
                round_rows += (
                    f'<tr>'
                    f'<td>Round {rnum} <span style="color:var(--muted);font-size:0.85rem;">(n={rsize})</span></td>'
                    f'{_tier_cell(rt["A"]["count"], rt["A"]["pct"])}'
                    f'{_tier_cell(rt["B"]["count"], rt["B"]["pct"])}'
                    f'{_tier_cell(rt["C"]["count"], rt["C"]["pct"])}'
                    f'</tr>'
                )

        merged_row = (
            f'<tr class="merged-recovery-row">'
            f'<td><strong>Merged</strong> <span style="color:var(--muted);font-size:0.85rem;">(n={library_size})</span></td>'
            f'{_tier_cell(tiers["A"]["count"], tiers["A"]["pct"])}'
            f'{_tier_cell(tiers["B"]["count"], tiers["B"]["pct"])}'
            f'{_tier_cell(tiers["C"]["count"], tiers["C"]["pct"])}'
            f'</tr>'
        )

        _sel_tier = (pick_state.get("tier") or "C").upper()
        _sel_count = tiers[_sel_tier]["count"] if _sel_tier in tiers else tiers["C"]["count"]
        _sel_pct = tiers[_sel_tier]["pct"] if _sel_tier in tiers else tiers["C"]["pct"]

        _recovery_hero = (
            f'<div style="border:2px solid #22c55e;border-radius:10px;padding:1rem 1.5rem;'
            f'text-align:center;margin-bottom:1rem;">'
            f'<div style="color:var(--muted);font-size:0.85rem;margin-bottom:0.25rem;">'
            f'Tier {_sel_tier} Variants Recovered</div>'
            f'<div style="color:#16a34a;font-size:2.5rem;font-weight:700;line-height:1;">'
            f'{_sel_count}</div>'
            f'<div style="color:var(--muted);font-size:0.85rem;margin-top:0.25rem;">'
            f'{_sel_pct:.1f}% of {library_size} library variants</div>'
            f'</div>'
        )

        library_section = f"""
    <h2>Library Recovery</h2>
    <div class="recovery-row">
    <div class="recovery-tiers">
    {_recovery_hero}
    <p class="note">All tiers require &gt;90% consensus and no mutations. Only unique variants per round.
    Tiers are cumulative (B includes A, C includes B).</p>
    <table style="margin-bottom:1rem;">
      <thead>
        <tr>
          <th>Round</th>
          <th>Tier A &nbsp;&geq;100 reads</th>
          <th>Tier B &nbsp;&geq;50 reads</th>
          <th>Tier C &nbsp;&geq;20 reads</th>
        </tr>
      </thead>
      <tbody>
        {round_rows}
        {merged_row}
      </tbody>
    </table>
    </div>
    {recovery_curve_html}
    </div>
"""
    elif tier_boxes:
        _sel_tier_s = (pick_state.get("tier") or "C").upper() if pick_state else "C"
        _sel_count_s = tiers[_sel_tier_s]["count"] if tiers and _sel_tier_s in tiers else 0
        _sel_pct_s = tiers[_sel_tier_s]["pct"] if tiers and _sel_tier_s in tiers else 0

        _recovery_hero_s = (
            f'<div style="border:2px solid #22c55e;border-radius:10px;padding:1rem 1.5rem;'
            f'text-align:center;margin-bottom:1rem;">'
            f'<div style="color:var(--muted);font-size:0.85rem;margin-bottom:0.25rem;">'
            f'Tier {_sel_tier_s} Variants Recovered</div>'
            f'<div style="color:#16a34a;font-size:2.5rem;font-weight:700;line-height:1;">'
            f'{_sel_count_s}</div>'
            f'<div style="color:var(--muted);font-size:0.85rem;margin-top:0.25rem;">'
            f'{_sel_pct_s:.1f}% of {library_size} library variants</div>'
            f'</div>'
        ) if tiers and _sel_count_s else ""

        library_section = f"""
    <h2>Library Recovery</h2>
    <div class="recovery-row">
        <div class="recovery-tiers">
            {_recovery_hero_s}
            <div class="stat-grid" style="margin:0;">
                {tier_boxes}
            </div>
            {tier_note}
        </div>
        {recovery_curve_html}
    </div>
"""
    else:
        library_section = ""

    # Hit Picking section — only rendered when pick step is complete
    pick_completed = pick_state.get("completed", False)
    if pick_completed:
        pie_block = (
            f'<div class="pie-container" style="margin-top:1rem;">{donut_div}</div>'
            if donut_div else ""
        )

        # Build streakout-recoverable and missing variant tables from pick_list.json
        _detail_tables = ""
        _pick_list_file = (
            (project_dir / "merged" / "pick_list.json") if merged_context
            else (project_dir / "pick" / "pick_list.json") if project_dir
            else None
        )
        if _pick_list_file and _pick_list_file.exists():
            import html as _html_mod
            _pick_data = json.load(open(_pick_list_file))

            # --- Streakout-recoverable table ---
            _so_hits = [h for h in _pick_data if h.get("tier_override") == "Streakout"]
            if _so_hits:
                _so_rows = ""
                for h in _so_hits:
                    _pct = f"{h['consensus_fraction']:.0%}" if h.get("consensus_fraction") else ""
                    _so_rows += (
                        f'<tr>'
                        f'<td>{_html_mod.escape(h["variant"])}</td>'
                        f'<td>{_html_mod.escape(h.get("target_well", ""))}</td>'
                        f'<td>{_html_mod.escape(h.get("source_plate", ""))}:{_html_mod.escape(h.get("source_well", ""))}</td>'
                        f'<td>{h.get("reads", 0):,}</td>'
                        f'<td>{_pct}</td>'
                        f'</tr>'
                    )
                _detail_tables += (
                    f'<div class="chart-card" style="margin-top:1.5rem;">'
                    f'<h3>Streakout-Recoverable Variants ({len(_so_hits)})</h3>'
                    f'<p class="note">Variants not found in a clean well but recoverable '
                    f'by streaking out a mixed colony.</p>'
                    f'<table class="detail-table">'
                    f'<thead><tr><th>Variant</th><th>Target Well</th><th>Source Well</th>'
                    f'<th>Reads</th><th>% of Well</th></tr></thead>'
                    f'<tbody>{_so_rows}</tbody></table></div>'
                )

            # --- Missing variants table ---
            _missing = [h for h in _pick_data if h.get("empty")]
            if _missing:
                # Load DNA sequences from variants.csv and translate to protein
                _codon_table = {
                    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
                    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
                    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
                }
                def _translate(dna: str) -> str:
                    dna = dna.upper().replace(" ", "")
                    aa = []
                    for i in range(0, len(dna) - 2, 3):
                        codon = dna[i:i+3]
                        aa.append(_codon_table.get(codon, '?'))
                    return ''.join(aa)

                _seq_map: dict = {}
                if project_dir:
                    _vcf = project_dir / "variants.csv"
                    if _vcf.exists():
                        import csv as _csv
                        with open(_vcf) as _f:
                            for _row in _csv.DictReader(_f):
                                _vname = _row.get("Name", "")
                                _dna = _row.get("Sequence", "")
                                if _vname and _dna:
                                    _seq_map[_vname] = _translate(_dna)

                _miss_rows = ""
                for h in _missing:
                    _vn = h["variant"]
                    _aa = _seq_map.get(_vn, "")
                    _miss_rows += (
                        f'<tr>'
                        f'<td>{_html_mod.escape(_vn)}</td>'
                        f'<td>{_html_mod.escape(h.get("target_well", ""))}</td>'
                        f'<td style="font-family:monospace;font-size:0.8rem;word-break:break-all;">'
                        f'{_html_mod.escape(_aa)}</td>'
                        f'</tr>'
                    )
                _detail_tables += (
                    f'<div class="chart-card" style="margin-top:1.5rem;">'
                    f'<h3>Missing Variants ({len(_missing)})</h3>'
                    f'<p class="note">Variants not recovered in any sequencing round.</p>'
                    f'<table class="detail-table">'
                    f'<thead><tr><th>Variant</th><th>Target Well</th>'
                    f'<th>AA Sequence of Insert</th></tr></thead>'
                    f'<tbody>{_miss_rows}</tbody></table></div>'
                )

        if pick_plate_iframe:
            hitpick_section = f"""
    <h2>Hit Picking</h2>
    <div class="picking-layout">
        <div class="picking-left">
            {pick_stat_box}
            {pie_block}
        </div>
        <div class="picking-right">
            {pick_plate_iframe}
        </div>
    </div>
    {_detail_tables}
"""
        else:
            hitpick_section = f"""
    <h2>Hit Picking</h2>
    {pick_stat_box}
    {pie_block}
    {_detail_tables}
"""
    else:
        hitpick_section = ""


    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=0.95">
    <title>uSort-M Report</title>
    {_bokeh_css}
    {_bokeh_js}
    <style>
        :root {{
            --bg: #fafafa;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --muted: #6b7280;
            --border: #e5e7eb;
            --accent: #2563eb;
            --accent-dark: #1e40af;
            --success: #059669;
            --hover-bg: #f9fafb;
            --th-bg: #2563eb;
            --th-text: #ffffff;
        }}
        [data-theme="dark"] {{
            --bg: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #e0e0e0;
            --muted: #94a3b8;
            --border: #334155;
            --accent: #4cc9f0;
            --accent-dark: #7dd3fc;
            --success: #34d399;
            --hover-bg: #1e293b;
            --th-bg: #1e3a5f;
            --th-text: #e0e0e0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1320px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
            color: var(--text-color);
            font-size: 0.95rem;
        }}
        h1 {{
            color: var(--accent);
            border-bottom: 3px solid var(--accent);
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: var(--accent-dark);
            margin-top: 2rem;
        }}
        h3 {{
            color: var(--text-color);
        }}
        a {{
            color: var(--accent);
        }}
        .theme-toggle {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            font-size: 1.1rem;
            color: var(--text-color);
            z-index: 100;
        }}
        .theme-toggle:hover {{
            background: var(--hover-bg);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-box {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}
        .stat-box.selected-tier {{
            border: 2px solid var(--success);
        }}
        .selected-tier-label {{
            font-size: 0.72rem;
            color: var(--success);
            font-weight: 600;
            margin-top: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .stat-value {{
            font-size: 2.25rem;
            font-weight: bold;
            color: var(--text-color);
            margin-top: 0.5rem;
        }}
        .stat-sub {{
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }}
        .success {{
            color: var(--success);
        }}
        .note {{
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.5rem;
        }}
        .chart-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.25rem 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .chart-card h3 {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin: 0 0 0.75rem;
        }}
        .detail-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        .detail-table th {{
            text-align: left;
            padding: 0.4rem 0.6rem;
            border-bottom: 2px solid var(--border);
            color: var(--th-text);
            font-weight: 600;
        }}
        .detail-table td {{
            padding: 0.35rem 0.6rem;
            border-bottom: 1px solid var(--border);
        }}
        .detail-table tr:last-child td {{
            border-bottom: none;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 1.25rem;
            margin: 1.25rem 0;
        }}
        .chart-grid-wide {{
            grid-template-columns: 1fr;
        }}
        .subsection-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin: 0 0 0.5rem;
        }}
        .demux-row {{
            display: flex;
            gap: 1.25rem;
            align-items: flex-start;
            margin: 1.25rem 0;
        }}
        .demux-stat {{
            flex: 0 0 200px;
        }}
        .demux-hist {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .read-len-chart {{
            margin-top: 0.25rem;
        }}
        .picking-layout {{
            display: flex;
            gap: 1.25rem;
            align-items: flex-start;
        }}
        .picking-left {{
            flex: 0 0 340px;
        }}
        .picking-right {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .pie-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }}
        .tier-grid {{
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        th {{
            background: var(--th-bg);
            color: var(--th-text);
            text-align: left;
            padding: 1rem;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-color);
        }}
        tr:hover {{
            background: var(--hover-bg);
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.875rem;
        }}
        .overview-chips {{
            margin: 0;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        }}
        .recovery-row {{
            display: flex;
            gap: 1.25rem;
            align-items: flex-start;
            margin: 1rem 0;
        }}
        .recovery-tiers {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .recovery-tiers .stat-grid {{
            grid-template-columns: repeat(3, 1fr);
        }}
        .recovery-row > .chart-card {{
            flex: 0 0 420px;
            max-width: 420px;
        }}
        .merged-recovery-row {{
            background: var(--hover-bg);
            font-weight: 600;
            border-top: 2px solid var(--accent);
        }}
        .merged-recovery-row td {{
            border-top: 2px solid var(--accent);
        }}
    </style>
</head>
<body>
    <button class="theme-toggle" id="themeToggle" title="Toggle dark mode">
        <span id="themeIcon">\u2600\ufe0f</span>
    </button>

    <h1>uSort-M Workflow Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Library Overview</h2>
    <div class="stat-grid overview-chips">
        <div class="stat-box">
            <div class="stat-label">Library Size</div>
            <div class="stat-value">{project.get('library_size', 'N/A')}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Sequence Length</div>
            <div class="stat-value">{seq_len_display}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">True Sampling</div>
            <div class="stat-value">{true_sampling_display}</div>
            <div class="stat-sub">{true_sampling_note}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Input Reads</div>
            <div class="stat-value">{input_reads:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Round</div>
            <div class="stat-value">{round_label}</div>
        </div>
    </div>

    <h2>Sequencing &amp; Read Depth</h2>
    <div class="chart-grid">
        <div class="chart-card">
            <h3>Read Depth per Well</h3>
            {read_depth_div}
        </div>
        <div class="chart-card">
            <h3>Reads per Plate</h3>
            {plate_bar_div}
        </div>
{read_len_col_html}
        <div class="chart-card">
            <h3>Read Depth Summary</h3>
            <table style="box-shadow:none; border:none;">
                <tbody>
                    <tr><td style="border:none;">Average reads/well</td><td style="border:none; text-align:right; font-weight:600;">{avg_reads:.0f}</td></tr>
                    <tr><td style="border:none;">Maximum reads</td><td style="border:none; text-align:right; font-weight:600;">{max_reads:,}</td></tr>
                    <tr><td style="border:none;">Total wells</td><td style="border:none; text-align:right; font-weight:600;">{len(well_data):,}</td></tr>
                </tbody>
            </table>
        </div>
    </div>
{library_section}
{plate_map_section}
{hitpick_section}
    <div class="footer">
        <p>Generated by <strong>uSort-M</strong> | <a href="https://github.com/FordyceLab/usortm">GitHub</a></p>
    </div>

    {_bokeh_script}

    <script>
    (function() {{
        var toggle = document.getElementById('themeToggle');
        var icon = document.getElementById('themeIcon');
        var stored = localStorage.getItem('usortm-theme');
        if (stored === 'dark') {{
            document.documentElement.setAttribute('data-theme', 'dark');
            icon.textContent = '\u263e';
        }}
        function updateBokehTheme() {{
            if (!window.Bokeh || !window.Bokeh.documents) return;
            var dark = document.documentElement.getAttribute('data-theme') === 'dark';
            var c = dark
                ? {{muted: '#94a3b8', border: '#334155'}}
                : {{muted: '#6b7280', border: '#e5e7eb'}};
            try {{
                window.Bokeh.documents.forEach(function(doc) {{
                    doc._all_models.forEach(function(m) {{
                        if (m.axis_label_text_color !== undefined) {{
                            m.axis_label_text_color = c.muted;
                            m.major_label_text_color = c.muted;
                            m.axis_line_color = c.border;
                        }}
                        if (m.grid_line_color !== undefined) {{
                            m.grid_line_color = c.border;
                        }}
                    }});
                }});
            }} catch(e) {{}}
        }}
        toggle.addEventListener('click', function() {{
            var current = document.documentElement.getAttribute('data-theme');
            if (current === 'dark') {{
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('usortm-theme', 'light');
                icon.textContent = '\u2600\ufe0f';
            }} else {{
                document.documentElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('usortm-theme', 'dark');
                icon.textContent = '\u263e';
            }}
            setTimeout(updateBokehTheme, 50);
        }});
        window.addEventListener('load', updateBokehTheme);
    }})();
    </script>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)
