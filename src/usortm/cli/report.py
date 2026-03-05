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


def _count_unique_variants(well_data: list) -> int:
    """Count unique variants, stripping ``|cons_check`` suffixes first."""
    return len(set(w["variant"].split("|")[0] for w in well_data))


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

    # Check if demux has been run
    if "workflow_steps" not in project or not project["workflow_steps"].get("demux", {}).get("completed"):
        console.print("[red]Error:[/red] No demultiplexing results found.")
        console.print("Run 'usortm demux' first to process sequencing data.")
        raise typer.Exit(1)

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Reporting",
        border_style=BORDER_STYLE,
    ))
    console.print()

    # Load demux results
    demux_output = project_dir / "demux_output"
    well_assignments_file = demux_output / "well_assignments.csv"
    demux_summary_file = demux_output / "demux_summary.json"

    if not well_assignments_file.exists() or not demux_summary_file.exists():
        console.print(f"[red]Error:[/red] Demux results incomplete")
        raise typer.Exit(1)

    well_data = _load_well_assignments(well_assignments_file)
    with open(demux_summary_file) as f:
        demux_summary = json.load(f)

    console.print(f"[green]✓[/green] Loaded demux results ({len(well_data)} wells with data)")

    # Create report directory
    report_dir = project_dir / "report"
    report_dir.mkdir(exist_ok=True)

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
        _save_missing_variants(project, well_data, missing_variants_file)
        generated_files.append(missing_variants_file)

    if format in ["json", "all"]:
        # Generate JSON report
        json_file = report_dir / "report.json"
        _save_json_report(project, demux_summary, well_data, json_file)
        generated_files.append(json_file)

    if format in ["html", "all"]:
        # Generate HTML report
        html_file = report_dir / "summary.html"
        _save_html_report(project, demux_summary, well_data, html_file, project_dir)
        generated_files.append(html_file)

    # Display summary
    console.print()
    console.print("[green]✓[/green] Reports generated!")
    console.print()

    for file_path in generated_files:
        console.print(f"  • {file_path.relative_to(project_dir)}")

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

    stats_table.add_row("Library size", f"{project.get('library_size', 'N/A')}")
    stats_table.add_row("Input reads", f"{demux_summary.get('input_reads', demux_summary.get('total_reads', 0)):,}")
    stats_table.add_row("Wells with data", f"{demux_summary.get('wells_with_data', 0):,}")

    unique_variants = _count_unique_variants(well_data)
    stats_table.add_row("Unique variants", f"{unique_variants}")

    library_size = project.get("library_size", 0)
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


def _load_well_assignments(assignments_file: Path) -> list:
    """Load well assignments from demux output."""
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
            })

    return well_data


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


def _save_missing_variants(project: dict, well_data: list, output_file: Path):
    """Save list of variants not recovered."""
    # Get expected variants from project
    expected_variants = set()
    library_file = project.get("library_file") or project.get("variants_file")

    if library_file:
        library_path = Path(library_file)
        if library_path.exists():
            try:
                with open(library_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "Name" in row:
                            expected_variants.add(row["Name"])
                        elif "name" in row:
                            expected_variants.add(row["name"])
                        elif "variant" in row:
                            expected_variants.add(row["variant"])
            except:
                pass  # If we can't read it, skip

    # Get recovered variants (strip legacy |cons_check suffix)
    recovered_variants = set(w["variant"].split("|")[0] for w in well_data)

    # Find missing variants
    missing_variants = expected_variants - recovered_variants

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "status"])

        if len(expected_variants) == 0:
            writer.writerow(["N/A", "No library file found in project"])
        else:
            for variant in sorted(missing_variants):
                writer.writerow([variant, "missing"])


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
    def _qualifying_variants(min_reads: int) -> set[str]:
        return {
            w["variant"].split("|")[0] for w in well_data
            if w["reads"] >= min_reads and w["consensus_fraction"] > 0.9
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


def _generate_read_depth_histogram_svg(read_counts: list) -> str:
    """Generate an inline SVG histogram of per-well read depths with tier markers."""
    if not read_counts:
        return ""

    max_val = max(read_counts)
    n_bins = 25
    bin_size = max(1, (max_val + n_bins) // n_bins)
    total_range = bin_size * n_bins

    bins = [0] * n_bins
    for r in read_counts:
        idx = min(int(r / bin_size), n_bins - 1)
        bins[idx] += 1
    max_count = max(bins) if any(bins) else 1

    # SVG layout
    ml, mr, mt, mb = 40, 15, 20, 42
    chart_w, chart_h = 280, 150
    svg_w = ml + chart_w + mr
    svg_h = mt + chart_h + mb

    bar_w = chart_w / n_bins
    els = []

    # Match plate map color scale: low=0 reads, high=200 reads (saturates above).
    plate_map_color_high = 200.0
    for i, count in enumerate(bins):
        if count == 0:
            continue
        x = ml + i * bar_w
        h = max(1, int((count / max_count) * chart_h))
        y = mt + chart_h - h
        t = min(((i + 0.5) * bin_size) / plate_map_color_high, 1.0)
        els.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(bar_w - 1, 1):.1f}" height="{h}" '
            f'rx="2" fill="{_cmap_hex(t)}" stroke="#aaa" stroke-width="0.3"/>'
        )

    # Tier threshold lines + rotated labels (avoids overlap at high read depths)
    for threshold, tier_label in [(20, "C"), (50, "B"), (100, "A")]:
        x = ml + (threshold / total_range) * chart_w
        if ml <= x <= ml + chart_w:
            els.append(
                f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + chart_h}" '
                f'stroke="var(--text-color)" stroke-width="1" stroke-dasharray="3,3" opacity="0.35"/>'
            )
            # Rotated label: text-anchor="start" with rotate(-90) around top of line
            # causes text to extend downward along the line from the top
            els.append(
                f'<text transform="rotate(-90, {x:.1f}, {mt})" '
                f'x="{x:.1f}" y="{mt}" '
                f'text-anchor="start" font-size="10" fill="var(--text-color)" opacity="0.55">'
                f' Tier {tier_label}</text>'
            )

    # X-axis ticks and labels — 5 evenly-spaced labels to avoid crowding
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_bin = int(i * n_bins / n_ticks)
        x = ml + tick_bin * bar_w
        label_val = tick_bin * bin_size
        els.append(
            f'<line x1="{x:.1f}" y1="{mt + chart_h}" x2="{x:.1f}" y2="{mt + chart_h + 4}" '
            f'stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{mt + chart_h + 16}" '
            f'text-anchor="middle" font-size="11" fill="var(--muted)">{label_val}</text>'
        )

    # X-axis label
    els.append(
        f'<text x="{ml + chart_w / 2:.1f}" y="{svg_h - 4}" '
        f'text-anchor="middle" font-size="11" fill="var(--muted)">Reads per well</text>'
    )

    # Y-axis label (rotated)
    els.append(
        f'<text x="{-(mt + chart_h / 2):.1f}" y="13" '
        f'transform="rotate(-90)" text-anchor="middle" font-size="11" fill="var(--muted)">Wells</text>'
    )

    # Y-axis max value
    els.append(
        f'<text x="{ml - 4}" y="{mt + 5}" '
        f'text-anchor="end" font-size="11" fill="var(--muted)">{max_count}</text>'
    )

    # Axes
    els.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
        f'<line x1="{ml}" y1="{mt + chart_h}" x2="{ml + chart_w}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
    )

    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif; overflow:visible; width:100%; height:auto;">'
        + "\n".join(els)
        + "</svg>"
    )


def _generate_plate_bar_svg(plate_reads: dict[str, int]) -> str:
    """Generate an inline SVG vertical bar chart for per-plate read counts."""
    if not plate_reads:
        return ""

    def _fmt_reads(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:,}"

    sorted_plates = sorted(plate_reads.items(), key=lambda x: int(x[0]))
    max_reads = max(plate_reads.values()) or 1
    ml, mr, mt, mb = 40, 15, 20, 42
    chart_w, chart_h = 310, 150
    svg_width = ml + chart_w + mr
    svg_height = mt + chart_h + mb

    slot_w = chart_w / len(sorted_plates)
    bar_w = max(min(slot_w * 0.65, 36), 6)
    bars = []
    for i, (plate, reads) in enumerate(sorted_plates):
        x_mid = ml + (i + 0.5) * slot_w
        bar_h = max(int((reads / max_reads) * chart_h), 2)
        x = x_mid - (bar_w / 2)
        y = mt + chart_h - bar_h
        fill = _cmap_hex(reads / max_reads)
        bars.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" '
            f'rx="3" fill="{fill}" stroke="#aaa" stroke-width="0.3"/>'
            f'<text x="{x_mid:.1f}" y="{max(y - 4, 11)}" '
            f'text-anchor="middle" font-size="15" fill="var(--muted)">{_fmt_reads(reads)}</text>'
            f'<text x="{x_mid:.1f}" y="{mt + chart_h + 16}" '
            f'text-anchor="middle" font-size="14" fill="var(--muted)">{plate}</text>'
        )

    bars.append(
        f'<text x="{ml - 4}" y="{mt + 5}" '
        f'text-anchor="end" font-size="14" fill="var(--muted)">{_fmt_reads(max_reads)}</text>'
        f'<text x="{ml + chart_w / 2:.1f}" y="{svg_height - 4}" '
        f'text-anchor="middle" font-size="14" fill="var(--muted)">Plate</text>'
        f'<text x="{-(mt + chart_h / 2):.1f}" y="13" '
        f'transform="rotate(-90)" text-anchor="middle" font-size="14" fill="var(--muted)">Reads</text>'
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
        f'<line x1="{ml}" y1="{mt + chart_h}" x2="{ml + chart_w}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
    )

    return (
        f'<svg viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif; overflow:visible; width:100%; height:auto;">'
        + "\n".join(bars)
        + "</svg>"
    )


def _generate_tier_pie_svg(tiers: dict, library_size: int) -> str:
    """Generate an inline SVG donut chart showing tier + untiered breakdown."""
    import math

    if not tiers or not library_size:
        return ""

    tier_a = tiers["A"]["count"]
    tier_b = tiers["B"]["count"]
    tier_c = tiers["C"]["count"]

    count_a = tier_a
    count_b = max(0, tier_b - tier_a)
    count_c = max(0, tier_c - tier_b)
    count_u = max(0, library_size - tier_c)

    total = count_a + count_b + count_c + count_u
    if total == 0:
        return ""

    segments = [
        (count_a, _cmap_hex(0.90), "Tier A (\u2265100 reads)"),
        (count_b, _cmap_hex(0.60), "Tier B (50\u201399 reads)"),
        (count_c, _cmap_hex(0.25), "Tier C (20\u201349 reads)"),
        (count_u, "#d1d5db",        "Untiered (<20 reads)"),
    ]

    cx, cy, R, r = 85, 85, 70, 38
    svg_w, svg_h = 170, 170
    els: list[str] = []

    start_angle = -90.0
    for count, color, label in segments:
        if count == 0:
            continue
        fraction = count / total
        end_angle = start_angle + fraction * 360.0

        sa = math.radians(start_angle)
        ea = math.radians(end_angle)

        ox1 = cx + R * math.cos(sa)
        oy1 = cy + R * math.sin(sa)
        ox2 = cx + R * math.cos(ea)
        oy2 = cy + R * math.sin(ea)
        ix1 = cx + r * math.cos(ea)
        iy1 = cy + r * math.sin(ea)
        ix2 = cx + r * math.cos(sa)
        iy2 = cy + r * math.sin(sa)

        large = 1 if fraction > 0.5 else 0
        pct = fraction * 100
        title_txt = f"{label}: {count:,} ({pct:.1f}%)"

        path_d = (
            f"M {ox1:.2f},{oy1:.2f} "
            f"A {R},{R} 0 {large},1 {ox2:.2f},{oy2:.2f} "
            f"L {ix1:.2f},{iy1:.2f} "
            f"A {r},{r} 0 {large},0 {ix2:.2f},{iy2:.2f} Z"
        )
        els.append(
            f'<path d="{path_d}" fill="{color}" stroke="var(--card-bg)" stroke-width="1.5" '
            f'style="cursor:pointer;" '
            f'onmouseover="this.style.opacity=\'0.8\'" '
            f'onmouseout="this.style.opacity=\'1\'">'
            f'<title>{title_txt}</title>'
            f'</path>'
        )
        start_angle = end_angle

    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif; overflow:visible; width:100%; height:auto; max-width:{svg_w}px;">'
        + "\n".join(els)
        + "</svg>"
    )


def _generate_read_length_hist_svg(hist_data: dict) -> str:
    """Generate an inline SVG histogram of input read lengths.

    Args:
        hist_data: Dict with bin_size, counts (list of 50 ints), median, n_reads.
    """
    if not hist_data or not hist_data.get("counts"):
        return ""

    counts = hist_data["counts"]
    bin_size = hist_data.get("bin_size", 1)
    median_bp = hist_data.get("median", 0)

    n_bins = len(counts)
    max_count = max(counts) if any(counts) else 1

    # SVG layout — same dimensions as read depth histogram
    ml, mr, mt, mb = 40, 15, 20, 42
    chart_w, chart_h = 280, 150
    svg_w = ml + chart_w + mr
    svg_h = mt + chart_h + mb

    bar_w = chart_w / n_bins
    els: list[str] = []

    # Reference read length for colormap (500 bp = saturated)
    ref_len = 500.0
    for i, count in enumerate(counts):
        if count == 0:
            continue
        x = ml + i * bar_w
        h = max(1, int((count / max_count) * chart_h))
        y = mt + chart_h - h
        mid_len = (i + 0.5) * bin_size
        t = min(mid_len / ref_len, 1.0)
        els.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(bar_w - 1, 1):.1f}" height="{h}" '
            f'rx="2" fill="{_cmap_hex(t)}" stroke="#aaa" stroke-width="0.3"/>'
        )

    # Find local maxima (peaks): bins[i] > bins[i-1] and > bins[i+1] by >5% of max
    peaks: list[tuple[int, int]] = []
    threshold_5pct = max_count * 0.05
    for i in range(1, n_bins - 1):
        if (counts[i] > counts[i - 1] and
                counts[i] > counts[i + 1] and
                counts[i] - counts[i - 1] > threshold_5pct and
                counts[i] - counts[i + 1] > threshold_5pct):
            peaks.append((i, counts[i]))

    # Annotate top 1-2 peaks with an upward triangle, suppressing any that
    # fall within 3 bins of the median to avoid label overlap.
    peaks_sorted = sorted(peaks, key=lambda p: p[1], reverse=True)[:2]
    _median_bin_pre = min(int(median_bp / bin_size), n_bins - 1) if median_bp > 0 else -100
    peaks_to_annotate = [
        (b, c) for b, c in peaks_sorted
        if abs(b - _median_bin_pre) > 3
    ]
    for peak_bin, peak_count in peaks_to_annotate:
        px = ml + (peak_bin + 0.5) * bar_w
        ph = max(1, int((peak_count / max_count) * chart_h))
        py = mt + chart_h - ph - 8
        peak_bp = int((peak_bin + 0.5) * bin_size)
        els.append(
            f'<polygon points="{px:.1f},{py:.0f} {px - 5:.1f},{py + 8:.0f} {px + 5:.1f},{py + 8:.0f}" '
            f'fill="var(--text-color)" opacity="0.5"/>'
            f'<text x="{px:.1f}" y="{py - 3:.0f}" text-anchor="middle" '
            f'font-size="9" fill="var(--text-color)" opacity="0.65">{peak_bp}bp</text>'
        )

    # Median marker: red inverted triangle + label
    if median_bp > 0:
        median_bin = min(int(median_bp / bin_size), n_bins - 1)
        median_count = counts[median_bin]
        mx = ml + (median_bp / bin_size) * bar_w
        mx = max(ml, min(ml + chart_w, mx))
        mh = max(1, int((median_count / max_count) * chart_h))
        my = mt + chart_h - mh - 8
        # Inverted triangle (pointing down)
        els.append(
            f'<polygon points="{mx:.1f},{my + 8:.0f} {mx - 5:.1f},{my:.0f} {mx + 5:.1f},{my:.0f}" '
            f'fill="#ef4444" opacity="0.85"/>'
            f'<text x="{mx:.1f}" y="{my - 3:.0f}" text-anchor="middle" '
            f'font-size="9" fill="#ef4444">med {median_bp}bp</text>'
        )

    # X-axis ticks — 5 evenly spaced
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_bin = int(i * n_bins / n_ticks)
        x = ml + tick_bin * bar_w
        label_val = tick_bin * bin_size
        els.append(
            f'<line x1="{x:.1f}" y1="{mt + chart_h}" x2="{x:.1f}" y2="{mt + chart_h + 4}" '
            f'stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{mt + chart_h + 16}" '
            f'text-anchor="middle" font-size="11" fill="var(--muted)">{label_val}</text>'
        )

    # X-axis label
    els.append(
        f'<text x="{ml + chart_w / 2:.1f}" y="{svg_h - 4}" '
        f'text-anchor="middle" font-size="11" fill="var(--muted)">Read Length (bp)</text>'
    )

    # Y-axis label (rotated)
    els.append(
        f'<text x="{-(mt + chart_h / 2):.1f}" y="13" '
        f'transform="rotate(-90)" text-anchor="middle" font-size="11" fill="var(--muted)">Reads</text>'
    )

    # Y-axis max value
    def _fmt(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    els.append(
        f'<text x="{ml - 4}" y="{mt + 5}" '
        f'text-anchor="end" font-size="11" fill="var(--muted)">{_fmt(max_count)}</text>'
    )

    # Axes
    els.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
        f'<line x1="{ml}" y1="{mt + chart_h}" x2="{ml + chart_w}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
    )

    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif; overflow:visible; width:100%; height:auto;">'
        + "\n".join(els)
        + "</svg>"
    )


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


def _generate_recovery_curve_svg(
    curve_data: dict,
    true_sampling: Optional[float],
    tier_c_pct: Optional[float],
    round_n: int = 1,
) -> str:
    """Generate an inline SVG recovery curve.

    Args:
        curve_data:    Dict with fold_samplings, coverage_means, coverage_stds.
        true_sampling: Actual fold sampling (x-position of the real point).
        tier_c_pct:    Actual Tier-C coverage % (y-position), or None.
        round_n:       Sort round number (shown in top-left corner label).
    """
    fold_samplings = curve_data.get("fold_samplings", [])
    coverage_means = curve_data.get("coverage_means", [])
    coverage_stds = curve_data.get("coverage_stds", [])

    if not fold_samplings or not coverage_means:
        return ""

    chart_w, chart_h = 150, 150
    ml, mr, mt, mb = 32, 10, 12, 32
    svg_w = ml + chart_w + mr
    svg_h = mt + chart_h + mb

    x_max = max(fold_samplings)
    x_min = 0.0

    def _x(fs: float) -> float:
        return ml + (fs - x_min) / (x_max - x_min) * chart_w

    def _y(pct: float) -> float:
        return mt + chart_h - (pct / 100.0) * chart_h

    els: list[str] = []

    # Round label in top-left corner of chart area
    els.append(
        f'<text x="{ml + 4}" y="{mt + 10}" font-size="9" fill="var(--muted)">Round {round_n}</text>'
    )

    # Horizontal grid lines + Y-axis tick labels
    for pct in (25, 50, 75, 100):
        gy = _y(pct)
        els.append(
            f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + chart_w}" y2="{gy:.1f}" '
            f'stroke="var(--border)" stroke-width="0.8" stroke-dasharray="3,3"/>'
            f'<text x="{ml - 4}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="var(--muted)">{pct}</text>'
        )

    # ±1 std ribbon as polygon — prepend origin to upper, append to lower
    origin_pt = f"{_x(0):.1f},{_y(0):.1f}"
    upper_pts = origin_pt + " " + " ".join(
        f"{_x(fs):.1f},{_y(min(m + s, 100)):.1f}"
        for fs, m, s in zip(fold_samplings, coverage_means, coverage_stds)
    )
    lower_pts = " ".join(
        f"{_x(fs):.1f},{_y(max(m - s, 0)):.1f}"
        for fs, m, s in zip(
            reversed(fold_samplings),
            reversed(coverage_means),
            reversed(coverage_stds),
        )
    ) + f" {_x(0):.1f},{_y(0):.1f}"
    els.append(
        f'<polygon points="{upper_pts} {lower_pts}" '
        f'fill="var(--accent)" opacity="0.15"/>'
    )

    # Mean curve — prepend origin
    mean_pts = origin_pt + " " + " ".join(
        f"{_x(fs):.1f},{_y(m):.1f}"
        for fs, m in zip(fold_samplings, coverage_means)
    )
    els.append(
        f'<polyline points="{mean_pts}" fill="none" stroke="var(--accent)" '
        f'stroke-width="2" stroke-linejoin="round"/>'
    )

    # Actual point or dashed vertical line
    if true_sampling is not None:
        ax = min(max(_x(true_sampling), ml), ml + chart_w)
        if tier_c_pct is not None:
            ay = _y(min(max(tier_c_pct, 0), 100))
            els.append(
                f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="5" fill="#22c55e"/>'
            )
            # Position label above and slightly right, clamping to chart bounds
            label_x = min(ax + 7, ml + chart_w - 2)
            label_y = max(ay - 7, mt + 10)
            els.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
                f'font-size="9" fill="#22c55e">actual</text>'
            )
        else:
            els.append(
                f'<line x1="{ax:.1f}" y1="{mt}" x2="{ax:.1f}" y2="{mt + chart_h}" '
                f'stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>'
                f'<text x="{ax + 3:.1f}" y="{mt + 12}" '
                f'font-size="9" fill="var(--muted)">current</text>'
            )

    # X-axis ticks
    tick_vals = [v for v in (0, 2, 4, 6, 8, 10, 12, 15) if v <= x_max]
    for tv in tick_vals:
        tx = _x(tv)
        els.append(
            f'<line x1="{tx:.1f}" y1="{mt + chart_h}" x2="{tx:.1f}" y2="{mt + chart_h + 4}" '
            f'stroke="var(--border)" stroke-width="1"/>'
            f'<text x="{tx:.1f}" y="{mt + chart_h + 15}" text-anchor="middle" '
            f'font-size="10" fill="var(--muted)">{tv}</text>'
        )

    # Axis labels
    els.append(
        f'<text x="{ml + chart_w / 2:.1f}" y="{svg_h - 4}" '
        f'text-anchor="middle" font-size="10" fill="var(--muted)">Fold Sampling</text>'
    )
    els.append(
        f'<text x="{-(mt + chart_h / 2):.1f}" y="13" '
        f'transform="rotate(-90)" text-anchor="middle" font-size="10" fill="var(--muted)">% Recovered</text>'
    )

    # Axes
    els.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
        f'<line x1="{ml}" y1="{mt + chart_h}" x2="{ml + chart_w}" y2="{mt + chart_h}" '
        f'stroke="var(--border)" stroke-width="1.5"/>'
    )

    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif; overflow:visible; width:100%; height:auto; max-width:{svg_w}px;">'
        + "\n".join(els)
        + "</svg>"
    )


def _save_html_report(project: dict, demux_summary: dict, well_data: list,
                      output_file: Path, project_dir: Path = None):
    """Save interactive HTML summary report with embedded plate maps."""
    import html as _html

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

    # Quality bins / recovery tiers
    bins_data = _compute_quality_bins(well_data, library_size) if library_size else None
    tiers = bins_data["recovery_tiers"] if bins_data else None
    qbins = bins_data["quality_bins"] if bins_data else None

    # Recovery curve SVG (pre-computed during demux, cached in demux_summary)
    recovery_curve_data = demux_summary.get("recovery_curve")
    if recovery_curve_data:
        tier_c_pct = tiers["C"]["pct"] if tiers else None
        rcurve_svg = _generate_recovery_curve_svg(
            recovery_curve_data, true_sampling, tier_c_pct,
            round_n=project.get("round", 1),
        )
        recovery_curve_html = f'<div>{rcurve_svg}</div>' if rcurve_svg else ""
    else:
        recovery_curve_html = ""

    # Per-plate read totals for bar chart
    plate_reads: dict[str, int] = {}
    for w in well_data:
        p = w["plate"]
        plate_reads[p] = plate_reads.get(p, 0) + w["reads"]
    plate_bar_svg = _generate_plate_bar_svg(plate_reads)
    read_depth_histogram_svg = _generate_read_depth_histogram_svg(read_counts)

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
                        # Cache back so subsequent reports skip recomputation
                        _dsf = project_dir / "demux_output" / "demux_summary.json"
                        with open(_dsf, "w") as _f:
                            json.dump(demux_summary, _f, indent=2)
                except Exception:
                    pass
                break

    read_len_hist_svg = _generate_read_length_hist_svg(
        demux_summary.get("read_len_hist") or {}
    )
    if read_len_hist_svg:
        _n_reads = (demux_summary.get("read_len_hist") or {}).get("n_reads", 0)
        _n_reads_note = f" ({_n_reads:,} reads)" if _n_reads else ""
        read_len_col_html = (
            f'<div class="demux-hist">\n'
            f'    <h3 class="subsection-title">Input Read Length Distribution{_n_reads_note}</h3>\n'
            f'    <p class="note" style="margin:0 0 0.5rem;">Dominant peaks labeled &#9650;. '
            f'Median in red &#9660;. Short-read peaks (&lt;200&nbsp;bp) may reduce mapping efficiency.</p>\n'
            f'    <div class="read-len-chart">{read_len_hist_svg}</div>\n'
            f'</div>'
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
        '<p class="note">All tiers require &gt;90% consensus and count unique variants '
        '(best well per variant). Tiers are cumulative (B includes A, C includes B).</p>'
        if tiers else ""
    )

    # Tier pie chart
    tier_pie_svg = _generate_tier_pie_svg(tiers, library_size) if tiers else ""

    # Check for plate map files — embed via srcdoc if available
    plate_map_section = ""
    pick_plate_iframe = ""

    if project_dir:
        demux_plate_map = project_dir / "demux_output" / "plate_map.html"
        if demux_plate_map.exists():
            try:
                raw_html = demux_plate_map.read_text()
                escaped = _html.escape(raw_html)
                plate_map_section = f"""
    <h2>Demux Plate Map</h2>
    <p>Interactive plate map showing per-well read depth and variant composition.</p>
    <iframe srcdoc="{escaped}" width="100%" height="620"
            style="border:none;"></iframe>
"""
            except Exception:
                rel_path = "../demux_output/plate_map.html"
                plate_map_section = f"""
    <h2>Demux Plate Map</h2>
    <p>Interactive plate map showing per-well read depth and variant composition.
       <a href="{rel_path}" target="_blank">Open full size</a></p>
    <iframe src="{rel_path}" width="100%" height="620"
            style="border:none;"></iframe>
"""

        pick_plate_map = project_dir / "pick" / "pick_plate_map.html"
        if not pick_plate_map.exists():
            pick_plate_map = project_dir / "pick_plate_map.html"  # backward compat
        if pick_plate_map.exists():
            try:
                raw_html = pick_plate_map.read_text()
                escaped = _html.escape(raw_html)
                pick_plate_iframe = (
                    f'<iframe srcdoc="{escaped}" width="100%" height="780" '
                    f'style="border:none;"></iframe>'
                )
            except Exception:
                rel_path = "../pick/pick_plate_map.html"
                pick_plate_iframe = (
                    f'<p><a href="{rel_path}" target="_blank">Open pick plate map</a></p>'
                    f'<iframe src="{rel_path}" width="100%" height="780" '
                    f'style="border:none;"></iframe>'
                )

    # Pick summary stat box (inline in library section)
    pick_stat_box = ""
    if pick_state.get("completed"):
        tier_sub = (
            f'<div class="stat-sub">Tier {pick_state["tier"]} filter</div>'
            if pick_state.get("tier") else ""
        )
        pick_stat_box = (
            f'<div class="stat-box" style="margin-top:1rem;">'
            f'<div class="stat-label">Unique Variants Picked</div>'
            f'<div class="stat-value success">{pick_state.get("unique_variants", "N/A")}</div>'
            f'{tier_sub}'
            f'</div>'
        )

    # Library Recovery section — tier chips only
    library_section = f"""
    <h2>Library Recovery</h2>
    <div class="stat-grid">
        {tier_boxes}
    </div>
{tier_note}
""" if tier_boxes else ""

    # Hit Picking section — only rendered when pick step is complete
    pick_completed = pick_state.get("completed", False)
    if pick_completed:
        pie_block = (
            f'<div class="pie-container" style="margin-top:1rem;">{tier_pie_svg}</div>'
            if tier_pie_svg else ""
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
"""
        else:
            hitpick_section = f"""
    <h2>Hit Picking</h2>
    {pick_stat_box}
    {pie_block}
"""
    else:
        hitpick_section = ""

    # Streak-out candidates section
    streakout_section = ""
    streakout_data = demux_summary.get("streakout", {})
    if streakout_data.get("candidates", 0) > 0 and project_dir:
        streakout_csv = project_dir / "demux_output" / "streakout" / "streakout_candidates.csv"
        if streakout_csv.exists():
            import csv as _csv_mod
            so_rows = []
            with open(streakout_csv) as _sf:
                for sr in _csv_mod.DictReader(_sf):
                    plate = sr["plate"]
                    well = sr["well"]
                    pileup_href = f"../demux_output/streakout/well_{plate}_{well}.html"
                    so_rows.append(
                        f'<tr>'
                        f'<td>{plate}-{well}</td>'
                        f'<td>{sr["total_reads"]}</td>'
                        f'<td>{float(sr["top_frac"]):.0%}</td>'
                        f'<td>{sr["recoverable_variants"].replace(";", ", ")}</td>'
                        f'<td><a href="{pileup_href}" target="_blank">View</a></td>'
                        f'</tr>'
                    )
            if so_rows:
                streakout_section = f"""
    <h2>Streak-Out Candidates</h2>
    <p>Wells with multiple correctly-assembled subpopulations. Minority variants
       can be recovered by streaking out. Click a well to view the read pileup.</p>
    <table>
      <thead>
        <tr><th>Well</th><th>Reads</th><th>Top %</th><th>Recoverable Variants</th><th>Pileup</th></tr>
      </thead>
      <tbody>
        {''.join(so_rows)}
      </tbody>
    </table>
"""

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=0.95">
    <title>uSort-M Report</title>
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
        .read-depth-row {{
            display: flex;
            gap: 1.5rem;
            align-items: flex-start;
            flex-wrap: nowrap;
        }}
        .read-depth-row > table {{
            flex: 0 0 auto;
            width: auto;
            min-width: 220px;
        }}
        .read-depth-row > .bar-chart {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .read-depth-row > .histogram-chart {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .subsection-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-color);
            margin: 1.5rem 0 0.25rem;
        }}
        .demux-row {{
            display: flex;
            gap: 1.5rem;
            align-items: flex-start;
            margin: 1.5rem 0;
        }}
        .demux-stat {{
            flex: 0 0 200px;
        }}
        .demux-hist {{
            flex: 1 1 0;
            min-width: 0;
        }}
        .read-len-chart {{
            max-width: 560px;
            margin-top: 0.25rem;
        }}
        .picking-layout {{
            display: flex;
            gap: 1.5rem;
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
        .overview-row {{
            display: flex;
            flex-wrap: nowrap;
            gap: 1.5rem;
            align-items: flex-start;
            margin: 1rem 0 2rem;
        }}
        .overview-chips {{
            flex: 0 0 auto;
            margin: 0;
            grid-template-columns: repeat(2, minmax(160px, 1fr));
            max-width: 400px;
        }}
        .overview-recovery {{
            flex: 0 0 auto;
            min-width: 0;
            align-self: center;
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
    <div class="overview-row">
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
                <div class="stat-label">Round</div>
                <div class="stat-value">{project.get('round', 1)}</div>
            </div>
        </div>
        <div class="overview-recovery">
            {recovery_curve_html}
        </div>
    </div>

    <h2>Demultiplexing Results</h2>
    <div class="demux-row">
        <div class="demux-stat">
            <div class="stat-box">
                <div class="stat-label">Input Reads</div>
                <div class="stat-value">{demux_summary.get('input_reads', demux_summary.get('total_reads', 0)):,}</div>
            </div>
        </div>
{read_len_col_html}    </div>

    <h2>Read Depth Statistics</h2>
    <div class="read-depth-row">
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Average reads</td>
                <td>{avg_reads:.0f}</td>
            </tr>
            <tr>
                <td>Maximum reads</td>
                <td>{max_reads}</td>
            </tr>
            <tr>
                <td>Total wells</td>
                <td>{len(well_data)}</td>
            </tr>
        </tbody>
    </table>
    <div class="histogram-chart">
        {read_depth_histogram_svg}
    </div>
    <div class="bar-chart">
        {plate_bar_svg}
    </div>
    </div>
{library_section}
{plate_map_section}
{hitpick_section}
{streakout_section}
    <div class="footer">
        <p>Generated by <strong>uSort-M</strong> | <a href="https://github.com/FordyceLab/usortm">GitHub</a></p>
    </div>

    <script>
    (function() {{
        var toggle = document.getElementById('themeToggle');
        var icon = document.getElementById('themeIcon');
        var stored = localStorage.getItem('usortm-theme');
        if (stored === 'dark') {{
            document.documentElement.setAttribute('data-theme', 'dark');
            icon.textContent = '\u263e';
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
        }});
    }})();
    </script>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)
