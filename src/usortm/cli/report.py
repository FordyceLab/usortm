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
    # Group wells by variant
    variant_map = {}
    for well in well_data:
        variant = well["variant"]
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

    # Get recovered variants
    recovered_variants = set(w["variant"] for w in well_data)

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
        }
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)


def _generate_plate_bar_svg(plate_reads: dict[str, int]) -> str:
    """Generate an inline SVG horizontal bar chart for per-plate read counts."""
    if not plate_reads:
        return ""

    max_reads = max(plate_reads.values()) or 1
    bar_height = 28
    label_width = 80
    chart_width = 500
    padding = 4
    svg_height = len(plate_reads) * (bar_height + padding) + 10

    bars = []
    for i, (plate, reads) in enumerate(sorted(plate_reads.items(), key=lambda x: int(x[0]))):
        y = i * (bar_height + padding)
        bar_w = max(int((reads / max_reads) * chart_width), 2)
        bars.append(
            f'<text x="{label_width - 8}" y="{y + bar_height * 0.7}" '
            f'text-anchor="end" font-size="13" fill="#374151">Plate {plate}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'rx="4" fill="#2563eb" opacity="0.85"/>'
            f'<text x="{label_width + bar_w + 6}" y="{y + bar_height * 0.7}" '
            f'font-size="12" fill="#6b7280">{reads:,}</text>'
        )

    return (
        f'<svg width="{label_width + chart_width + 80}" height="{svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif;">'
        + "\n".join(bars)
        + "</svg>"
    )


def _save_html_report(project: dict, demux_summary: dict, well_data: list,
                      output_file: Path, project_dir: Path = None):
    """Save interactive HTML summary report with embedded plate maps."""
    # Calculate statistics — strip |cons_check suffix before counting
    unique_variants = _count_unique_variants(well_data)
    stripped = [w["variant"].split("|")[0] for w in well_data]
    variant_counts = Counter(stripped)

    read_counts = [w["reads"] for w in well_data]
    avg_reads = sum(read_counts) / len(read_counts) if read_counts else 0
    max_reads = max(read_counts) if read_counts else 0

    library_size = project.get("library_size", 0)
    coverage_pct = min((unique_variants / library_size) * 100, 100.0) if library_size else 0

    # Per-plate read totals for bar chart
    plate_reads: dict[str, int] = {}
    for w in well_data:
        p = w["plate"]
        plate_reads[p] = plate_reads.get(p, 0) + w["reads"]
    plate_bar_svg = _generate_plate_bar_svg(plate_reads)

    # Check for plate map files relative to project_dir
    plate_map_section = ""
    pick_map_section = ""
    pick_summary_section = ""

    if project_dir:
        demux_plate_map = project_dir / "demux_output" / "plate_map.html"
        if demux_plate_map.exists():
            rel_path = f"../demux_output/plate_map.html"
            plate_map_section = f"""
    <h2>Demux Plate Map</h2>
    <p>Interactive plate map showing per-well read depth and variant composition.
       <a href="{rel_path}" target="_blank">Open full size</a></p>
    <iframe src="{rel_path}" width="100%" height="620"
            style="border:1px solid #e5e7eb; border-radius:8px;"></iframe>
"""

        pick_plate_map = project_dir / "pick_plate_map.html"
        if pick_plate_map.exists():
            rel_path = f"../pick_plate_map.html"
            pick_map_section = f"""
    <h2>Pick Plate Map</h2>
    <p>Interactive plate map showing cherry-picked wells.
       <a href="{rel_path}" target="_blank">Open full size</a></p>
    <iframe src="{rel_path}" width="100%" height="620"
            style="border:1px solid #e5e7eb; border-radius:8px;"></iframe>
"""

        # Add pick summary if pick step was completed
        pick_state = project.get("workflow_steps", {}).get("pick", {})
        if pick_state.get("completed"):
            pick_summary_section = f"""
    <h2>Hit Picking Summary</h2>
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-label">Total Hits</div>
            <div class="stat-value">{pick_state.get('total_hits', 'N/A')}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Unique Variants Picked</div>
            <div class="stat-value success">{pick_state.get('unique_variants', 'N/A')}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Target Format</div>
            <div class="stat-value">{pick_state.get('target_format', 384)}-well</div>
        </div>
    </div>
"""

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>uSort-M Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #fafafa;
        }}
        h1 {{
            color: #2563eb;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: #1e40af;
            margin-top: 2rem;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-box {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 0.875rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1e293b;
            margin-top: 0.5rem;
        }}
        .success {{
            color: #059669;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #2563eb;
            color: white;
            text-align: left;
            padding: 1rem;
        }}
        td {{
            padding: 0.75rem 1rem;
            border-top: 1px solid #e5e7eb;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        .footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <h1>uSort-M Workflow Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Project Overview</h2>
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-label">Library Size</div>
            <div class="stat-value">{project.get('library_size', 'N/A')}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Sequence Length</div>
            <div class="stat-value">{project.get('seq_length', 'N/A')} bp</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Fold Sampling</div>
            <div class="stat-value">{project.get('fold_sampling', 'N/A')}×</div>
        </div>
    </div>

    <h2>Demultiplexing Results</h2>
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-label">Input Reads</div>
            <div class="stat-value">{demux_summary.get('input_reads', demux_summary.get('total_reads', 0)):,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Assigned Reads</div>
            <div class="stat-value">{demux_summary.get('assigned_reads', 0):,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Wells with Data</div>
            <div class="stat-value">{demux_summary.get('wells_with_data', 0):,}</div>
        </div>
    </div>

    <h2>Variant Recovery</h2>
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-label">Unique Variants</div>
            <div class="stat-value success">{unique_variants}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Library Coverage</div>
            <div class="stat-value success">{coverage_pct:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Avg Reads/Well</div>
            <div class="stat-value">{avg_reads:.0f}</div>
        </div>
    </div>

    <h2>Read Depth Statistics</h2>
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
    <h3>Reads per Plate</h3>
    {plate_bar_svg}
{plate_map_section}{pick_summary_section}{pick_map_section}
    <div class="footer">
        <p>Generated by <strong>uSort-M</strong> | <a href="https://github.com/FordyceLab/usortm">GitHub</a></p>
    </div>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)
