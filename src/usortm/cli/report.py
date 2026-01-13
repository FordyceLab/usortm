"""Generate final report and plate maps for a uSort-M project."""

from typing import Optional
from pathlib import Path
import csv
import json
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

PROJECT_STATE_FILE = "usortm_project.json"


def report(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    output_format: str = typer.Option(
        "all",
        "--format", "-f",
        help="Output format: 'csv', 'html', 'json', or 'all'.",
    ),
):
    """
    Generate final report and plate maps for a [blue]uSort-M[/blue] project.
    
    Creates:
    
    • Plate maps showing variant locations
    • Coverage summary statistics
    • Missing variants list
    • Final variant → well mapping
    
    [bold]Example:[/bold]
    
        usortm report my_project/
    """
    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project.")
        raise typer.Exit(1)
    
    with open(state_file) as f:
        project = json.load(f)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]uSort-M[/bold blue] Final Report",
        border_style="blue",
    ))
    console.print()
    
    # Create report directory
    report_dir = project_dir / "report"
    report_dir.mkdir(exist_ok=True)
    
    # Load original variants
    variants_file = project_dir / "variants.csv"
    original_variants = set()
    if variants_file.exists():
        with open(variants_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name") or row.get("variant") or list(row.values())[0]
                original_variants.add(name)
    
    # Load demux results
    demux_output = project_dir / "demux_output"
    recovered_variants = {}
    
    if (demux_output / "well_assignments.csv").exists():
        with open(demux_output / "well_assignments.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                variant = row["variant"]
                if variant not in recovered_variants:
                    recovered_variants[variant] = []
                recovered_variants[variant].append({
                    "plate": int(row["plate"]),
                    "well": row["well"],
                    "reads": int(row["reads"]),
                })
    
    # Load hit list if available
    hitlist_file = project_dir / "hitlist.csv"
    picked_variants = {}
    if hitlist_file.exists():
        with open(hitlist_file, newline="") as f:
            # Handle semicolon-delimited
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                picked_variants[row["SampleID"]] = {
                    "target_plate": int(row["TargetPlateID"]),
                    "target_well": row["TargetWell"],
                }
    
    # Calculate statistics
    library_size = project.get("library_size", len(original_variants))
    n_recovered = len(recovered_variants)
    n_picked = len(picked_variants)
    coverage = n_recovered / library_size if library_size > 0 else 0
    
    # Find missing variants
    missing_variants = original_variants - set(recovered_variants.keys())
    
    # Generate plate maps
    _generate_plate_maps(recovered_variants, picked_variants, report_dir, output_format)
    
    # Generate missing variants list
    _generate_missing_list(missing_variants, report_dir)
    
    # Generate final mapping
    _generate_final_mapping(picked_variants, recovered_variants, report_dir)
    
    # Generate summary report
    _generate_summary_html(project, recovered_variants, picked_variants, missing_variants, report_dir)
    
    # Update project state
    project["workflow_steps"]["report"] = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "coverage": round(coverage, 4),
        "recovered": n_recovered,
        "missing": len(missing_variants),
    }
    
    with open(state_file, "w") as f:
        json.dump(project, f, indent=2)
    
    # Display summary
    console.print()
    summary_table = Table(
        title="Project Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Library size", f"{library_size:,}")
    summary_table.add_row("Variants recovered", f"{n_recovered:,}")
    summary_table.add_row("Coverage", f"[green]{coverage:.1%}[/green]")
    summary_table.add_row("Missing variants", f"{len(missing_variants):,}")
    summary_table.add_row("Variants picked", f"{n_picked:,}")
    
    console.print(summary_table)
    console.print()
    
    # Cost summary
    if "costs" in project:
        costs = project["costs"]
        console.print(f"[bold]Final cost:[/bold] [green]${costs['total']:,.2f}[/green]")
        console.print(f"[bold]Cost per variant recovered:[/bold] ${costs['total']/n_recovered:.2f}")
        console.print()
    
    console.print("[green]✓[/green] Report generated:")
    console.print(f"  • {report_dir}/summary.html")
    console.print(f"  • {report_dir}/plate_maps.csv")
    console.print(f"  • {report_dir}/final_mapping.csv")
    console.print(f"  • {report_dir}/missing_variants.csv")
    console.print()
    
    if missing_variants:
        console.print(f"[yellow]Note:[/yellow] {len(missing_variants)} variants not recovered.")
        console.print("  Consider re-sorting with increased fold sampling, or")
        console.print("  re-synthesize missing variants if critical.")
    console.print()


def _generate_plate_maps(recovered: dict, picked: dict, output_dir: Path, format: str):
    """Generate plate maps showing variant locations."""
    # CSV format
    if format in ["csv", "all"]:
        with open(output_dir / "plate_maps.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "variant", "source_plate", "source_well", "reads",
                "target_plate", "target_well", "status"
            ])
            
            for variant, wells in recovered.items():
                # Use the best well (most reads)
                best_well = max(wells, key=lambda w: w["reads"])
                
                target_info = picked.get(variant, {})
                status = "picked" if variant in picked else "recovered"
                
                writer.writerow([
                    variant,
                    best_well["plate"],
                    best_well["well"],
                    best_well["reads"],
                    target_info.get("target_plate", ""),
                    target_info.get("target_well", ""),
                    status,
                ])


def _generate_missing_list(missing: set, output_dir: Path):
    """Generate list of missing variants."""
    with open(output_dir / "missing_variants.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"])
        for v in sorted(missing):
            writer.writerow([v])


def _generate_final_mapping(picked: dict, recovered: dict, output_dir: Path):
    """Generate final variant → well mapping for picked variants."""
    with open(output_dir / "final_mapping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "plate", "well"])
        
        for variant, info in sorted(picked.items()):
            writer.writerow([
                variant,
                info["target_plate"] + 1,  # Convert to 1-indexed
                info["target_well"],
            ])


def _generate_summary_html(project: dict, recovered: dict, picked: dict, 
                          missing: set, output_dir: Path):
    """Generate HTML summary report."""
    library_size = project.get("library_size", len(recovered) + len(missing))
    n_recovered = len(recovered)
    coverage = n_recovered / library_size if library_size > 0 else 0
    costs = project.get("costs", {})
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>uSort-M Project Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        .header p {{
            margin: 0.5rem 0 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #1e40af;
            font-size: 1.25rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        .stat {{
            text-align: center;
            padding: 1rem;
            background: #f0f9ff;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1e40af;
        }}
        .stat-label {{
            font-size: 0.875rem;
            color: #64748b;
        }}
        .cost-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .cost-row:last-child {{
            border-bottom: none;
            font-weight: bold;
        }}
        .progress-bar {{
            height: 24px;
            background: #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #22c55e, #16a34a);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>uSort-M Project Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    
    <div class="card">
        <h2>Coverage Summary</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {coverage*100:.1f}%">
                {coverage*100:.1f}%
            </div>
        </div>
        <div class="stats-grid" style="margin-top: 1rem;">
            <div class="stat">
                <div class="stat-value">{library_size:,}</div>
                <div class="stat-label">Library Size</div>
            </div>
            <div class="stat">
                <div class="stat-value">{n_recovered:,}</div>
                <div class="stat-label">Recovered</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(picked):,}</div>
                <div class="stat-label">Picked</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(missing):,}</div>
                <div class="stat-label">Missing</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Cost Breakdown</h2>
        <div class="cost-row">
            <span>Synthesis</span>
            <span>${costs.get('synthesis', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Cloning</span>
            <span>${costs.get('cloning', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Sorting</span>
            <span>${costs.get('sorting', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Barcoding</span>
            <span>${costs.get('barcoding', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Sequencing</span>
            <span>${costs.get('sequencing', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Hit-picking</span>
            <span>${costs.get('hitpicking', 0):,.2f}</span>
        </div>
        <div class="cost-row">
            <span>Total</span>
            <span>${costs.get('total', 0):,.2f}</span>
        </div>
        <div class="cost-row" style="margin-top: 1rem; border-top: 2px solid #1e40af; padding-top: 1rem;">
            <span>Cost per variant recovered</span>
            <span>${costs.get('total', 0) / max(n_recovered, 1):.2f}</span>
        </div>
    </div>
    
    <div class="card">
        <h2>Experiment Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Sequence length</td>
                <td>{project.get('seq_length', 'N/A')} bp</td>
            </tr>
            <tr>
                <td>Fold sampling</td>
                <td>{project.get('fold_sampling', 'N/A')}×</td>
            </tr>
            <tr>
                <td>Expected skew</td>
                <td>{project.get('skew', 'N/A')}× (Q90/Q10)</td>
            </tr>
            <tr>
                <td>Plates sorted</td>
                <td>{project.get('n_plates', 'N/A')}</td>
            </tr>
            <tr>
                <td>Total wells</td>
                <td>{project.get('total_wells', 'N/A'):,}</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Files Generated</h2>
        <ul>
            <li><code>plate_maps.csv</code> - Source and target well assignments</li>
            <li><code>final_mapping.csv</code> - Final variant → well mapping</li>
            <li><code>missing_variants.csv</code> - Variants not recovered</li>
            <li><code>hitlist.csv</code> - Integra ASSIST PLUS input file</li>
        </ul>
    </div>
</body>
</html>
"""
    
    (output_dir / "summary.html").write_text(html)
