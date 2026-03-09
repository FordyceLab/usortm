"""Generate hit-picking lists from demultiplexing results."""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import csv
import json

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"

TIER_THRESHOLDS: dict[str, dict] = {
    "A": {"min_reads": 100, "min_consensus": 0.9},
    "B": {"min_reads": 50, "min_consensus": 0.9},
    "C": {"min_reads": 20, "min_consensus": 0.9},
}


def pick(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory (with demux results).",
        exists=True,
    ),
    targets: Optional[Path] = typer.Option(
        None,
        "--targets", "-t",
        help="CSV of specific variants to pick (columns: variant, count)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path for hit-picking list",
    ),
    volume: float = typer.Option(
        5.0,
        "--volume", "-v",
        help="Transfer volume in µL",
    ),
    target_format: int = typer.Option(
        384,
        "--target-format",
        help="Target plate format (96 or 384)",
    ),
    fill_order: str = typer.Option(
        "row",
        "--fill-order",
        help="Fill order for target plate (row or column)",
    ),
    tier: Optional[str] = typer.Option(
        "A",
        "--tier",
        help="Filter by quality tier: A (>=100 reads), B (>=50), C (>=20). All require >90% consensus. Use --tier '' to disable.",
    ),
    unique_only: bool = typer.Option(
        True,
        "--unique-only/--all-hits",
        help="Pick only one well per unique variant",
    ),
    compact: bool = typer.Option(
        False,
        "--compact/--no-compact",
        help="Pack recovered hits into adjacent wells; omit empty placeholders for unrecovered variants.",
    ),
):
    """
    Generate hit-picking list from demultiplexing results.

    Output is formatted for [bold]Integra ASSIST PLUS[/bold] liquid handling robots
    as semicolon-delimited CSV.

    [bold]Example:[/bold]

        usortm pick my_project/ --unique-only --volume 5.0
    """
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
        "[brand]uSort-M[/brand] Hit Picking",
        border_style=BORDER_STYLE,
    ))
    console.print()

    # Load demux results
    demux_output = project_dir / "demux_output"
    well_assignments_file = demux_output / "well_assignments.csv"

    if not well_assignments_file.exists():
        console.print(f"[red]Error:[/red] Well assignments not found: {well_assignments_file}")
        raise typer.Exit(1)

    # Validate tier option (empty string disables filtering)
    if tier is not None and tier.strip() == "":
        tier = None
    if tier is not None:
        tier = tier.upper()
        if tier not in TIER_THRESHOLDS:
            console.print(
                f"[red]Error:[/red] Invalid tier '{tier}'. Choose from: A, B, C"
            )
            raise typer.Exit(1)
        thresh = TIER_THRESHOLDS[tier]
        console.print(
            f"[green]\u2713[/green] Tier {tier} filter: "
            f"\u2265{thresh['min_reads']} reads, >{thresh['min_consensus']:.0%} consensus"
        )

    well_data = _load_well_assignments(well_assignments_file)
    console.print(f"[green]\u2713[/green] Loaded {len(well_data)} wells with data")

    # Load target variants if specified
    target_variants = None
    if targets:
        target_variants = _load_targets(targets)
        console.print(f"[green]\u2713[/green] Loaded {len(target_variants)} target variants")

    # Load library ordering from variants file (if available)
    library_order = _load_library_order(project)

    # Generate pick list
    pick_list = _generate_pick_list(
        well_data=well_data,
        target_variants=target_variants,
        unique_only=unique_only,
        target_format=target_format,
        fill_order=fill_order,
        library_order=library_order,
        tier=tier,
        compact=compact,
    )

    if len(pick_list) == 0:
        console.print("[yellow]Warning:[/yellow] No hits to pick!")
        console.print("Check your demux results and target criteria.")
        raise typer.Exit(1)

    # Determine output file path
    pick_dir = project_dir / "pick"
    pick_dir.mkdir(exist_ok=True)

    output_file = output
    if output_file is None:
        output_file = pick_dir / "hitlist.csv"

    # Save pick list in Integra ASSIST PLUS format
    _save_pick_list(pick_list, output_file, volume)

    # Generate interactive pick plate map (Bokeh is optional)
    try:
        from usortm.demux.viz import save_pick_plate_map_html

        pick_map_path = pick_dir / "pick_plate_map.html"
        save_pick_plate_map_html(
            pick_list, str(pick_map_path),
            title="Pick Plate Map",
            target_format=target_format,
        )
        console.print(
            f"[green]\u2713[/green] Pick plate map saved to {pick_map_path}"
        )
    except ImportError:
        pass  # Bokeh not installed — skip
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not generate pick plate map: {e}")

    # Save pick workflow state
    pick_state = {
        "completed": True,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "total_hits": len([h for h in pick_list if not h.get("empty")]),
        "unique_variants": len(set(h["variant"] for h in pick_list if not h.get("empty"))),
        "target_format": target_format,
        "compact": compact,
    }
    if tier:
        pick_state["tier"] = tier
    project["workflow_steps"]["pick"] = pick_state
    with open(state_file, "w") as f:
        json.dump(project, f, indent=2)

    # Display summary
    console.print()
    summary_table = Table(
        title="Hit Picking Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="muted")
    summary_table.add_column("Value", justify="right")

    recovered = [h for h in pick_list if not h.get("empty")]
    empty_count = len(pick_list) - len(recovered)
    unique_variants = len(set(h["variant"] for h in recovered))
    summary_table.add_row("Total hits", f"{len(recovered)}")
    summary_table.add_row("Unique variants", f"{unique_variants}")
    if compact:
        summary_table.add_row("Compact mode", "[green]on[/green]")
    elif empty_count > 0:
        summary_table.add_row("Empty wells (unrecovered)", f"{empty_count}")
    if tier:
        summary_table.add_row("Quality tier", f"Tier {tier}")
    summary_table.add_row("Transfer volume", f"{volume} \u00b5L")
    summary_table.add_row("Target format", f"{target_format}-well")
    summary_table.add_row("Fill order", fill_order)

    console.print(summary_table)
    console.print()

    console.print("[green]✓[/green] Pick list generated!")
    console.print(f"  Output: {output_file}")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(f"  [cyan]usortm report {project_dir}/[/cyan]  → Generate final report")
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
                "variant": row["variant"].split("|")[0],  # strip legacy |cons_check suffix
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
                "cons_check": row.get("cons_check", ""),
            })

    return well_data


def _load_library_order(project: dict) -> Optional[dict]:
    """Load variant ordering from the library/variants CSV.

    Returns a dict mapping variant name to its row index (0-based) in the
    original CSV, or None if the file isn't available.
    """
    variants_path = project.get("library_file") or project.get("variants_file")
    if not variants_path:
        return None

    variants_path = Path(variants_path)
    if not variants_path.exists():
        return None

    order = {}
    try:
        with open(variants_path, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                name = row.get("Name") or row.get("name") or row.get("variant")
                if name:
                    order[name] = idx
    except Exception:
        return None

    return order if order else None


def _load_targets(targets_file: Path) -> set:
    """Load target variants from CSV."""
    targets = set()

    with open(targets_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "variant" in row:
                targets.add(row["variant"])

    return targets


def _generate_pick_list(
    well_data: list,
    target_variants: Optional[set],
    unique_only: bool,
    target_format: int,
    fill_order: str,
    library_order: Optional[dict] = None,
    tier: Optional[str] = None,
    compact: bool = False,
) -> list:
    """Generate pick list from well data.

    When *library_order* is provided, the final pick list is sorted to
    match the input library CSV ordering.  The highest-read-count well
    is still chosen for each variant (when unique_only=True), but the
    output order reflects the library rather than read depth.

    When *tier* is set (A/B/C), wells are pre-filtered to meet the
    tier's minimum reads and consensus thresholds.

    When *compact* is True, empty placeholders for unrecovered variants
    are omitted so all recovered hits are packed into adjacent wells.
    """
    pick_list = []
    seen_variants = set()

    # Sort by reads (descending) to pick highest quality wells first
    sorted_wells = sorted(well_data, key=lambda x: x["reads"], reverse=True)

    # Apply tier filter
    if tier and tier in TIER_THRESHOLDS:
        thresh = TIER_THRESHOLDS[tier]
        sorted_wells = [
            w for w in sorted_wells
            if w["reads"] >= thresh["min_reads"]
            and w["consensus_fraction"] > thresh["min_consensus"]
        ]

    for well in sorted_wells:
        variant = well["variant"]

        # Filter by target variants if specified
        if target_variants and variant not in target_variants:
            continue

        # Skip if we've already picked this variant and unique_only is True
        if unique_only and variant in seen_variants:
            continue

        pick_list.append({
            "variant": variant,
            "source_plate": well["plate"],
            "source_well": well["well"],
            "reads": well["reads"],
            "consensus_fraction": well["consensus_fraction"],
        })

        seen_variants.add(variant)

    # Re-sort by library ordering if available.
    if library_order:
        max_idx = len(library_order)

        if not compact:
            # Default: insert empty placeholders for unrecovered variants so
            # the pick plate preserves library order with gaps.
            for variant_name, _idx in sorted(library_order.items(), key=lambda x: x[1]):
                if variant_name not in seen_variants:
                    pick_list.append({
                        "variant": variant_name,
                        "source_plate": "",
                        "source_well": "",
                        "reads": 0,
                        "consensus_fraction": 0,
                        "empty": True,
                    })

        # Sort hits (and empties, if any) by library order
        pick_list.sort(
            key=lambda h: (library_order.get(h["variant"], max_idx), h["variant"])
        )

    # Assign target wells based on fill order
    _assign_target_wells(pick_list, target_format, fill_order)

    return pick_list


def _assign_target_wells(pick_list: list, target_format: int, fill_order: str):
    """Assign target plate and well positions."""
    if target_format == 96:
        rows, cols = 8, 12
    elif target_format == 384:
        rows, cols = 16, 24
    else:
        rows, cols = 16, 24  # Default to 384

    target_plate = 0
    well_index = 0

    for hit in pick_list:
        # Calculate target well position
        if fill_order == "column":
            # Fill column-wise (A1, B1, C1... then A2, B2, C2...)
            col = well_index // rows
            row = well_index % rows
        else:  # row
            # Fill row-wise (A1, A2, A3... then B1, B2, B3...)
            row = well_index // cols
            col = well_index % cols

        # Convert to well name (e.g., A1, B2, etc.)
        row_letter = chr(ord('A') + row)
        col_number = col + 1
        target_well = f"{row_letter}{col_number}"

        hit["target_plate"] = str(target_plate)
        hit["target_well"] = target_well

        well_index += 1

        # Move to next plate if current is full
        if well_index >= rows * cols:
            target_plate += 1
            well_index = 0


def _save_pick_list(pick_list: list, output_file: Path, volume: float):
    """Save pick list in Integra ASSIST PLUS format."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        # Header for Integra ASSIST PLUS
        writer.writerow([
            "SampleID",
            "SourcePlateID",
            "SourceWell",
            "TargetPlateID",
            "TargetWell",
            "TransferVolume",
        ])

        for hit in pick_list:
            # Empty wells (unrecovered variants) get 0 volume so the
            # Integra ASSIST PLUS skips them while preserving plate layout.
            vol = 0.0 if hit.get("empty") else volume
            writer.writerow([
                hit["variant"],
                hit["source_plate"],
                hit["source_well"],
                hit["target_plate"],
                hit["target_well"],
                f"{vol:.1f}",
            ])
