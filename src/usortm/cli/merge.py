"""Merge pick results from multiple sequencing rounds into a unified pick list."""
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
    "B": {"min_reads": 50,  "min_consensus": 0.9},
    "C": {"min_reads": 20,  "min_consensus": 0.9},
}
TIER_ORDER = ["A", "B", "C"]


def merge(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    rounds: Optional[str] = typer.Option(
        None,
        "--rounds", "-r",
        help="Comma-separated round numbers to merge (e.g. '1,2'). Default: all completed rounds.",
    ),
    tier: str = typer.Option(
        "C",
        "--tier",
        help="Minimum quality tier (A/B/C). Use '' to disable filtering.",
    ),
    volume: float = typer.Option(
        5.0,
        "--volume", "-v",
        help="Transfer volume in µL for Integra ASSIST output.",
    ),
    target_format: int = typer.Option(
        384,
        "--target-format",
        help="Target plate format (96 or 384).",
    ),
    fill_order: str = typer.Option(
        "row",
        "--fill-order",
        help="Fill order for target plate (row or column).",
    ),
):
    """
    Merge pick results from multiple sequencing rounds.

    Combines results from all rounds, filling gaps from earlier rounds with
    results from later rounds. Preserves the original library order.

    Source plate IDs are prefixed with the round number (e.g. R1_3, R2_4)
    so the Integra ASSIST can distinguish plates from different rounds.

    [bold]Example:[/bold]

        usortm merge my_project/               # merge all completed rounds
        usortm merge my_project/ --rounds 1,2  # only merge rounds 1 and 2
    """
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project (missing {PROJECT_STATE_FILE})")
        raise typer.Exit(1)

    with open(state_file) as f:
        project = json.load(f)

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Round Merge",
        border_style=BORDER_STYLE,
    ))
    console.print()

    # Determine available demuxed rounds
    available = _get_demuxed_rounds(project, project_dir)
    if not available:
        console.print("[red]Error:[/red] No completed demux rounds found.")
        console.print("Run 'usortm demux' for at least one round first.")
        raise typer.Exit(1)

    if rounds:
        selected = [int(r.strip()) for r in rounds.split(",")]
        missing = [r for r in selected if r not in available]
        if missing:
            console.print(f"[red]Error:[/red] Rounds not found or not demuxed: {missing}")
            console.print(f"Available rounds: {sorted(available.keys())}")
            raise typer.Exit(1)
        round_nums = selected
    else:
        round_nums = sorted(available.keys())

    console.print(f"[green]\u2713[/green] Merging rounds: {round_nums}")

    # Validate tier
    if tier and tier.strip():
        tier = tier.upper()
        if tier not in TIER_THRESHOLDS:
            console.print(f"[red]Error:[/red] Invalid tier '{tier}'. Choose A, B, or C.")
            raise typer.Exit(1)
        thresh = TIER_THRESHOLDS[tier]
        console.print(
            f"[green]\u2713[/green] Tier {tier} filter: "
            f"\u2265{thresh['min_reads']} reads, >{thresh['min_consensus']:.0%} consensus"
        )
    else:
        tier = None

    # Load well assignments for each round
    all_wells: dict[int, list] = {}
    for rnum in round_nums:
        rinfo = available[rnum]
        wa_file = project_dir / rinfo["demux_output"] / "well_assignments.csv"
        if not wa_file.exists():
            console.print(f"[yellow]Warning:[/yellow] Round {rnum} well_assignments.csv not found, skipping.")
            continue
        all_wells[rnum] = _load_well_assignments(wa_file)
        console.print(f"[green]\u2713[/green] Round {rnum}: {len(all_wells[rnum])} wells loaded")

    if not all_wells:
        console.print("[red]Error:[/red] No well data loaded from any round.")
        raise typer.Exit(1)

    # Load full library order from top-level variants.csv
    library_order = _load_library_order(project, project_dir)
    if library_order is None:
        console.print("[yellow]Warning:[/yellow] Could not load library variants. Output will not be ordered.")
    else:
        console.print(f"[green]\u2713[/green] Library order loaded ({len(library_order)} variants)")

    # Build merged pick list (best well per variant across all rounds)
    pick_list = _build_merged_pick_list(all_wells, library_order, tier)

    if not any(not h.get("empty") for h in pick_list):
        console.print("[yellow]Warning:[/yellow] No hits found after merge!")
        raise typer.Exit(1)

    _assign_target_wells(pick_list, target_format, fill_order)

    # Write outputs
    merged_dir = project_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    pick_dir = merged_dir / "pick"
    pick_dir.mkdir(exist_ok=True)
    integra_dir = pick_dir / "Integra ASSIST Input"
    integra_dir.mkdir(exist_ok=True)

    output_file = integra_dir / "hitlist_integra_assist_merged.csv"
    _save_pick_list(pick_list, output_file, volume)
    _write_integra_readme(integra_dir, output_file, volume, target_format, round_nums)

    # Save combined well_assignments for the merged report
    _save_merged_well_assignments(all_wells, merged_dir)

    # Generate interactive pick plate map (Bokeh optional)
    try:
        from usortm.demux.viz import save_pick_plate_map_html
        pick_map_path = pick_dir / "pick_plate_map.html"
        pileup_url_map = _build_merged_pileup_url_map(pick_list, project_dir)
        save_pick_plate_map_html(
            pick_list, str(pick_map_path),
            title="Merged Pick Plate Map",
            target_format=target_format,
            pileup_url_map=pileup_url_map or None,
        )
        n_pileups = sum(len(v) for v in pileup_url_map.values())
        console.print(
            f"[green]\u2713[/green] Pick plate map saved to {pick_map_path}"
            + (f" ({n_pileups} wells with pileup links)" if n_pileups else "")
        )
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not generate pick plate map: {e}")

    # Update master project state
    recovered = [h for h in pick_list if not h.get("empty")]
    project["merged"] = {
        "completed": True,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "rounds": round_nums,
        "total_hits": len(recovered),
        "unique_variants": len(set(h["variant"] for h in recovered)),
        "tier": tier or "none",
    }
    with open(state_file, "w") as f:
        json.dump(project, f, indent=2)

    # Display summary
    console.print()
    summary_table = Table(
        title="Merge Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="muted")
    summary_table.add_column("Value", justify="right")

    unique_variants = len(set(h["variant"] for h in recovered))
    empty_count = len(pick_list) - len(recovered)
    library_size = len(library_order) if library_order else 0

    summary_table.add_row("Rounds merged", ", ".join(str(r) for r in round_nums))
    summary_table.add_row("Total hits", str(len(recovered)))
    summary_table.add_row("Unique variants", str(unique_variants))
    if library_size:
        coverage = unique_variants / library_size * 100
        summary_table.add_row("Library coverage", f"{coverage:.1f}%")
    if empty_count > 0:
        summary_table.add_row("Still missing", str(empty_count))
    summary_table.add_row("Quality tier", tier or "no filter")

    for rnum in round_nums:
        contributed = sum(1 for h in recovered if h.get("source_round") == rnum)
        summary_table.add_row(f"  \u2192 From round {rnum}", str(contributed))

    console.print(summary_table)
    console.print()
    console.print(f"[green]\u2713[/green] Merged pick list: {output_file}")
    console.print(f"[green]\u2713[/green] Combined well assignments: {merged_dir / 'well_assignments.csv'}")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(f"  [cyan]usortm report {project_dir}/ --round merged[/cyan]  \u2192 Merged report")
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_demuxed_rounds(project: dict, project_dir: Path) -> dict:
    """Return {round_num: round_info} for all rounds that have well_assignments.csv.

    Round 1 uses the top-level demux_output/. Rounds 2+ use rounds/N/demux_output/.
    """
    available: dict[int, dict] = {}

    # Round 1: top-level demux_output
    r1_wa = project_dir / "demux_output" / "well_assignments.csv"
    if r1_wa.exists():
        available[1] = {
            "demux_output": "demux_output",
            "pick_dir": "pick",
            "variants_file": project.get("variants_file", "variants.csv"),
            "library_size": project.get("library_size", 0),
        }

    # Rounds N > 1: from project["rounds"]
    for rnum_str, rinfo in project.get("rounds", {}).items():
        rnum = int(rnum_str)
        if rnum == 1:
            continue
        demux_output = rinfo.get("demux_output", f"rounds/{rnum}/demux_output")
        if (project_dir / demux_output / "well_assignments.csv").exists():
            available[rnum] = rinfo

    return available


def _load_well_assignments(wa_file: Path) -> list:
    """Load a well_assignments.csv into a list of dicts."""
    wells = []
    with open(wa_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wells.append({
                "plate": row["plate"],
                "well": row["well"],
                "variant": row["variant"].split("|")[0],
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
                "cons_check": row.get("cons_check", ""),
            })
    return wells


def _load_library_order(project: dict, project_dir: Path) -> Optional[dict]:
    """Load variant ordering from the top-level library CSV.

    Returns an {name: index} dict in original library order, or None.
    """
    candidates = []
    vf = project.get("variants_file")
    if vf:
        candidates.append(Path(vf))
    candidates.append(project_dir / "variants.csv")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            order: dict[str, int] = {}
            with open(candidate, newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    name = (
                        row.get("Name") or row.get("name")
                        or row.get("variant") or row.get("variant_name")
                    )
                    if name:
                        order[name] = idx
            return order or None
        except Exception:
            pass
    return None


def _passes_tier(well: dict, tier: Optional[str]) -> bool:
    """Return True if the well meets the tier threshold."""
    if tier is None:
        return True
    thresh = TIER_THRESHOLDS[tier]
    return (
        well["reads"] >= thresh["min_reads"]
        and well["consensus_fraction"] > thresh["min_consensus"]
    )


def _build_merged_pick_list(
    all_wells: dict[int, list],
    library_order: Optional[dict],
    tier: Optional[str],
) -> list:
    """Build the merged pick list.

    For each variant, selects the best well (highest reads) across all rounds
    that passes the tier filter. Source plate IDs are prefixed with round
    number: R{N}_{plate}.
    """
    # Find best well per variant across all rounds.
    # Normalize variant names: some rounds use "." as separator (e.g. "ATF4.25.171")
    # while the top-level library uses ";" (e.g. "ATF4;25;171"). Normalize to ";"
    # so round 2+ variants are correctly matched to their library position.
    def _norm(name: str) -> str:
        return name.replace(".", ";")

    best: dict[str, tuple[int, dict]] = {}  # canonical_name -> (round_num, well)
    for rnum, wells in sorted(all_wells.items()):
        for well in wells:
            if not _passes_tier(well, tier):
                continue
            variant = _norm(well["variant"])
            if variant not in best or well["reads"] > best[variant][1]["reads"]:
                best[variant] = (rnum, well)

    # Build a normalized lookup for library_order so "." and ";" variants both match
    norm_library_order: dict[str, str] = {}  # normalized_name -> original_key
    if library_order:
        for k in library_order:
            norm_library_order[_norm(k)] = k

    pick_list: list[dict] = []

    if library_order:
        seen: set[str] = set()
        for variant_name in sorted(library_order, key=lambda v: library_order[v]):
            # Look up by normalized name; fall back to original
            norm_name = _norm(variant_name)
            if norm_name in best:
                rnum, well = best[norm_name]
                pick_list.append({
                    "variant": variant_name,
                    "source_plate": f"R{rnum}_{well['plate']}",
                    "source_well": well["well"],
                    "reads": well["reads"],
                    "consensus_fraction": well["consensus_fraction"],
                    "source_round": rnum,
                })
            else:
                pick_list.append({
                    "variant": variant_name,
                    "source_plate": "",
                    "source_well": "",
                    "reads": 0,
                    "consensus_fraction": 0,
                    "source_round": None,
                    "empty": True,
                })
            seen.add(norm_name)

        # Append any variants found in wells but absent from library (e.g. controls)
        for norm_name, (rnum, well) in sorted(best.items()):
            if norm_name not in seen:
                # Use the original library name if available, else the normalized name
                display_name = norm_library_order.get(norm_name, norm_name)
                pick_list.append({
                    "variant": display_name,
                    "source_plate": f"R{rnum}_{well['plate']}",
                    "source_well": well["well"],
                    "reads": well["reads"],
                    "consensus_fraction": well["consensus_fraction"],
                    "source_round": rnum,
                })
    else:
        for variant_name, (rnum, well) in sorted(best.items()):
            pick_list.append({
                "variant": variant_name,
                "source_plate": f"R{rnum}_{well['plate']}",
                "source_well": well["well"],
                "reads": well["reads"],
                "consensus_fraction": well["consensus_fraction"],
                "source_round": rnum,
            })

    return pick_list


def _assign_target_wells(pick_list: list, target_format: int, fill_order: str):
    """Assign target plate/well positions (mirrors pick.py logic)."""
    rows, cols = (8, 12) if target_format == 96 else (16, 24)
    target_plate = 0
    well_index = 0

    for hit in pick_list:
        if fill_order == "column":
            col = well_index // rows
            row = well_index % rows
        else:
            row = well_index // cols
            col = well_index % cols

        hit["target_plate"] = str(target_plate)
        hit["target_well"] = f"{chr(ord('A') + row)}{col + 1}"

        well_index += 1
        if well_index >= rows * cols:
            target_plate += 1
            well_index = 0


def _save_pick_list(pick_list: list, output_file: Path, volume: float):
    """Save merged pick list in Integra ASSIST PLUS format."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "SampleID", "SourcePlateID", "SourceWell",
            "TargetPlateID", "TargetWell", "TransferVolume",
        ])
        for hit in pick_list:
            vol = 0.0 if hit.get("empty") else volume
            writer.writerow([
                hit["variant"],
                hit["source_plate"],
                hit["source_well"],
                hit["target_plate"],
                hit["target_well"],
                f"{vol:.1f}",
            ])


def _save_merged_well_assignments(all_wells: dict[int, list], merged_dir: Path):
    """Write a combined well_assignments.csv with an added 'round' column."""
    output_file = merged_dir / "well_assignments.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "plate", "well", "variant", "reads", "consensus_fraction", "cons_check"])
        for rnum, wells in sorted(all_wells.items()):
            for w in wells:
                writer.writerow([
                    rnum,
                    w["plate"],
                    w["well"],
                    w["variant"],
                    w["reads"],
                    w["consensus_fraction"],
                    w.get("cons_check", ""),
                ])


def _write_integra_readme(
    integra_dir: Path,
    hitlist_file: Path,
    volume: float,
    target_format: int,
    round_nums: list,
):
    """Write README explaining the merged Integra ASSIST input."""
    rounds_str = ", ".join(str(r) for r in round_nums)
    content = f"""\
Integra ASSIST PLUS — Merged Hit-Picking Input
===============================================

File: {hitlist_file.name}
Rounds merged: {rounds_str}

SourcePlateID format: R{{round}}_{{plate_number}}
  e.g. R1_3  = Round 1, Plate 3
       R2_4  = Round 2, Plate 4

Columns
-------
  SampleID       Variant name
  SourcePlateID  Source plate (round-prefixed)
  SourceWell     Source well position (e.g. A1)
  TargetPlateID  Destination plate number
  TargetWell     Destination well position
  TransferVolume Transfer volume in µL (0 = empty/unrecovered well)

Settings used
-------------
  Transfer volume : {volume:.1f} µL
  Target format   : {target_format}-well plate

Notes
-----
  • Load source plates in the order indicated by SourcePlateID.
  • Round 1 and Round 2+ source plates are physically separate — load
    them separately when the robot requests each SourcePlateID group.
  • Wells with TransferVolume = 0 are unrecovered library variants
    preserved to maintain library layout. The Integra ASSIST PLUS
    will skip these wells automatically.
"""
    (integra_dir / "README.txt").write_text(content)


def _build_merged_pileup_url_map(pick_list: list, project_dir: Path) -> dict:
    """Build pileup_url_map for the merged pick plate map.

    Maps target_plate (str) -> target_well -> absolute file:// URL of the
    pileup HTML. Uses pre-generated pileup files from each round's pick
    directory, so no re-computation is needed.

    Round 1 pileup files: {project_dir}/pick/pileup/well_{plate}_{well}.html
    Round N pileup files: {project_dir}/rounds/{N}/pick/pileup/well_{plate}_{well}.html
    """
    pileup_map: dict[str, dict[str, str]] = {}

    for hit in pick_list:
        if hit.get("empty") or not hit.get("source_round"):
            continue
        rnum = hit["source_round"]
        # source_plate is "R{n}_{plate}", extract the raw plate number
        sp = hit["source_plate"]  # e.g. "R1_3"
        raw_plate = sp.split("_", 1)[1] if "_" in sp else sp
        source_well = hit["source_well"]
        target_plate = str(hit.get("target_plate", "0"))
        target_well = hit.get("target_well", "")

        if not target_well:
            continue

        if rnum == 1:
            pileup_file = project_dir / "pick" / "pileup" / f"well_{raw_plate}_{source_well}.html"
        else:
            pileup_file = (
                project_dir / "rounds" / str(rnum) / "pick" / "pileup"
                / f"well_{raw_plate}_{source_well}.html"
            )

        if pileup_file.exists():
            # Use absolute file:// URL so links work regardless of where the
            # merged pick_plate_map.html is embedded (srcdoc or standalone).
            pileup_map.setdefault(target_plate, {})[target_well] = (
                pileup_file.resolve().as_uri()
            )

    return pileup_map
