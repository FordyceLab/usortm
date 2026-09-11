"""Merge pick results from multiple sequencing rounds into a unified pick list."""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import csv
import json

from usortm import provenance as _provenance

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE
from usortm.demux.utils import column_agreement_class
from usortm.paths import INTEGRA_DIRNAME, input_file

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
    max_disagreement: Optional[float] = typer.Option(
        None,
        "--max-disagreement",
        min=0.0, max=1.0,
        help="Exclude a well when more than this fraction of its reads "
             "disagree with the reference at any one position, in every "
             "round. Default: the limit each round's own pick recorded, or "
             "the mixed-template threshold where it recorded none.",
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
        help="Fill order for target plate (row or column). Ignored with --layout.",
    ),
    layout: Optional[Path] = typer.Option(
        None,
        "--layout",
        help=(
            "CSV giving the destination well for each variant. Defaults to "
            "the project's inputs/final_plate_layout/*.csv when there is one; "
            "pass --layout '' to fill sequentially instead."
        ),
    ),
):
    """
    Merge pick results from multiple sequencing rounds.

    Combines results from all rounds, filling gaps from earlier rounds with
    results from later rounds. Preserves the original library order.

    The Integra files are one per source plate, named by round and plate
    (integra_assist_R1_plate3.csv, integra_assist_R2_plate1.csv), so plates
    from different rounds are told apart by file; the SourcePlateID in the
    cells is the plate number alone, which is how the robot reads it.

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

    # The disagreement limit each round is held to: the one its pick applied,
    # unless one is given here for all of them.
    limits = _round_limits(project, list(all_wells), max_disagreement)
    for rnum in sorted(all_wells):
        lim = limits.get(rnum)
        console.print(
            f"[green]✓[/green] Round {rnum}: worst-column disagreement "
            + (f"limited to {lim:.0%} (from its pick)" if lim is not None
               and max_disagreement is None else
               f"limited to {lim:.0%}" if lim is not None else
               "limited to the mixed-template threshold")
        )

    # Where a round was assembled in 96-well plates and consolidated for
    # sequencing, the robot picks from the former: carry the bench position.
    from usortm.verify import bench_positions_for_round

    bench_by_round = {rnum: bench_positions_for_round(project_dir, rnum)
                      for rnum in all_wells}
    for rnum, bench in sorted(bench_by_round.items()):
        if bench:
            colonies = sorted({rep for rep, _ in bench.values()})
            console.print(
                f"[green]✓[/green] Round {rnum}: picked from its "
                f"{len(colonies)} colony plate(s), in 96-well positions"
            )

    # Build merged pick list (best well per variant across all rounds)
    pick_list = _build_merged_pick_list(all_wells, library_order, tier, limits,
                                        bench_by_round)

    if not any(not h.get("empty") for h in pick_list):
        console.print("[yellow]Warning:[/yellow] No hits found after merge!")
        raise typer.Exit(1)

    # Upgrade empty placeholders with streakout-recoverable variants from all rounds
    streakout_map: dict = {}  # variant -> best source info
    for rnum in round_nums:
        rinfo = available[rnum]
        so_csv = project_dir / rinfo["demux_output"] / "streakout" / "streakout_candidates.csv"
        if not so_csv.exists():
            continue
        with open(so_csv) as _sf:
            for _row in csv.DictReader(_sf):
                groups = json.loads(_row.get("groups_json", "[]"))
                for g in groups:
                    variant = g.get("variant", "")
                    if not g.get("is_recoverable"):
                        continue
                    reads = int(g.get("reads", 0))
                    if variant not in streakout_map or reads > streakout_map[variant]["reads"]:
                        pileup_html = (
                            f"../../{rinfo['demux_output']}/streakout/"
                            f"well_{_row['plate']}_{_row['well']}.html"
                        )
                        streakout_map[variant] = {
                            "source_plate": f"R{rnum}_{_row['plate']}",
                            "source_well": _row["well"],
                            "reads": reads,
                            "frac": float(g.get("frac", 0)),
                            "pileup_url": pileup_html,
                            "source_round": rnum,
                        }

    if streakout_map:
        picked_variants = {h["variant"] for h in pick_list if not h.get("empty")}
        n_upgraded = 0
        for h in pick_list:
            if (
                h.get("empty")
                and h["variant"] in streakout_map
                and h["variant"] not in picked_variants
            ):
                info = streakout_map[h["variant"]]
                h.update({
                    "source_plate": info["source_plate"],
                    "source_well": info["source_well"],
                    "reads": info["reads"],
                    "consensus_fraction": info["frac"],
                    "pileup_url": info["pileup_url"],
                    "tier_override": "Streakout",
                    "source_round": info["source_round"],
                    "empty": False,
                })
                n_upgraded += 1
        if n_upgraded:
            console.print(
                f"[cyan]↑[/cyan] {n_upgraded} streakout-recoverable variant(s) "
                f"added to pick plate (blue)"
            )

    # Onto the layout the library was designed for, when there is one.  A
    # sequential fill puts each variant wherever the library order happens to
    # reach it, so the same library merged twice lands in different wells and
    # neither matches the plate the bench keeps.
    layout_rows, layout_stats = _resolve_layout(layout, project_dir)
    if layout_rows:
        from usortm.cli.pick import _apply_layout

        layout_stats = _apply_layout(pick_list, layout_rows)
        console.print(
            f"[green]\u2713[/green] Destination layout: "
            f"{layout_stats['filled']} of "
            f"{layout_stats['filled'] + layout_stats['not_recovered']} "
            f"designed wells filled"
            + (f", {layout_stats['designed_blank']} left blank by the design"
               if layout_stats["designed_blank"] else "")
        )
        unplaced = layout_stats.get("unplaced") or []
        if unplaced:
            shown = ", ".join(unplaced[:6])
            more = f" and {len(unplaced) - 6} more" if len(unplaced) > 6 else ""
            console.print(
                f"[yellow]![/yellow] {len(unplaced)} recovered variant(s) have "
                f"no well in the layout and were dropped: {shown}{more}"
            )
    else:
        _assign_target_wells(pick_list, target_format, fill_order)

    # Write outputs
    merged_dir = project_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    pick_dir = merged_dir / "pick"
    pick_dir.mkdir(exist_ok=True)
    integra_dir = project_dir / INTEGRA_DIRNAME
    integra_dir.mkdir(exist_ok=True)

    # One file per plate the robot is loaded with: a round's sequenced plates,
    # or its colony plates where it was arrayed from them.
    source_plates = set()
    for rnum, wells in all_wells.items():
        bench = bench_by_round.get(rnum) or {}
        if bench:
            source_plates |= {f"R{rnum}_{rep}" for rep, _ in bench.values()}
        else:
            source_plates |= {f"R{rnum}_{w['plate']}" for w in wells}
    written_files = _save_pick_list(pick_list, integra_dir, volume,
                                    source_plates=source_plates)
    _write_integra_readme(
        integra_dir, written_files, volume, target_format, round_nums,
        bench_rounds=[r for r, b in sorted(bench_by_round.items()) if b],
    )

    # Save combined well_assignments for the merged report
    _save_merged_well_assignments(all_wells, merged_dir)

    # Save pick list as JSON for the report to build detail tables
    pick_list_file = merged_dir / "pick_list.json"
    with open(pick_list_file, "w") as f:
        json.dump(pick_list, f, indent=2)

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
    _streakout_hits = [h for h in recovered if h.get("tier_override") == "Streakout"]
    project["merged"] = {
        "completed": True,
        "command": _provenance.current_command(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "rounds": round_nums,
        "total_hits": len(recovered),
        "unique_variants": len(set(h["variant"] for h in recovered)),
        "streakout_variants": len(_streakout_hits),
        "tier": tier or "none",
        "max_disagreement": {str(r): v for r, v in limits.items()},
    }
    with open(state_file, "w") as f:
        json.dump(project, f, indent=2)
    _provenance.refresh_commands(project, project_dir)

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
    if written_files:
        console.print(
            f"[green]\u2713[/green] Integra files: {len(written_files)}, one "
            f"per source plate, in {written_files[0].parent}/"
        )
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
                # Carried because the quality test reads them.  Dropping them
                # here did not fail the test, it passed it: a missing flank
                # reads as intact and a missing worst-column figure as clean,
                # so every well cleared a filter that never ran.
                "flank_check": row.get("flank_check", ""),
                "max_mismatch_frac": (
                    float(row["max_mismatch_frac"])
                    if (row.get("max_mismatch_frac") or "").strip() else None),
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
    candidates.append(input_file(project_dir, "variants.csv"))

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


def _norm_variant_name(name: str) -> str:
    """One spelling for a variant across rounds.

    Some rounds separate a name's parts with "." and the top-level library
    with ";", so the same construct arrives written two ways and would be
    merged as two.
    """
    return str(name or "").replace(".", ";")


def _passes_tier(well: dict, tier: Optional[str],
                 designed: Optional[set] = None) -> bool:
    """Whether the well holds a library member, cleanly, at the tier's depth.

    *designed* is the library's members.  Without it the merge took whatever
    a well was called, so the parent -- which every plate carries and which is
    not a variant of anything -- was picked as a hit and counted towards
    coverage, putting a 376-member library over 100%.

    The quality test is the one the recovery tiers and the plate maps use.
    Judged separately here it excluded error calls but never flanks or how far
    the worst column disagreed, so wells the rest of the report had discarded
    were merged as recovered.
    """
    if designed is not None:
        from usortm.report.plates import carries_designed_sequence

        row = dict(well)
        row["variant"] = _norm_variant_name(well.get("variant"))
        if not carries_designed_sequence(row, designed):
            return False
    elif well.get("cons_check", "") in ("Error", "Other Error"):
        return False
    if tier is None:
        return True
    thresh = TIER_THRESHOLDS[tier]
    return (
        well["reads"] >= thresh["min_reads"]
        and well["consensus_fraction"] > thresh["min_consensus"]
    )


def _round_limits(project: dict, round_nums: list,
                  override: Optional[float]) -> dict:
    """The largest per-position disagreement a picked well may carry, by round.

    *override* applies to every round.  Otherwise each round's limit is the
    one its own pick recorded (``workflow_steps.pick.max_disagreement``), so
    the merged plate is built by the policy the round's plate was; None
    means the mixed-template threshold, as it always did.  A merge that
    ranked and filtered by a rule of its own put back the ten watch-band
    wells a strict pick had just removed.
    """
    limits = {}
    for rnum in round_nums:
        if override is not None:
            limits[rnum] = override
            continue
        if rnum == 1:
            step = (project.get("workflow_steps") or {}).get("pick") or {}
        else:
            step = ((project.get("rounds") or {}).get(str(rnum)) or {}) \
                .get("workflow_steps", {}).get("pick") or {}
        limits[rnum] = step.get("max_disagreement")
    return limits


_AGREEMENT_RANK = {"clean": 0, "unknown": 1, "watch": 2, "mixed": 3}


def _well_rank(well: dict) -> tuple:
    """Better wells sort first: reads that agree, then more of them.

    The same order the pick uses within a consensus category
    (:func:`usortm.cli.pick._generate_pick_list`).  The merge only sees wells
    that already pass the consensus test, so the category is not needed.
    """
    agreement = _AGREEMENT_RANK.get(
        column_agreement_class(well.get("max_mismatch_frac")), 1)
    return (agreement, -int(well.get("reads") or 0))


def _within_limit(well: dict, limit: Optional[float]) -> bool:
    """Whether a well's worst column is within the round's limit.

    A well without a measurement is kept: the limit is about what was seen.
    """
    if limit is None:
        return column_agreement_class(well.get("max_mismatch_frac")) != "mixed"
    mmf = well.get("max_mismatch_frac")
    if mmf is None:
        return True
    return float(mmf) <= limit


def _bench_fields(rnum: int, well: dict, bench_by_round: Optional[dict]) -> dict:
    """The colony plate and 96-well position of a re-ordered well, if known.

    A re-order round's constructs were assembled in 96-well plates and
    consolidated into a 384-well plate for sequencing; the robot picks from
    the former.  Where the round describes that arraying, the merged hit
    carries ``bench_plate`` (``R<round>_<colony plate>``) and ``bench_well``
    beside the sequenced coordinates, and the Integra files are written in
    them.
    """
    bench = (bench_by_round or {}).get(rnum) or {}
    try:
        key = (int(well["plate"]), str(well["well"]).strip().upper())
    except (TypeError, ValueError):
        return {}
    found = bench.get(key)
    if not found:
        return {}
    return {"bench_plate": f"R{rnum}_{found[0]}", "bench_well": found[1]}


def _build_merged_pick_list(
    all_wells: dict[int, list],
    library_order: Optional[dict],
    tier: Optional[str],
    limits: Optional[dict] = None,
    bench_by_round: Optional[dict] = None,
) -> list:
    """Build the merged pick list.

    For each variant, selects the best well across all rounds that passes
    the tier filter and the round's disagreement limit: the well whose
    reads agree with each other first, and the deeper one among those.
    Source plate IDs are prefixed with round number: R{N}_{plate}.
    """
    limits = limits or {}
    # Find best well per variant across all rounds.
    # Normalize variant names: some rounds use "." as separator (e.g. "ATF4.25.171")
    # while the top-level library uses ";" (e.g. "ATF4;25;171"). Normalize to ";"
    # so round 2+ variants are correctly matched to their library position.
    _norm = _norm_variant_name

    # The library's own members, in the one spelling.  A well is only a hit
    # if it holds one of these: the parent is on every plate and is not a
    # variant of anything.
    designed = ({_norm(k) for k in library_order} if library_order else None)

    best: dict[str, tuple[int, dict]] = {}  # canonical_name -> (round_num, well)
    for rnum, wells in sorted(all_wells.items()):
        limit = limits.get(rnum)
        for well in wells:
            if not _passes_tier(well, tier, designed):
                continue
            if not _within_limit(well, limit):
                continue
            variant = _norm(well["variant"])
            if variant not in best or _well_rank(well) < _well_rank(best[variant][1]):
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
                    **_bench_fields(rnum, well, bench_by_round),
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
                    **_bench_fields(rnum, well, bench_by_round),
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


def _resolve_layout(layout: "Path | None", project_dir: Path):
    """The destination layout to merge onto, and what reading it said.

    Given explicitly it is used as given.  Otherwise the project's own
    ``inputs/final_plate_layout`` is looked in: a library designed to sit in
    named wells keeps that layout in the project, and a merge that ignored it
    would hand the bench a plate that does not match the one it keeps.

    Passing an empty path turns it off, which is the way back to a sequential
    fill for a project that has a layout it does not want used.

    Returns ``(rows, stats)`` with rows None when nothing is to be applied.
    """
    from usortm.cli.pick import LayoutError, _load_layout

    # An empty --layout is the way to turn it off.  Path("") stringifies to
    # ".", so the empty string alone never matched what typer hands over.
    if layout is not None and str(layout) in ("", "."):
        return None, {}

    candidate = layout
    if candidate is None:
        found = sorted((project_dir / "inputs" / "final_plate_layout")
                       .glob("*.csv"))
        if not found:
            return None, {}
        candidate = found[0]
        console.print(f"[green]\u2713[/green] Using the project's destination "
                      f"layout: {candidate.name}")

    try:
        return _load_layout(candidate), {}
    except LayoutError as exc:
        # A layout that cannot be read is worth stopping for: filling
        # sequentially instead would quietly produce a different plate.
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


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


def _save_pick_list(pick_list: list, output_dir: Path, volume: float,
                    source_plates=None):
    """Save the merged pick in Integra ASSIST PLUS format, one file per
    source plate, named by round and plate (``integra_assist_R2_plate3.csv``).

    See :mod:`usortm.integra`.  Streak-out entries are left out: they are
    instructions for a person, not transfers for the robot.
    """
    from usortm.integra import write_integra_files

    return write_integra_files(pick_list, output_dir, volume,
                               source_plates=source_plates)


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
    hitlist_files: list,
    volume: float,
    target_format: int,
    round_nums: list,
    bench_rounds: Optional[list] = None,
):
    """Write README explaining the merged Integra ASSIST input."""
    rounds_str = ", ".join(str(r) for r in round_nums)
    bench_line = (
        "\n  • Round(s) " + ", ".join(str(r) for r in bench_rounds)
        + " were arrayed from 96-well colony plates: their files are named\n"
        "    R<round>_plate<colony plate>, SourcePlateID is the colony plate number\n"
        "    and SourceWell the 96-well position."
        if bench_rounds else ""
    )
    files_str = "\n".join(f"  • {f.name}" for f in hitlist_files)
    content = f"""\
Integra ASSIST PLUS — Merged Hit-Picking Input
===============================================

Files (one per source plate; load that plate, run its file):
{files_str}
Rounds merged: {rounds_str}

File names carry the round and plate: integra_assist_R{{round}}_plate{{N}}.csv.
SourcePlateID in the cells is the plate number alone; the robot reads that
column as a number and rejects a prefixed one.
  e.g. R1_3  = Round 1, Plate 3
       R2_4  = Round 2, Plate 4

Columns
-------
  SampleID       Variant name
  SourcePlateID  Source plate number; the round is in the file name
  SourceWell     Source well position (e.g. A1)
  TargetPlateID  Destination plate number
  TargetWell     Destination well position
  TransferVolume Transfer volume in µL

Settings used
-------------
  Transfer volume : {volume:g} µL
  Target format   : {target_format}-well plate

Notes
-----
  • Each file holds the transfers out of one source plate; a file with
    only a header is a plate with nothing to pick.
  • SampleID is the variant name; a stop codon's * is written tag (the amber
    codon), since the robot software rejects the character: K16* is K16tag.{bench_line}
  • Round 1 and Round 2+ source plates are physically separate — load
    them separately when the robot requests each SourcePlateID group.
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
