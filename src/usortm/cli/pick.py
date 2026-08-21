"""Generate hit-picking lists from demultiplexing results."""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import csv
import json

import typer
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE, section
from usortm.demux.utils import MIXED_TEMPLATE_THRESHOLD
from usortm.paths import input_file

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
        help="Fill order for target plate (row or column). Ignored with --layout.",
    ),
    layout: Optional[Path] = typer.Option(
        None,
        "--layout",
        help="CSV giving the well each variant is picked into, for a "
             "destination plate that has already been designed. Needs a well "
             "column and a variant column; a row naming no variant is left "
             "blank. Overrides --target-format, --fill-order and --compact.",
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
    pileups: bool = typer.Option(
        True,
        "--pileups/--no-pileups",
        help="Generate per-well pileup HTML visualizations (disable to speed up pick).",
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Number of parallel workers for pileup generation.",
    ),
    include_flank_errors: bool = typer.Option(
        False,
        "--include-flank-errors/--exclude-flank-errors",
        help="Include wells with flanking region mismatches in pick list.",
    ),
    include_cons_errors: bool = typer.Option(
        False,
        "--include-cons-errors/--exclude-cons-errors",
        help="Include wells with consensus mismatches (Other Error/Error/Silent Mutation) in pick list.",
    ),
    round_num: int = typer.Option(
        1,
        "--round", "-r",
        help="Sequencing round to pick from (1 for initial sort, 2+ for re-order rounds).",
        min=1,
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

    # ------------------------------------------------------------------
    # Resolve round-specific context
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
        if not round_state.get("workflow_steps", {}).get("demux", {}).get("completed"):
            console.print(f"[red]Error:[/red] Round {round_num} demux not completed.")
            console.print(f"Run: [cyan]usortm demux {project_dir}/ --round {round_num}[/cyan]")
            raise typer.Exit(1)
        demux_output = round_dir / "demux_output"
        pick_dir_base = round_dir / "pick"
        # For round > 1, use the round-specific variants for library order
        round_variants_file = round_dir / "variants.csv"
        effective_project = round_state
        effective_project_dir = round_dir
    else:
        # Check if demux has been run for round 1
        if "workflow_steps" not in project or not project["workflow_steps"].get("demux", {}).get("completed"):
            console.print("[red]Error:[/red] No demultiplexing results found.")
            console.print("Run 'usortm demux' first to process sequencing data.")
            raise typer.Exit(1)
        demux_output = project_dir / "demux_output"
        pick_dir_base = project_dir / "pick"
        round_variants_file = None
        effective_project = project
        effective_project_dir = project_dir

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Hit Picking",
        border_style=BORDER_STYLE,
    ))
    console.print()

    # Load demux results
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

    if not include_cons_errors:
        console.print("[green]\u2713[/green] Excluding wells with consensus errors (use --include-cons-errors to override)")

    well_data = _load_well_assignments(well_assignments_file)
    section(console, "Selection")

    console.print(f"[green]\u2713[/green] Loaded {len(well_data)} wells with data")

    # Load target variants if specified
    target_variants = None
    if targets:
        target_variants = _load_targets(targets)
        console.print(f"[green]\u2713[/green] Loaded {len(target_variants)} target variants")

    # Load library ordering from variants file (if available)
    # For round > 1, use the round-specific variants CSV so the per-round
    # pick list is ordered by the reorder set (not the full 500-variant library).
    if round_variants_file is not None and round_variants_file.exists():
        library_order, lib_path_tried = _load_library_order(
            {"variants_file": str(round_variants_file)},
            project_dir=effective_project_dir,
        )
    else:
        library_order, lib_path_tried = _load_library_order(project, project_dir=project_dir)
    if library_order is not None:
        console.print(f"[green]\u2713[/green] Library order loaded ({len(library_order)} variants)")
    else:
        if lib_path_tried:
            console.print(
                f"[yellow]Warning:[/yellow] Library variants file not found: {lib_path_tried}"
            )
            console.print(
                "  Empty placeholders for unrecovered variants will not be inserted.\n"
                "  Copy your variants CSV to the project directory as 'variants.csv' to enable ordering."
            )
        else:
            console.print(
                "[yellow]Warning:[/yellow] No library variants file configured. "
                "Empty placeholders will not be inserted."
            )

    # A designed destination plate, when one was given.
    layout_rows = None
    layout_stats: dict = {}
    if layout is not None:
        try:
            layout_rows = _load_layout(layout)
        except LayoutError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        plates = {row["plate"] for row in layout_rows}
        console.print(
            f"[green]✓[/green] Layout: {len(layout_rows)} wells across "
            f"{len(plates)} plate(s) from {layout.name}"
        )

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
        include_flank_errors=include_flank_errors,
        include_cons_errors=include_cons_errors,
        layout=layout_rows,
        layout_stats=layout_stats,
    )

    if layout_stats:
        console.print(
            f"  {layout_stats['filled']} of "
            f"{layout_stats['filled'] + layout_stats['not_recovered']} "
            f"designed wells filled"
            + (f", {layout_stats['designed_blank']} left blank by the design"
               if layout_stats["designed_blank"] else "")
        )
        unplaced = layout_stats.get("unplaced") or []
        if unplaced:
            shown = ", ".join(unplaced[:6])
            more = f" (+{len(unplaced) - 6} more)" if len(unplaced) > 6 else ""
            console.print(
                f"[yellow]⚠[/yellow] {len(unplaced)} recovered variant(s) have "
                f"no well in the layout and were dropped: {shown}{more}"
            )

    # Upgrade empty placeholders to Streakout entries where the variant can be
    # recovered by streaking out a mixed well.
    streakout_csv = demux_output / "streakout" / "streakout_candidates.csv"
    if streakout_csv.exists():
        # Build map: variant -> best source (most reads)
        streakout_map: dict = {}
        with open(streakout_csv) as _sf:
            for _row in csv.DictReader(_sf):
                groups = json.loads(_row.get("groups_json", "[]"))
                for g in groups:
                    variant = g.get("variant", "")
                    if not g.get("is_recoverable"):
                        continue
                    reads = int(g.get("reads", 0))
                    if variant not in streakout_map or reads > streakout_map[variant]["reads"]:
                        pileup_html = (
                            f"../demux_output/streakout/"
                            f"well_{_row['plate']}_{_row['well']}.html"
                        )
                        streakout_map[variant] = {
                            "source_plate": _row["plate"],
                            "source_well": _row["well"],
                            "reads": reads,
                            "frac": float(g.get("frac", 0)),
                            "pileup_url": pileup_html,
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
                        "empty": False,
                    })
                    n_upgraded += 1
            if n_upgraded:
                console.print(
                    f"[cyan]↑[/cyan] {n_upgraded} streakout-recoverable variant(s) "
                    f"added to pick plate (blue)"
                )

    if len(pick_list) == 0:
        console.print("[yellow]Warning:[/yellow] No hits to pick!")
        console.print("Check your demux results and target criteria.")
        raise typer.Exit(1)

    # Determine output file path
    pick_dir = pick_dir_base
    pick_dir.mkdir(parents=True, exist_ok=True)

    integra_dir = pick_dir / "Integra ASSIST Input"
    integra_dir.mkdir(exist_ok=True)

    output_dir = integra_dir
    if output is not None:
        output_dir = output.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save pick list in Integra ASSIST PLUS format (one file per target plate)
    written_files = _save_pick_list(pick_list, output_dir, volume)

    # Write README for the Integra ASSIST Input folder
    _write_integra_readme(integra_dir, written_files, volume, target_format)

    # Generate per-well pileup HTMLs for picked hits
    pileup_url_map: dict = {}
    if pileups:
        # Check if read data is available (may be missing for remote demux)
        read_df_path = demux_output / "read_df.csv"
        if not read_df_path.exists():
            _remote = project.get("workflow_steps", {}).get("demux", {}).get("remote")
            if _remote:
                console.print(
                    "[yellow]Skipping pileups:[/yellow] read_df.csv not downloaded from remote.\n"
                    f"  Run: [cyan]usortm remote fetch {project_dir}/ --read-data[/cyan] to enable pileups."
                )
                pileups = False

    if pileups:
        try:
            from usortm.demux.streakout import generate_pick_pileups

            section(console, "Pileups")
            demux_output_dir = demux_output
            n_hits = len([h for h in pick_list if not h.get("empty")])

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    f"Hit pileups ({n_hits} wells)", total=n_hits
                )

                def _on_progress(well_pos: str, success: bool):
                    # Keep the stage in the label; the well is context, not the
                    # heading, so the bar still says what it is doing at a
                    # glance once it has scrolled.
                    label = well_pos if success else f"{well_pos} [yellow](skipped)[/yellow]"
                    progress.update(
                        task_id, advance=1,
                        description=f"Hit pileups ({n_hits} wells) · {label}",
                    )

                pileup_url_map = generate_pick_pileups(
                    pick_list=pick_list,
                    demux_output_dir=str(demux_output_dir),
                    output_dir=str(pick_dir),
                    workers=workers,
                    progress_callback=_on_progress,
                )

            n_pileups = sum(len(v) for v in pileup_url_map.values())
            console.print(f"[green]\u2713[/green] {n_pileups} pileup HTMLs saved to {pick_dir / 'pileup'}")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not generate pileup HTMLs: {e}")

    # Generate pileups for mutation wells (excluded from pick but still visualized).
    # Saved to demux_output/mutation/ so the demux plate map can link to them.
    # Skip streakout candidates — they already have their own pileups.
    if pileups:
        streakout_csv = demux_output / "streakout" / "streakout_candidates.csv"
        streakout_well_keys: set = set()
        if streakout_csv.exists():
            with open(streakout_csv) as _sf:
                for _row in csv.DictReader(_sf):
                    streakout_well_keys.add(f"{_row['plate']}_{_row['well']}")

        mutation_well_data = [
            w for w in well_data
            if w.get("cons_check", "") in ("Other Error", "Error")
            and w.get("reads", 0) >= 20
            and f"{w['plate']}_{w['well']}" not in streakout_well_keys
        ]
        if mutation_well_data:
            try:
                from usortm.demux.streakout import generate_pick_pileups

                # Build a pseudo pick-list where source == target so generate_pick_pileups
                # can find and process the reads.
                mutation_list = [
                    {
                        "source_plate": w["plate"],
                        "source_well": w["well"],
                        "variant": w["variant"],
                        "reads": w["reads"],
                        "consensus_fraction": w["consensus_fraction"],
                        "cons_check": w.get("cons_check", ""),
                        "target_plate": w["plate"],
                        "target_well": w["well"],
                    }
                    for w in mutation_well_data
                ]
                n_mut = len(mutation_list)

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    task_id = progress.add_task(
                        f"Mutation pileups ({n_mut} wells)", total=n_mut
                    )

                    def _on_mut_progress(well_pos: str, success: bool):
                        label = well_pos if success else f"{well_pos} [yellow](skipped)[/yellow]"
                        progress.update(
                            task_id, advance=1,
                            description=f"Mutation pileups ({n_mut} wells) · {label}",
                        )

                    generate_pick_pileups(
                        pick_list=mutation_list,
                        demux_output_dir=str(demux_output),
                        output_dir=str(demux_output / "mutation"),
                        workers=workers,
                        progress_callback=_on_mut_progress,
                    )

                console.print(
                    f"[green]\u2713[/green] {n_mut} mutation pileup(s) saved to "
                    f"{demux_output / 'mutation' / 'pileup'}"
                )
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not generate mutation pileups: {e}")

    # Generate interactive pick plate map (Bokeh is optional)
    try:
        from usortm.demux.viz import save_pick_plate_map_html

        pick_map_path = pick_dir / "pick_plate_map.html"
        save_pick_plate_map_html(
            pick_list, str(pick_map_path),
            title="Pick Plate Map",
            target_format=target_format,
            pileup_url_map=pileup_url_map,
        )
        console.print(
            f"[green]\u2713[/green] Pick plate map saved to {pick_map_path}"
        )
    except ImportError:
        pass  # Bokeh not installed — skip
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Could not generate pick plate map: {e}")

    # Save pick list as JSON for the report to build detail tables
    import json as _json
    with open(pick_dir / "pick_list.json", "w") as _plf:
        _json.dump(pick_list, _plf, indent=2)

    # Save pick workflow state
    _all_hits = [h for h in pick_list if not h.get("empty")]
    _streakout_hits = [h for h in _all_hits if h.get("tier_override") == "Streakout"]
    pick_state = {
        "completed": True,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "total_hits": len(_all_hits),
        "unique_variants": len(set(h["variant"] for h in _all_hits if h.get("tier_override") != "Streakout")),
        "streakout_variants": len(set(h["variant"] for h in _streakout_hits)),
        "target_format": target_format,
        "compact": compact,
    }
    if tier:
        pick_state["tier"] = tier

    if round_num > 1:
        round_state["workflow_steps"]["pick"] = pick_state
        with open(round_state_file, "w") as f:
            json.dump(round_state, f, indent=2)
        project.setdefault("rounds", {}).setdefault(str(round_num), {}).setdefault(
            "workflow_steps", {}
        )["pick"] = pick_state
        with open(state_file, "w") as f:
            json.dump(project, f, indent=2)
    else:
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
    streakout_hits = [h for h in recovered if h.get("tier_override") == "Streakout"]
    regular_hits = [h for h in recovered if h.get("tier_override") != "Streakout"]
    empty_count = len(pick_list) - len(recovered)
    unique_variants = len(set(h["variant"] for h in recovered))
    summary_table.add_row("Total hits", f"{len(recovered)}")
    if regular_hits:
        summary_table.add_row("  Regular picks", f"{len(regular_hits)}")
    if streakout_hits:
        summary_table.add_row("  Streakout recoverable", f"[cyan]{len(streakout_hits)}[/cyan]")
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

    console.print("[green]\u2713[/green] Pick list generated!")
    for wf in written_files:
        console.print(f"  Hitlist: {wf}")
    console.print(f"  README: {integra_dir / 'README.txt'}")
    console.print()

    # Determine whether a multi-round merge is possible
    has_other_rounds = bool(project.get("rounds")) and any(
        str(k) != str(round_num) for k in project["rounds"]
    )

    if round_num > 1 or has_other_rounds:
        console.print("[bold]Next step:[/bold]")
        console.print(
            f"  [cyan]usortm merge {project_dir}/[/cyan]  "
            "\u2192 Merge all rounds into final pick list"
        )
        if round_num == 1:
            console.print(
                f"  [cyan]usortm report {project_dir}/[/cyan]  "
                "\u2192 Generate round 1 report only"
            )
    else:
        console.print("[bold]Next step:[/bold]")
        console.print(f"  [cyan]usortm report {project_dir}/[/cyan]  \u2192 Generate final report")
    console.print()


def _load_well_assignments(assignments_file: Path) -> list:
    """Load well assignments from demux output."""
    well_data = []

    with open(assignments_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "plate": row["plate"],
                "well": row["well"],
                "variant": row["variant"].split("|")[0],  # strip legacy |cons_check suffix
                "reads": int(row["reads"]),
                "consensus_fraction": float(row["consensus_fraction"]),
                "cons_check": row.get("cons_check", ""),
            }
            fc = row.get("flank_check", "")
            if fc:
                entry["flank_check"] = fc
            ac = row.get("assignment_confidence", "")
            if ac:
                entry["assignment_confidence"] = float(ac)
            nfp = row.get("n_flagged_positions", "")
            if nfp:
                entry["n_flagged_positions"] = int(nfp)
            well_data.append(entry)

    return well_data


def _load_library_order(
    project: dict,
    project_dir: Optional[Path] = None,
) -> tuple[Optional[dict], Optional[Path]]:
    """Load variant ordering from the library/variants CSV.

    Returns ``(order_dict, path_used)`` where *order_dict* maps variant name
    to its 0-based row index in the CSV, or ``(None, tried_path)`` when the
    file cannot be found/read.  *tried_path* is the absolute path that was
    attempted so callers can surface a useful warning.
    """
    variants_path_str = project.get("library_file") or project.get("variants_file")

    candidates: list[Path] = []
    if variants_path_str:
        candidates.append(Path(variants_path_str))
    # Fallback: look for a variants.csv next to the project state file
    if project_dir:
        candidates.append(input_file(project_dir, "variants.csv"))

    tried: Optional[Path] = candidates[0] if candidates else None
    resolved: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        return None, tried

    order = {}
    try:
        with open(resolved, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                name = (
                    row.get("Name")
                    or row.get("name")
                    or row.get("variant")
                    or row.get("variant_name")
                )
                if name:
                    order[name] = idx
    except Exception:
        return None, resolved

    return (order if order else None), resolved


class LayoutError(Exception):
    """A destination layout that cannot be used as given."""


def _load_layout(layout_file: Path) -> list:
    """Read a destination layout: which well each variant is picked into.

    Ordinarily pick decides where a hit lands, filling a plate in row or
    column order.  A layout takes that decision instead, which is what you
    want when the destination plate has already been designed -- the wells
    were chosen to put the scans in known quadrants, and a hit has to arrive
    where the design says regardless of how many other variants were
    recovered.

    Two columns are read: the destination well, and the variant that belongs
    in it.  A row naming no variant is a well the design leaves blank, and is
    kept as one.  Any other column is ignored, which matters here: a designed
    layout often carries its own ``source_plate`` and ``source_well``, meaning
    where the variant came from in the *synthesis* plates.  Those are not the
    sorted wells pick draws from, and reading them would put hits in the wrong
    place.

    Args:
        layout_file: CSV with a well column (``well`` or ``target_well``) and
            a variant column (``variant``, ``name`` or ``Name``).  An optional
            plate column (``plate`` or ``target_plate``) spreads the layout
            over more than one plate; without it everything is plate 0.

    Returns:
        One entry per destination well, in file order, as
        ``{"plate": str, "well": str, "variant": str or None}``.

    Raises:
        LayoutError: If the file is unreadable, either column is missing,
            it holds no rows, or a well is named twice on one plate.
    """
    try:
        with open(layout_file, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise LayoutError(f"could not read {layout_file}: {exc}") from exc

    if not rows:
        raise LayoutError(f"{layout_file} has no rows")

    field_names = rows[0].keys()

    def _pick_column(*names):
        for name in names:
            if name in field_names:
                return name
        return None

    well_col = _pick_column("well", "target_well", "Well")
    variant_col = _pick_column("variant", "name", "Name", "variant_name")
    plate_col = _pick_column("plate", "target_plate", "Plate")

    if well_col is None or variant_col is None:
        raise LayoutError(
            f"{layout_file} needs a well column (well/target_well) and a "
            f"variant column (variant/name); found: "
            f"{', '.join(field_names)}"
        )

    layout = []
    seen = set()
    for row in rows:
        well = (row.get(well_col) or "").strip()
        if not well:
            continue
        plate = (row.get(plate_col) or "0").strip() if plate_col else "0"
        key = (plate, well)
        if key in seen:
            raise LayoutError(
                f"{layout_file}: plate {plate} well {well} appears twice"
            )
        seen.add(key)
        variant = (row.get(variant_col) or "").strip()
        layout.append({
            "plate": plate,
            "well": well,
            "variant": variant or None,
        })

    if not layout:
        raise LayoutError(f"{layout_file} names no wells")
    return layout


def _apply_layout(pick_list: list, layout: list) -> dict:
    """Place each hit in the well the layout gives it.

    Returns counts describing what the layout and the run had to say about
    each other: how many designed wells were filled, how many stay blank
    because nothing was recovered for them, and which recovered variants the
    layout has no well for.  The last of those is the one worth acting on --
    a hit with nowhere to go is dropped, so it is reported rather than
    silently lost.
    """
    hits = {}
    for hit in pick_list:
        if hit.get("empty"):
            continue
        hits.setdefault(hit["variant"], hit)

    placed, blank, designed_blank = [], 0, 0
    for slot in layout:
        variant = slot["variant"]
        if variant is None:
            designed_blank += 1
            continue
        hit = hits.pop(variant, None)
        if hit is None:
            placed.append({
                "variant": variant,
                "source_plate": "",
                "source_well": "",
                "reads": 0,
                "consensus_fraction": 0,
                "empty": True,
                "target_plate": slot["plate"],
                "target_well": slot["well"],
            })
            blank += 1
            continue
        hit["target_plate"] = slot["plate"]
        hit["target_well"] = slot["well"]
        placed.append(hit)

    pick_list[:] = placed
    return {
        "filled": len(placed) - blank,
        "not_recovered": blank,
        "designed_blank": designed_blank,
        "unplaced": sorted(hits),
    }


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
    include_flank_errors: bool = False,
    include_cons_errors: bool = False,
    layout: Optional[list] = None,
    layout_stats: Optional[dict] = None,
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

    # Sort wells: Perfect Match first, then Silent Mutation (fewest mismatches
    # = highest consensus_fraction), then everything else.  Reads are the
    # tiebreaker within each category.
    def _well_sort_key(w):
        cons = w.get("cons_check", "")
        category = (
            0 if cons == "Perfect Match" else
            1 if cons == "Silent Mutation" else
            2
        )
        conf = w.get("assignment_confidence", 0) or 0
        return (category, -conf, -w["reads"], -w["consensus_fraction"])

    sorted_wells = sorted(well_data, key=_well_sort_key)

    # Apply tier filter
    if tier and tier in TIER_THRESHOLDS:
        thresh = TIER_THRESHOLDS[tier]
        sorted_wells = [
            w for w in sorted_wells
            if w["reads"] >= thresh["min_reads"]
            and w["consensus_fraction"] > thresh["min_consensus"]
        ]

    # Exclude wells with flanking region errors (unless overridden)
    if not include_flank_errors:
        has_any_flank_data = any(w.get("flank_check") for w in sorted_wells)
        if has_any_flank_data:
            sorted_wells = [
                w for w in sorted_wells
                if not w.get("flank_check") or w["flank_check"] == "OK"
            ]

    # Exclude wells with consensus errors (unless overridden).
    # Silent Mutations are always accepted — they encode the correct protein
    # and the demux plate map displays them as mutation-free (green).
    _ACCEPTABLE_CONS = {"Perfect Match", "Silent Mutation"}
    if not include_cons_errors:
        has_any_cons_data = any(w.get("cons_check") for w in sorted_wells)
        if has_any_cons_data:
            sorted_wells = [
                w for w in sorted_wells
                if not w.get("cons_check")
                or w["cons_check"] in _ACCEPTABLE_CONS
            ]

    # Exclude wells whose worst ORF column disagrees by more than the mixed
    # template threshold, which is a second population rather than the
    # base-caller's error.  Judged on the fraction rather than on a count of
    # flagged columns, so the criterion is the one stated here and not the one
    # in force when the run was demultiplexed.
    has_mismatch_data = any(w.get("max_mismatch_frac") is not None
                            for w in sorted_wells)
    if has_mismatch_data:
        sorted_wells = [
            w for w in sorted_wells
            if float(w.get("max_mismatch_frac") or 0.0)
            <= MIXED_TEMPLATE_THRESHOLD
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
            "cons_check": well.get("cons_check", ""),
            "assignment_confidence": well.get("assignment_confidence", 0),
        })

        seen_variants.add(variant)

    # A layout decides both order and position, so the library ordering and
    # the placeholder-packing below have nothing left to decide.
    if layout is not None:
        stats = _apply_layout(pick_list, layout)
        if layout_stats is not None:
            layout_stats.update(stats)
        return pick_list

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


def _write_integra_readme(
    integra_dir: Path,
    hitlist_files: list,
    volume: float,
    target_format: int,
):
    """Write a README.txt explaining how to use the Integra ASSIST PLUS input files."""
    readme_path = integra_dir / "README.txt"
    files_str = "\n".join(f"  • {f.name}" for f in hitlist_files)
    content = f"""\
Integra ASSIST PLUS — Hit-Picking Input
========================================

Files (one per target plate):
{files_str}

These semicolon-delimited CSVs are formatted for direct import into the
Integra ASSIST PLUS liquid handling robot software.

Columns
-------
  SampleID       Variant name
  SourcePlateID  Source plate number (from demultiplexing)
  SourceWell     Source well position (e.g. A1)
  TargetPlateID  Destination plate number
  TargetWell     Destination well position
  TransferVolume Transfer volume in µL

Settings used
-------------
  Transfer volume : {volume:.1f} µL
  Target format   : {target_format}-well plate

Notes
-----
  • Load source plates in the order indicated by SourcePlateID.
  • Verify tip type and labware definitions match your plate format before
    running the protocol.
"""
    readme_path.write_text(content)


def _save_pick_list(pick_list: list, output_dir: Path, volume: float):
    """Save pick list in Integra ASSIST PLUS format, one file per target plate."""
    from collections import defaultdict
    plates: dict[str, list] = defaultdict(list)
    for hit in pick_list:
        if hit.get("empty"):
            continue
        plates[str(hit["target_plate"])].append(hit)

    written_files = []
    for plate_id in sorted(plates):
        fname = f"hitlist_plate_{plate_id}.csv"
        out = output_dir / fname
        with open(out, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "SampleID", "SourcePlateID", "SourceWell",
                "TargetPlateID", "TargetWell", "TransferVolume",
            ])
            for hit in plates[plate_id]:
                writer.writerow([
                    hit["variant"].replace(";", "."),
                    hit["source_plate"],
                    hit["source_well"],
                    hit["target_plate"],
                    hit["target_well"],
                    f"{volume:.1f}",
                ])
        written_files.append(out)
    return written_files
