"""Judge a sequenced pick plate against the layout the merge gave it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from usortm import provenance as _provenance
from usortm.cli.theme import BORDER_STYLE, get_console
from usortm import pickplate as _pickplate
from usortm.pickplate import (CONFIRMED, EMPTY, LAYOUT_FILE, ROUND_KIND,
                              VERDICT_FILE, WRONG, designed_names, judge,
                              load_well_rows, read_expected_layout,
                              verdict_rows, write_verdicts)

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"
ROUND_STATE_FILE = "usortm_round.json"


def verify(
    project_dir: Path = typer.Argument(..., help="Path to uSort-M project directory.",
                                       exists=True),
    pick_plate: Optional[str] = typer.Option(
        None, "--pick-plate",
        help="The pick-plate run to judge, by its sequencing run ID (e.g. "
             "G3Y8KW). Default: the newest run in 7_pick_plate/."),
    round_num: Optional[int] = typer.Option(
        None, "--round", "-r", min=2,
        help="Judge a pick plate recorded as a round, the layout used before "
             "7_pick_plate/. Cannot be combined with --pick-plate."),
    min_reads: int = typer.Option(20, "--min-reads",
                                  help="Reads a well needs before its call is used; "
                                       "below this it is reported empty."),
    render_pileups: bool = typer.Option(True, "--pileups/--no-pileups",
                                        help="Render a pileup and summary page for every "
                                             "intended well that returned reads, so the "
                                             "report's plate links to them."),
    workers: int = typer.Option(6, "--workers", "-w",
                                help="Parallel workers for pileup rendering."),
):
    """
    Judge each well of a sequenced pick plate against the variant the merge
    placed there.

    Every well of the destination plate has an intended variant, so each is
    reported as [bold]confirmed[/bold] (holds it, read cleanly by the same
    test the pick used), [bold]wrong[/bold] (holds something else, or holds
    it unreadably), or [bold]empty[/bold] (too few reads).  The verdicts are
    written to 7_pick_plate/<run>/results/ and drawn into summary.html by
    `usortm report`.
    """
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] Not a valid uSort-M project (missing {PROJECT_STATE_FILE})")
        raise typer.Exit(1)
    with open(state_file) as fh:
        project = json.load(fh)

    if pick_plate is not None and round_num is not None:
        console.print("[red]Error:[/red] --pick-plate and --round cannot be combined.")
        raise typer.Exit(1)
    target = _resolve_target(project_dir, project, pick_plate, round_num)
    layout_file, wells_file = target["layout"], target["wells"]
    for path, what in ((layout_file, "expected layout"), (wells_file, "demux results")):
        if not path.exists():
            console.print(f"[red]Error:[/red] {target['label']} has no {what} at {path}")
            raise typer.Exit(1)

    console.print()
    console.print(Panel.fit("[brand]uSort-M[/brand] Pick Plate Verification",
                            border_style=BORDER_STYLE))
    console.print()

    layout = read_expected_layout(layout_file)
    rows = load_well_rows(wells_file)
    designed = designed_names(target["demux"])

    # The same disagreement limit the plate was built to, so a well confirmed
    # here is one the pick would have taken.
    from usortm.report.plates import (disagreement_limit_from_project,
                                      set_applied_disagreement_limit)
    limit = disagreement_limit_from_project(project)
    set_applied_disagreement_limit(limit)
    console.print(f"[green]✓[/green] {len(layout)} intended wells; {len(rows)} wells with reads; "
                  f"worst-column limit "
                  + (f"{limit:.0%}" if limit is not None else "mixed-template threshold"))

    verdicts, summary = judge(rows, layout, designed, min_reads=min_reads)
    table_rows = verdict_rows(verdicts, rows, layout)
    out_dir = target["out"]
    out_csv = write_verdicts(table_rows, out_dir / VERDICT_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Beside demux_summary.json in a run's results/, so named for its step.
    summary_name = "verify_summary.json" if target["run"] else "summary.json"
    with open(out_dir / summary_name, "w") as fh:
        json.dump({"n_wells": summary["n_wells"], "wells": summary["wells"],
                   "confirmed": sorted(summary["confirmed"]),
                   "not_confirmed": sorted(summary["not_confirmed"]),
                   "min_reads": min_reads, "max_disagreement": limit}, fh, indent=2)

    n = summary["n_wells"] or 1
    t = Table(title="Pick plate, as sequenced", box=box.ROUNDED, show_header=True,
              header_style="bold cyan")
    t.add_column("Outcome", style="muted")
    t.add_column("Wells", justify="right")
    t.add_column("%", justify="right")
    for status, label in ((CONFIRMED, "Held the variant placed there"),
                          (WRONG, "Held something else / not readable"),
                          (EMPTY, f"Fewer than {min_reads} reads")):
        c = summary["wells"].get(status, 0)
        t.add_row(label, str(c), f"{100 * c / n:.1f}%")
    console.print(t)

    failed = [r for r in table_rows if r["status"] != CONFIRMED]
    if failed:
        console.print(f"\n[yellow]{len(failed)} well(s) did not hold their variant:[/yellow]")
        for r in failed[:40]:
            src = f"{r['source_plate']} {r['source_well']}"
            if r.get("bench_well"):
                src += f" (colony plate {r['bench_plate']} {r['bench_well']})"
            console.print(f"  {r['well']:>4}  {r['expected']:<7} read as {r['observed'] or 'nothing':<10} "
                          f"{r['reads']:>5} reads  {r['reason']:<16} from {src}")
        if len(failed) > 40:
            console.print(f"  ... {len(failed) - 40} more in {out_csv}")
    console.print(f"\n[green]✓[/green] Verdicts: {out_csv}")

    # The plate in the report links each well to its reads.  A pick-plate
    # round has no pick step to render them, so they are rendered here, for
    # every intended well that returned anything, whatever its depth: the
    # wells worth looking at are exactly the shallow and the wrong ones.
    if render_pileups:
        with_reads = {(int(r["plate"]), str(r["well"]).upper())
                      for r in rows if int(r.get("reads") or 0) > 0}
        wanted = [f"1{lay['well'].upper()}" for lay in layout
                  if (1, lay["well"].upper()) in with_reads]
        if wanted:
            from usortm.cli.pileups import pileups as _pileups
            console.print()
            try:
                _pileups(project_dir=project_dir, min_reads=1, plate=None,
                         well=",".join(wanted), workers=workers, output=None,
                         round_num=target["round"] or 1,
                         pick_plate=target["run"])
            except typer.Exit:
                pass

    # Record the step where the round records its others.
    step = {"completed": True, "command": _provenance.current_command(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "n_wells": summary["n_wells"], "wells": summary["wells"],
            "min_reads": min_reads, "max_disagreement": limit}
    target["record"](step)
    with open(state_file, "w") as fh:
        json.dump(project, fh, indent=2)
    _provenance.refresh_commands(project, project_dir)
    console.print("[bold]Next step:[/bold] usortm report "
                  f"{project_dir}/  → the plate, drawn by verdict, in summary.html")


def _resolve_target(project_dir: Path, project: dict, run_name, round_num) -> dict:
    """Where the plate to judge lives, and how to record its verify step.

    A pick-plate run under 7_pick_plate/ is the current form.  A pick plate
    recorded as a round (``kind = "pick_plate"``) is the form used before
    it, still read so a project made then can be judged until it is moved.
    """
    if round_num is None:
        name = run_name or _pickplate.newest_run(project)
        if name is None:
            console.print("[red]Error:[/red] No pick-plate run to judge. Demultiplex one "
                          f"with: usortm demux {project_dir}/ --pick-plate <run ID> ...")
            raise typer.Exit(1)
        if name not in (project.get(_pickplate.STATE_KEY) or {}):
            console.print(f"[red]Error:[/red] No pick-plate run {name!r} in this "
                          f"project. Recorded: "
                          f"{', '.join(n for n, _ in _pickplate.list_runs(project)) or 'none'}")
            raise typer.Exit(1)
        run = _pickplate.run_paths(project_dir, name)

        def record(step):
            project[_pickplate.STATE_KEY][name].setdefault(
                "workflow_steps", {})["verify"] = step

        return {"label": f"Pick plate {name}", "layout": run.layout,
                "wells": run.wells_csv, "demux": run.demux, "out": run.results,
                "run": name, "round": None, "record": record}

    round_dir = project_dir / "rounds" / str(round_num)
    round_state_file = round_dir / ROUND_STATE_FILE
    if not round_state_file.exists():
        console.print(f"[red]Error:[/red] Round {round_num} has not been planned.")
        raise typer.Exit(1)
    with open(round_state_file) as fh:
        round_state = json.load(fh)
    if round_state.get("kind") != ROUND_KIND:
        console.print(f"[red]Error:[/red] Round {round_num} is not a pick plate. "
                      f"A pick plate is judged with --pick-plate <run ID>.")
        raise typer.Exit(1)

    def record(step):
        round_state.setdefault("workflow_steps", {})["verify"] = step
        with open(round_state_file, "w") as fh:
            json.dump(round_state, fh, indent=2)
        project.setdefault("rounds", {}).setdefault(str(round_num), {}).setdefault(
            "workflow_steps", {})["verify"] = step

    return {"label": f"Round {round_num}", "layout": round_dir / LAYOUT_FILE,
            "wells": round_dir / "demux_output" / "well_assignments.csv",
            "demux": round_dir / "demux_output", "out": round_dir / "verify",
            "run": None, "round": round_num, "record": record}
