"""Generate read pileups for every well in a demux run.

The pipeline already renders pileups for the wells it flags — streak-out
candidates during demux, and picked or mutated wells during ``usortm pick``.
This command covers the rest, so any well can be inspected rather than only
the ones something went looking for.
"""

from typing import Optional
from pathlib import Path
import csv
import html as html_mod

import typer
from rich.progress import (
    BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)

from usortm.cli.theme import get_console

console = get_console()

PROJECT_STATE_FILE = "usortm_project.json"


def pileups(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to uSort-M project directory.",
        exists=True,
    ),
    min_reads: int = typer.Option(
        20,
        "--min-reads",
        help=(
            "Skip wells with fewer reads than this. Shallow wells make "
            "uninformative pileups and dominate the file count."
        ),
    ),
    plate: Optional[str] = typer.Option(
        None,
        "--plate",
        help="Only this plate, or a comma-separated list (e.g. '1,3'). Default: all.",
    ),
    workers: int = typer.Option(
        6,
        "--workers", "-w",
        help="Parallel workers.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory. Defaults to <project>/demux_output/pileups.",
    ),
    round_num: int = typer.Option(
        1,
        "--round",
        help="Sequencing round.",
        min=1,
    ),
):
    """
    Generate read pileups for every well in a [#4096E3]uSort-M[/#4096E3] run.

    Renders one interactive pileup per well, plus an index page grouping them
    by plate.

    [bold]Example:[/bold]

        usortm pileups my_project/ --min-reads 50
    """
    if round_num > 1:
        demux_output = project_dir / "rounds" / str(round_num) / "demux_output"
    else:
        demux_output = project_dir / "demux_output"

    assignments = demux_output / "well_assignments.csv"
    if not assignments.exists():
        console.print(
            f"[red]Error:[/red] Could not find {assignments}\n"
            "Run [cyan]usortm demux[/cyan] first."
        )
        raise typer.Exit(1)

    wanted_plates = None
    if plate is not None:
        wanted_plates = {p.strip() for p in plate.split(",") if p.strip()}

    with open(assignments) as fh:
        rows = list(csv.DictReader(fh))

    selected, skipped_shallow, skipped_plate = [], 0, 0
    for row in rows:
        if wanted_plates is not None and str(row["plate"]) not in wanted_plates:
            skipped_plate += 1
            continue
        if int(row.get("reads", 0) or 0) < min_reads:
            skipped_shallow += 1
            continue
        selected.append(row)

    if not selected:
        console.print(
            f"[yellow]No wells to render.[/yellow] {len(rows)} well(s) in the run; "
            f"{skipped_shallow} below --min-reads {min_reads}"
            + (f", {skipped_plate} outside --plate {plate}" if wanted_plates else "")
            + "."
        )
        raise typer.Exit(0)

    out_dir = output if output is not None else demux_output / "pileups"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Rendering {len(selected):,} pileup(s) of {len(rows):,} well(s) "
        f"(skipping {skipped_shallow:,} below {min_reads} reads)"
    )

    pick_list = [
        {
            "source_plate": r["plate"], "source_well": r["well"],
            "variant": r["variant"], "reads": int(r["reads"]),
            "consensus_fraction": float(r.get("consensus_fraction", 0) or 0),
            "cons_check": r.get("cons_check", ""),
            # generate_pick_pileups keys output on the source well; pointing
            # target at the same well keeps one file per well.
            "target_plate": r["plate"], "target_well": r["well"],
        }
        for r in selected
    ]

    from usortm.demux.streakout import generate_pick_pileups

    rendered = {"n": 0, "skipped": 0}
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Rendering pileups...", total=len(pick_list))

        def _on_progress(well_pos: str, success: bool):
            if success:
                rendered["n"] += 1
            else:
                rendered["skipped"] += 1
            label = well_pos if success else f"{well_pos} [yellow](skipped)[/yellow]"
            progress.update(task_id, advance=1, description=f"Pileup: {label}")

        generate_pick_pileups(
            pick_list=pick_list,
            demux_output_dir=str(demux_output),
            output_dir=str(out_dir),
            workers=workers,
            progress_callback=_on_progress,
        )

    index_path = _write_pileup_index(out_dir, selected, min_reads)

    console.print(
        f"\n[green]✓[/green] {rendered['n']:,} pileup(s) written to {out_dir}"
        + (f" ([yellow]{rendered['skipped']:,} skipped[/yellow])"
           if rendered["skipped"] else "")
    )
    console.print(f"[green]✓[/green] Index: {index_path}")


def _write_pileup_index(out_dir: Path, wells: list, min_reads: int) -> Path:
    """Write an index page linking every rendered pileup, grouped by plate.

    Thousands of loose HTML files are not navigable on their own.

    Args:
        out_dir: Directory the pileups were written to.
        wells: Selected well rows from well_assignments.csv.
        min_reads: Depth cutoff used, shown in the header.

    Returns:
        Path to the written index.
    """
    pileup_dir = out_dir / "pileup"
    by_plate: dict = {}
    for row in wells:
        fname = f"well_{row['plate']}_{row['well']}.html"
        if not (pileup_dir / fname).exists():
            continue
        by_plate.setdefault(str(row["plate"]), []).append((row, fname))

    def _well_key(item):
        well = item[0]["well"]
        return (well[0], int(well[1:] or 0))

    parts = [
        "<meta charset='utf-8'><title>uSort-M pileups</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,sans-serif;margin:2rem;"
        "background:#fafafa;color:#111}",
        "h1{font-size:1.3rem}h2{font-size:1rem;margin-top:2rem}",
        "table{border-collapse:collapse;width:100%;max-width:900px}",
        "th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #ddd}",
        "td.num{text-align:right;font-variant-numeric:tabular-nums}",
        "a{color:#2563eb;text-decoration:none}a:hover{text-decoration:underline}",
        "@media (prefers-color-scheme:dark){body{background:#16213e;color:#e0e0e0}"
        "th,td{border-color:#334}a{color:#7aa7ff}}",
        "</style>",
        "<h1>Read pileups</h1>",
        f"<p>{sum(len(v) for v in by_plate.values()):,} well(s) with at least "
        f"{min_reads} reads.</p>",
    ]

    for plate in sorted(by_plate, key=lambda p: int(p) if p.isdigit() else 0):
        entries = sorted(by_plate[plate], key=_well_key)
        parts.append(f"<h2>Plate {html_mod.escape(plate)} "
                     f"<span style='font-weight:400'>({len(entries)} wells)</span></h2>")
        parts.append("<table><tr><th>Well</th><th>Variant</th>"
                     "<th class='num'>Reads</th><th class='num'>Consensus</th></tr>")
        for row, fname in entries:
            frac = float(row.get("consensus_fraction", 0) or 0)
            parts.append(
                f"<tr><td><a href='pileup/{html_mod.escape(fname)}'>"
                f"{html_mod.escape(row['well'])}</a></td>"
                f"<td>{html_mod.escape(row['variant'])}</td>"
                f"<td class='num'>{int(row['reads']):,}</td>"
                f"<td class='num'>{frac:.0%}</td></tr>"
            )
        parts.append("</table>")

    index_path = out_dir / "index.html"
    index_path.write_text("\n".join(parts))
    return index_path
