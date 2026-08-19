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
    relinked = _relink_plate_map(demux_output, out_dir, selected, min_reads)

    console.print(
        f"\n[green]✓[/green] {rendered['n']:,} pileup(s) written to {out_dir}"
        + (f" ([yellow]{rendered['skipped']:,} skipped[/yellow])"
           if rendered["skipped"] else "")
    )
    console.print(f"[green]✓[/green] Index: {index_path}")
    try:
        from usortm.cli.project_index import write_index

        write_index(project_dir, round_num)
    except Exception:
        pass
    if relinked:
        console.print(
            f"[green]✓[/green] Plate map wells now link to their pileups "
            f"({relinked}) — rebuild the report with "
            f"[cyan]usortm report {project_dir}[/cyan] to pick them up there"
        )


def _flag_well_keys(demux_output: Path) -> tuple:
    """Read the streak-out, mutation and silent-mutation well sets.

    These drive the plate map's corner tabs, so a rebuilt map has to carry
    them or the flags would silently disappear.
    """
    streakout: set = set()
    mutation: set = set()
    silent: set = set()

    so_csv = demux_output / "streakout" / "streakout_candidates.csv"
    if so_csv.exists():
        with open(so_csv) as fh:
            for row in csv.DictReader(fh):
                streakout.add(f"{row['plate']}_{row['well']}")

    wa_csv = demux_output / "well_assignments.csv"
    if wa_csv.exists():
        with open(wa_csv) as fh:
            for row in csv.DictReader(fh):
                key = f"{row['plate']}_{row['well']}"
                cons = row.get("cons_check", "")
                reads = int(row.get("reads", 0) or 0)
                if reads >= 20 and key not in streakout:
                    if cons in ("Other Error", "Error"):
                        mutation.add(key)
                    elif cons == "Silent Mutation":
                        silent.add(key)
                elif cons == "Silent Mutation":
                    silent.add(key)
    return streakout, mutation, silent


def _relink_plate_map(demux_output: Path, out_dir: Path, wells: list,
                      min_reads: int) -> Optional[str]:
    """Rebuild ``plate_map.html`` so every rendered well links to its pileup.

    The map is regenerated rather than patched because the links are baked
    into the Bokeh data at build time.  Returns None when the pileups live
    outside the demux directory, since a relative link could not reach them.

    Args:
        demux_output: Demux output directory holding ``plate_map.html``.
        out_dir: Directory the pileups were written to.
        wells: Selected well rows.
        min_reads: Depth cutoff, forwarded to the plate map's own tiering.

    Returns:
        Description of what was linked, or None if nothing was.
    """
    plate_map_path = demux_output / "plate_map.html"
    read_df_path = demux_output / "read_df.csv"
    if not plate_map_path.exists() or not read_df_path.exists():
        return None

    try:
        rel_root = out_dir.resolve().relative_to(demux_output.resolve())
    except ValueError:
        # Pileups written outside the demux directory: a relative URL from the
        # plate map cannot reach them, so leave the existing map alone.
        return None

    pileup_dir = out_dir / "pileup"
    url_map = {}
    for row in wells:
        key = f"{row['plate']}_{row['well']}"
        fname = f"well_{key}.html"
        if (pileup_dir / fname).exists():
            url_map[key] = f"{rel_root.as_posix()}/pileup/{fname}"
    if not url_map:
        return None

    try:
        from usortm.demux.viz import load_plate_map_reads, save_plate_map_html

        read_df = load_plate_map_reads(read_df_path)
        if read_df.empty:
            return None
        streakout, mutation, silent = _flag_well_keys(demux_output)
        save_plate_map_html(
            read_df, str(plate_map_path),
            title="Demux Plate Map",
            streakout_wells=streakout,
            mutation_wells=mutation,
            silent_mutation_wells=silent,
            pileup_url_map=url_map,
            min_reads=min_reads,
        )
    except ImportError:
        return None      # Bokeh is optional
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] could not relink the plate map: {exc}"
        )
        return None

    return f"{len(url_map):,} well(s)"


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
