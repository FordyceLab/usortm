"""Generate synthesis ordering files for dropout variants."""
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

# 96-well row/column layout for eBlocks plate format
_ROWS = list("ABCDEFGH")
_COLS = list(range(1, 13))
_WELLS_96 = [f"{r}{c}" for r in _ROWS for c in _COLS]  # A1, A2, ..., H12


def reorder(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to completed uSort-M project directory (pick must be done).",
        exists=True,
    ),
    format: str = typer.Option(
        ...,
        "--format", "-f",
        help=(
            "Ordering format: "
            "'eblocks' (IDT eBlocks, 96-well plate), "
            "'twist' (Twist Gene Fragments), "
            "'twist_oligo' (Twist Oligonucleotide Pools), "
            "'opools' (IDT oPools)."
        ),
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output CSV file path. Defaults to <project_dir>/reorder_<format>.csv.",
    ),
    pool_name: str = typer.Option(
        "dropout_pool",
        "--pool-name", "-p",
        help="Pool name for IDT oPools format (ignored for other formats).",
    ),
):
    """
    Generate a synthesis ordering file for dropout variants.

    Identifies variants from the library that were [bold]not recovered[/bold] in the
    most recent pick run, then exports them in the requested vendor format.

    [bold]Supported formats:[/bold]

        [cyan]eblocks[/cyan]      IDT eBlocks — 96-well plate upload (Well Position, Name, Sequence)
        [cyan]twist[/cyan]        Twist Gene Fragments — (Sequence name, Sequence)
        [cyan]twist_oligo[/cyan]  Twist Oligonucleotide Pools — (name, sequence)
        [cyan]opools[/cyan]       IDT oPools — (Pool name, Sequence)

    [bold]Example:[/bold]

        usortm reorder my_project/ --format eblocks
        usortm reorder my_project/ --format opools --pool-name round2_dropouts
    """
    fmt = format.lower().strip()
    valid_formats = ("eblocks", "twist", "twist_oligo", "opools")
    if fmt not in valid_formats:
        console.print(
            f"[red]Error:[/red] Unknown format '{format}'. "
            f"Choose from: {', '.join(valid_formats)}"
        )
        raise typer.Exit(1)

    # Load project state
    state_file = project_dir / PROJECT_STATE_FILE
    if not state_file.exists():
        console.print(f"[red]Error:[/red] No project found at {project_dir}")
        raise typer.Exit(1)
    with open(state_file) as f:
        project = json.load(f)

    pick_state = project.get("workflow_steps", {}).get("pick", {})
    if not pick_state.get("completed"):
        console.print("[red]Error:[/red] Pick step not completed. Run [cyan]usortm pick[/cyan] first.")
        raise typer.Exit(1)

    # Load variants (name + sequence)
    variants_path = _find_variants_file(project, project_dir)
    if variants_path is None or not variants_path.exists():
        console.print("[red]Error:[/red] Could not find variants file in project.")
        raise typer.Exit(1)

    variants = _load_variants(variants_path)
    if not variants:
        console.print("[red]Error:[/red] No variants found in variants file.")
        raise typer.Exit(1)

    # Load recovered variants from hitlist
    hitlist_path = project_dir / "hitlist.csv"
    if not hitlist_path.exists():
        console.print(f"[red]Error:[/red] hitlist.csv not found in {project_dir}")
        raise typer.Exit(1)

    recovered = _load_recovered(hitlist_path)

    # Identify dropouts
    dropouts = [v for v in variants if _normalize(v["name"]) not in recovered]

    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Reorder Generator",
        border_style=BORDER_STYLE,
    ))
    console.print()
    console.print(f"[green]✓[/green] Library size:     [cyan]{len(variants)}[/cyan] variants")
    console.print(f"[green]✓[/green] Recovered:        [cyan]{len(variants) - len(dropouts)}[/cyan] variants")
    console.print(f"[yellow]→[/yellow] Dropouts to order: [cyan]{len(dropouts)}[/cyan] variants")
    console.print()

    if not dropouts:
        console.print("[green]✓[/green] All variants recovered — nothing to order!")
        raise typer.Exit(0)

    # Write output
    if output is None:
        output = project_dir / f"reorder_{fmt}.csv"

    if fmt == "eblocks":
        n_plates = _write_idt_eblocks(dropouts, output)
        console.print(f"[green]✓[/green] IDT eBlocks CSV written to [cyan]{output}[/cyan]")
        console.print(f"   {len(dropouts)} sequences across {n_plates} plate(s)")
    elif fmt == "twist":
        _write_twist_gene_fragments(dropouts, output)
        console.print(f"[green]✓[/green] Twist Gene Fragments CSV written to [cyan]{output}[/cyan]")
        console.print(f"   {len(dropouts)} sequences")
    elif fmt == "twist_oligo":
        _write_twist_oligo_pools(dropouts, output)
        console.print(f"[green]✓[/green] Twist Oligonucleotide Pools CSV written to [cyan]{output}[/cyan]")
        console.print(f"   {len(dropouts)} sequences")
    elif fmt == "opools":
        _write_idt_opools(dropouts, output, pool_name)
        console.print(f"[green]✓[/green] IDT oPools CSV written to [cyan]{output}[/cyan]")
        console.print(f"   {len(dropouts)} sequences in pool '{pool_name}'")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    next_step_map = {
        "eblocks":     "Upload [cyan]reorder_eblocks.csv[/cyan] to IDT eBlocks plate order",
        "twist":       "Upload [cyan]reorder_twist.csv[/cyan] to Twist Gene Fragments order",
        "twist_oligo": "Upload [cyan]reorder_twist_oligo.csv[/cyan] to Twist Oligonucleotide Pools order",
        "opools":      "Upload [cyan]reorder_opools.csv[/cyan] to IDT oPools order",
    }
    console.print(f"  1. {next_step_map[fmt]}")
    console.print("  2. Clone received fragments and barcode")
    round_n = project.get("round", 1)
    console.print(
        f"  3. Sequence and run: [cyan]usortm plan variants.csv --round {round_n + 1}[/cyan]"
    )
    console.print()


def _find_variants_file(project: dict, project_dir: Path) -> Optional[Path]:
    """Find the variants file from project JSON or fallback locations."""
    for key in ("variants_file", "library_file"):
        path_str = project.get(key)
        if path_str:
            p = Path(path_str)
            if p.exists():
                return p
    # Fallback: look for variants.csv in project dir
    fallback = project_dir / "variants.csv"
    if fallback.exists():
        return fallback
    return None


def _load_variants(variants_path: Path) -> list[dict]:
    """Load variants from CSV, returning list of {name, sequence} dicts."""
    with open(variants_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        name_col = _find_col(headers, ("Name", "name", "variant", "id"))
        seq_col = _find_col(headers, ("Sequence", "sequence", "dna_sequence", "dna", "seq"))

        if name_col is None:
            raise typer.BadParameter(
                f"No name column found in {variants_path}. Expected one of: Name, name, variant, id"
            )
        if seq_col is None:
            raise typer.BadParameter(
                f"No sequence column found in {variants_path}. Expected one of: Sequence, sequence, dna_sequence, dna, seq"
            )

        return [{"name": row[name_col], "sequence": row[seq_col]} for row in reader if row.get(name_col)]


def _load_recovered(hitlist_path: Path) -> set[str]:
    """Load recovered variant names from hitlist CSV (semicolon-delimited)."""
    recovered = set()
    with open(hitlist_path, newline="") as f:
        # hitlist is semicolon-delimited (Integra format)
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sample_id = row.get("SampleID", "").strip()
            try:
                vol = float(row.get("TransferVolume", 0))
            except ValueError:
                vol = 0.0
            if sample_id and vol > 0:
                recovered.add(_normalize(sample_id))
    return recovered


def _normalize(name: str) -> str:
    """Strip cons_check / Perfect Match suffixes before comparison."""
    for suffix in ("|cons_check", "|Perfect Match"):
        name = name.split(suffix)[0]
    return name.strip()


def _find_col(headers: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first header that matches any candidate (case-insensitive)."""
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _write_idt_eblocks(dropouts: list[dict], output: Path) -> int:
    """Write IDT eBlocks 96-well plate upload CSV. Returns number of plates."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Well Position", "Name", "Sequence"])

        plate = 1
        well_idx = 0
        for variant in dropouts:
            if well_idx >= len(_WELLS_96):
                # Start new plate — blank separator row then plate header
                writer.writerow([])
                writer.writerow([f"# Plate {plate + 1}"])
                writer.writerow(["Well Position", "Name", "Sequence"])
                plate += 1
                well_idx = 0
            writer.writerow([_WELLS_96[well_idx], variant["name"], variant["sequence"]])
            well_idx += 1

    return plate


def _write_twist_gene_fragments(dropouts: list[dict], output: Path) -> None:
    """Write Twist Gene Fragments two-column CSV (Sequence name, Sequence)."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence name", "Sequence"])
        for variant in dropouts:
            writer.writerow([variant["name"], variant["sequence"]])


def _write_twist_oligo_pools(dropouts: list[dict], output: Path) -> None:
    """Write Twist Oligonucleotide Pools two-column CSV (name, sequence)."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sequence"])
        for variant in dropouts:
            writer.writerow([variant["name"], variant["sequence"]])


def _write_idt_opools(dropouts: list[dict], output: Path, pool_name: str) -> None:
    """Write IDT oPools two-column CSV (Pool name, Sequence)."""
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pool name", "Sequence"])
        for variant in dropouts:
            writer.writerow([pool_name, variant["sequence"]])
