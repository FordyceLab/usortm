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

_BSAI_FWD = "GGTCTC"   # BsaI recognition site (forward)
_BSAI_REV = "GAGACC"   # BsaI recognition site (reverse complement)
_BSAI_FLANK = 7        # bp to keep outside each recognition site

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
    library: Optional[Path] = typer.Option(
        None,
        "--library", "-l",
        help=(
            "Path to the original library CSV containing full sequences (e.g. with golden gate "
            "adapters or length-normalising padding). The name column is auto-detected; the "
            "sequence column used is the one with the longest average length — the chosen column "
            "name is always printed. Dropout sequences are replaced with those from the library."
        ),
    ),
    trim_bsai: bool = typer.Option(
        False,
        "--trim-bsai",
        help=(
            f"Trim each sequence to {_BSAI_FLANK} bp outside its flanking BsaI sites "
            f"(GGTCTC / GAGACC). Requires sequences to contain exactly one forward and one "
            "reverse BsaI site. Sequences where sites cannot be found are kept as-is with a warning."
        ),
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
        usortm reorder my_project/ --format eblocks --library full_library.csv
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

    # Optionally replace sequences with full sequences from a library CSV
    if library is not None:
        if not library.exists():
            console.print(f"[red]Error:[/red] Library file not found: {library}")
            raise typer.Exit(1)
        library_seqs = _load_library_sequences(library)
        not_found = []
        for v in dropouts:
            full_seq = library_seqs.get(v["name"])
            if full_seq:
                v["sequence"] = full_seq
            else:
                not_found.append(v["name"])
        if not_found:
            console.print(
                f"[yellow]Warning:[/yellow] {len(not_found)} dropout(s) not found in library "
                f"— original sequences kept:"
            )
            for name in not_found:
                console.print(f"  {name}")

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

    # Optionally trim sequences to 5 bp outside flanking BsaI sites
    if trim_bsai:
        untrimmed = []
        for v in dropouts:
            trimmed = _trim_to_bsai(v["sequence"])
            if trimmed is None:
                untrimmed.append(v["name"])
            else:
                v["sequence"] = trimmed
        if untrimmed:
            console.print(
                f"[yellow]Warning:[/yellow] {len(untrimmed)} sequence(s) missing a BsaI site — kept untrimmed:"
            )
            for name in untrimmed:
                console.print(f"  {name}")
        else:
            console.print(
                f"[green]✓[/green] Sequences trimmed to {_BSAI_FLANK} bp outside BsaI sites"
            )

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

    total_bp = sum(len(v["sequence"]) for v in dropouts)
    cost = total_bp * 0.07
    console.print(f"[yellow]→[/yellow] Estimated cost: [cyan]{total_bp:,} bp[/cyan] × $0.07 = [cyan]${cost:,.2f}[/cyan]")

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


def _load_library_sequences(library_path: Path) -> dict[str, str]:
    """Load name→sequence mapping from a library CSV.

    The name column is detected using the same flexible logic as variant loading,
    falling back to the first column (handles unnamed-index CSVs).
    The sequence column is whichever non-name column has the longest average value
    length — the chosen column name is always printed.
    """
    with open(library_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise typer.BadParameter(f"Library file {library_path} is empty.")

    # Find name column — fall back to first column for unnamed-index CSVs
    name_col = _find_col(headers, ("Name", "name", "variant", "id"))
    if name_col is None:
        name_col = headers[0] if headers else None
    if name_col is None:
        raise typer.BadParameter(f"Could not find a name column in {library_path}.")

    # Pick the longest column whose values are pure DNA (only A/C/G/T)
    _DNA_CHARS = frozenset("ACGTacgt")

    def _is_dna_col(col: str) -> bool:
        vals = [row.get(col) or "" for row in rows if row.get(col)]
        return bool(vals) and all(set(v) <= _DNA_CHARS for v in vals)

    def _avg_len(col: str) -> float:
        vals = [row.get(col) or "" for row in rows]
        return sum(len(v) for v in vals) / len(vals)

    candidate_cols = [h for h in headers if h != name_col and h and _is_dna_col(h)]
    if not candidate_cols:
        raise typer.BadParameter(
            f"No DNA-only columns found in {library_path}. "
            "Expected at least one column whose values contain only A, C, G, T."
        )

    seq_col = max(candidate_cols, key=_avg_len)
    console.print(
        f"[yellow]→[/yellow] Library sequence column auto-detected: "
        f"[cyan]{seq_col!r}[/cyan] (longest DNA column, avg {_avg_len(seq_col):.0f} bp)"
    )

    return {row[name_col]: row[seq_col] for row in rows if row.get(name_col)}


def _trim_to_bsai(seq: str) -> Optional[str]:
    """Trim sequence to _BSAI_FLANK bp outside flanking BsaI sites.

    Finds the first forward BsaI site (GGTCTC) and the last reverse BsaI site
    (GAGACC), then keeps _BSAI_FLANK bp of flanking sequence on each side.
    Returns None if either site is not found.
    """
    upper = seq.upper()
    fwd_pos = upper.find(_BSAI_FWD)
    rev_pos = upper.rfind(_BSAI_REV)

    if fwd_pos == -1 or rev_pos == -1:
        return None

    start = max(0, fwd_pos - _BSAI_FLANK)
    end = rev_pos + len(_BSAI_REV) + _BSAI_FLANK

    return seq[start:end]


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
