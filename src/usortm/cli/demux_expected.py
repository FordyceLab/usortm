"""``usortm demux --expected``: check a barcoded plate against what it should hold.

Outside a uSort-M sort there is no project, no library to recover and nothing
to pick; there is a plate of known constructs and the question of whether
each well holds its own.  This runs the same LevSeq pipeline as a project
demux, with the expected plate as the reference, and then gives every well a
verdict against its expectation (:mod:`usortm.demux.verify`).

The construct is given one of three ways, in order of preference:

``--vector``
    The parent vector as sequenced.  The read layout -- amplicon, variable
    region, primer tails -- is worked out from it, the expected sequences and
    the reads (:mod:`usortm.demux.vector_layout`), and written out as a read
    template to inspect or correct.
``--read-template``
    A read drawn by hand with its three spans masked.
``--vector-fasta``
    The vector with only the variable region masked, and barcode masks from
    ``--mask-config`` or the defaults.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from usortm.cli.theme import BORDER_STYLE, get_console, section
from usortm.demux.verify import DEFAULT_MIN_READS  # noqa: F401  (re-exported)

console = get_console()

# Wells listed individually below the tally; the rest are in the CSV.
MAX_LISTED = 40

_VERDICT_STYLE = {
    "match": "green",
    "mixed": "yellow",
    "wrong construct": "red",
    "changed": "yellow",
    "unrecognised": "red",
    "too few reads": "muted",
    "no reads": "muted",
    "unexpected reads": "yellow",
}


def _fail(message: str, *hints: str) -> None:
    console.print(f"[red]Error:[/red] {escape(message)}")
    for hint in hints:
        console.print(f"  {hint}")
    raise typer.Exit(1)


def _describe(v) -> str:
    """The detail column: what differs, or what the note says."""
    parts = []
    if v.differences:
        shown = " ".join(v.differences[:4])
        if len(v.differences) > 4:
            shown += f" (+{len(v.differences) - 4})"
        parts.append(shown)
    if v.protein_changes:
        parts.append("(" + " ".join(v.protein_changes[:4]) + ")")
    if v.note:
        parts.append(v.note)
    return "; ".join(parts)


def _render_pileups(plate, verdicts, demux_dir: Path, output_dir: Path,
                    flank_5p: str, flank_3p: str, tools: dict, workers: int) -> dict:
    """A pileup and its summary for every well with reads, against the
    construct expected there.

    Against the expected construct rather than the one the pipeline assigned,
    so a well holding the wrong thing shows its reads disagreeing with what
    should be there.  A well with no expectation -- expected empty, or not
    listed -- is shown against what it appears to hold.

    Returns:
        ``{"<plate>_<384-well>": {"pileup": href, "summary": href}}``, relative
        to *output_dir*.
    """
    from usortm.demux.streakout import generate_pick_pileups, summary_path_for

    # The pileup reads the flank lengths from here to mark the insert; a
    # project demux writes it, and the plate check has to as well.
    summary_path = demux_dir / "demux_summary.json"
    if not summary_path.exists():
        summary_path.write_text(json.dumps({
            "flank_5p_len": len(flank_5p), "flank_3p_len": len(flank_3p),
        }))

    constructs = set(plate.constructs().values())
    first = next(iter(plate.constructs().values()), None)
    pick_list = []
    for v in verdicts:
        if v.reads <= 0:
            continue
        ref = v.expected or (v.observed if v.observed in constructs else first)
        if not ref:
            continue
        pick_list.append({
            "source_plate": v.plate, "source_well": v.well, "variant": ref,
            "reads": v.reads, "consensus_fraction": 0.0, "cons_check": "",
            "target_plate": v.plate, "target_well": v.well,
        })
    if not pick_list:
        return {}
    url_map = generate_pick_pileups(
        pick_list=pick_list,
        demux_output_dir=str(demux_dir),
        output_dir=str(output_dir),
        workers=workers,
        minimap2_path=tools["minimap2"],
        samtools_path=tools["samtools"],
        summaries=True,
    )
    return {f"{p}_{w}": {"pileup": url, "summary": summary_path_for(url)}
            for p, wells in url_map.items() for w, url in wells.items()}


def _usortm_commit() -> str:
    """The commit this usortm was installed from, or "" when not a checkout."""
    import subprocess

    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        if commit.returncode != 0:
            return ""
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit.stdout.strip() + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _command(parts: list) -> str:
    """A shell command, one option to a line."""
    import shlex

    lines, current = [], []
    for part in parts:
        if part.startswith("-") and current:
            lines.append(" ".join(current))
            current = []
        current.append(shlex.quote(part) if not part.startswith("-") else part)
    lines.append(" ".join(current))
    return " \\\n    ".join(lines)


def _write_commands(output_dir: Path, *, expected, fastq, vector, read_template,
                    derived_template, vector_fasta, mask_config_file, min_reads,
                    threads, workers, subsample, reads_per_well, tools,
                    columns=None) -> Path:
    """Write commands.txt: how to run this check again, on this machine or another.

    Paths are written absolute, so the command runs from any directory; on
    another machine they are the ones to change.
    """
    from usortm import __version__

    def a(path) -> str:
        return str(Path(path).expanduser().resolve())

    base = ["usortm", "demux", "--expected", a(expected), "--fastq", a(fastq)]
    if columns:
        base += ["--columns", columns]
    if vector is not None:
        base += ["--vector", a(vector)]
    if vector_fasta is not None:
        base += ["--vector-fasta", a(vector_fasta)]
    if mask_config_file and vector_fasta is not None:
        base += ["--mask-config", str(mask_config_file)]
    rest = ["--min-reads", str(min_reads), "--threads", str(threads),
            "--workers", str(workers)]
    if subsample:
        rest += ["--subsample", str(subsample)]
    if reads_per_well != 20:
        rest += ["--reads-per-well", str(reads_per_well)]
    out = ["-o", a(output_dir)]

    rerun = base + (["--read-template", a(read_template)] if read_template else []) + rest + out
    commit = _usortm_commit()
    install = (f'pip install "usortm[demux] @ git+https://github.com/FordyceLab/usortm@'
               f'{commit.split()[0]}"' if commit else "pip install \"usortm[demux]\"")
    lines = [
        "# How this plate check was run, so it can be run again.",
        f"# Written {datetime.now().strftime('%Y-%m-%d %H:%M')} by usortm {__version__}"
        + (f", commit {commit}" if commit else "") + ".",
        "",
        "# 1. Install the same usortm (Python 3.9+), with the demux extras:",
        f"{install}",
        "#    and the external tools on PATH: dorado (1.3+), minimap2, samtools.",
        "#    This run used:",
        *[f"#      {name}: {path}" for name, path in sorted(tools.items())],
        "",
        "# 2. Run the check.  Paths are absolute; change them on another machine.",
        "#    The output folder is overwritten.",
        _command(rerun),
    ]
    if derived_template is not None:
        lines += [
            "",
            "# 3. Or run it from the read layout this run worked out, which skips",
            "#    detecting it again.  Edit derived_read_template.fasta first if",
            "#    any part of the layout was wrong: it is one read, with the forward",
            "#    barcode, the variable region and the reverse barcode written as N.",
            _command([x for x in base if x != "--vector" and x != (a(vector) if vector else None)]
                     + ["--read-template", a(derived_template)] + rest + out),
        ]
    lines += [
        "",
        "# Then open plate_check.html in this folder.  Each well opens the summary",
        "# of its reads against the construct expected there; verification.csv has",
        "# one row per well.",
        "",
    ]
    path = output_dir / "commands.txt"
    path.write_text("\n".join(lines))
    return path


def run_expected_demux(
    *,
    expected: Path,
    fastq: Optional[Path],
    output_dir: Path,
    vector: Optional[Path],
    read_template: Optional[Path],
    vector_fasta: Optional[Path],
    mask_config_file: Optional[str],
    min_reads: int,
    threads: int,
    workers: int,
    subsample: Optional[int],
    reads_per_well: int,
    resume: bool,
    columns: Optional[str] = None,
) -> None:
    """Demultiplex a plate and check every well against its expectation."""
    from usortm.demux.deps import check_all_dependencies
    from usortm.demux.expected_plate import (
        ExpectedPlateError, parse_columns, read_expected_plate,
    )
    from usortm.demux.pipeline import run_levseq_pipeline
    from usortm.demux.read_template import (
        ReadTemplateError, parse_read_template, write_mask_config,
        write_vector_fasta,
    )
    from usortm.demux.verify import (
        stray_read_wells, tally, verify_plate, write_verification,
    )
    from usortm.cli.demux_cmd import (
        _find_fastqs, _load_mask_config, _resolve_mask_config,
    )

    console.print()
    console.print(Panel.fit("[brand]uSort-M[/brand] Plate check",
                            border_style=BORDER_STYLE))

    # --- Inputs ---------------------------------------------------------
    section(console, "Inputs")
    try:
        plate = read_expected_plate(expected, columns=parse_columns(columns))
    except (ExpectedPlateError, OSError) as exc:
        _fail(str(exc))
    n_expected = sum(1 for w in plate.wells.values() if not w.empty)
    n_constructs = len(plate.constructs())
    layout_word = ("96-well positions in LevSeq quadrants"
                   if plate.layout == "96" else "384-well positions")
    console.print(
        f"[green]✓[/green] Expected plate: {n_expected} wells, "
        f"{n_constructs} distinct construct(s), plate(s) "
        f"{', '.join(map(str, plate.plates))}, as {layout_word}"
    )
    for note in plate.notes:
        console.print(f"[yellow]⚠[/yellow] {note}")

    if fastq is None:
        _fail("--fastq is required.")
    fastqs = _find_fastqs(fastq)
    if not fastqs:
        _fail(f"No FASTQ files found at {fastq}.")
    console.print(f"[green]✓[/green] Reads: {len(fastqs)} FASTQ file(s) at {fastq}")

    try:
        tools = check_all_dependencies()
    except Exception as exc:
        _fail(str(exc), "Install missing tools or add them to your PATH.")

    given = [flag for flag, v in (("--vector", vector),
                                  ("--read-template", read_template),
                                  ("--vector-fasta", vector_fasta)) if v]
    # Whole amplicons in the CSV carry the layout themselves: the flanks are
    # what they all share, so no construct needs to be given.
    from_amplicons = not given and plate.amplicons
    if not given and not from_amplicons:
        _fail(
            "--expected needs the construct the reads come from.",
            "Give whole amplicons in the CSV's sequence column, or pass one of:",
            "  [cyan]--vector[/cyan] parent.fasta         "
            "(the parent vector; the read layout is worked out from it)",
            "  [cyan]--read-template[/cyan] read.fasta    "
            "(one read with both barcodes and the variable region masked)",
            "  [cyan]--vector-fasta[/cyan] vector.fasta   "
            "(the vector with the variable region masked)",
        )
    if len(given) > 1:
        _fail(f"Give only one of {', '.join(given)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    demux_dir = output_dir / "demux"
    reference = plate.write_reference_fasta(output_dir / "expected_reference.fasta")

    # --- Read layout ------------------------------------------------------
    section(console, "Read layout")
    layout_summary = None
    mask_config = None
    if vector is not None or from_amplicons:
        from usortm.demux.vector_layout import (
            LayoutError, detect_layout, write_read_template,
        )

        inserts = list(plate.constructs())
        amplicon = variable = None
        if from_amplicons:
            amplicon = plate.flank_5p + inserts[0] + plate.flank_3p
            variable = (len(plate.flank_5p), len(plate.flank_5p) + len(inserts[0]))
        origin = vector.name if vector is not None else Path(expected).name
        with console.status("Working out the read layout..."):
            try:
                layout = detect_layout(
                    vector, inserts, fastqs,
                    tools["minimap2"], output_dir / "read_layout",
                    threads=threads, amplicon=amplicon, variable=variable,
                )
            except LayoutError as exc:
                _fail(str(exc))
        read_template = write_read_template(
            layout, output_dir / "derived_read_template.fasta", source=origin,
        )
        layout_summary = layout.summary()
        console.print(f"[green]✓[/green] Detected from {origin} and the reads:")
        for line in layout.describe():
            console.print(f"  {escape(line)}")
        console.print(
            f"[muted]  Written as {read_template}; correct it by hand and pass "
            "it back with --read-template if any of this is wrong.[/muted]"
        )

    flank_5p = flank_3p = None
    if read_template is not None:
        try:
            parsed = parse_read_template(read_template)
        except ReadTemplateError as exc:
            _fail(str(exc))
        derived = output_dir / "read_template"
        vector_fasta = write_vector_fasta(parsed, derived / "vector.fasta")
        mask_config = _load_mask_config(
            write_mask_config(parsed, derived / "mask_config.toml",
                              source=read_template.name)
        )
        flank_5p, flank_3p = parsed.flank_5p, parsed.flank_3p
        if vector is None:
            console.print(f"[green]✓[/green] Read template: {parsed.describe()}")
        if mask_config_file is not None:
            console.print("[yellow]Warning:[/yellow] --mask-config is ignored; "
                          "the read template supplies the masks.")
    else:
        from usortm.demux.utils import parse_vector_fasta

        try:
            flank_5p, flank_3p = parse_vector_fasta(str(vector_fasta))
        except ValueError as exc:
            _fail(str(exc))
        if mask_config_file is not None:
            mask_config = _load_mask_config(_resolve_mask_config(mask_config_file))
        console.print(
            f"[green]✓[/green] Vector FASTA: 5' flank {len(flank_5p):,} bp, "
            f"3' flank {len(flank_3p):,} bp"
            + ("" if mask_config else ", default barcode masks")
        )

    # --- Demultiplex ------------------------------------------------------
    section(console, "Demultiplexing")
    started = datetime.now()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Starting pipeline...", total=None)
        results = run_levseq_pipeline(
            fastq=fastqs if len(fastqs) > 1 else fastqs[0],
            output_dir=demux_dir,
            reference=reference,
            n_plates=plate.n_plates,
            min_reads=min_reads,
            threads=threads,
            workers=workers,
            progress_callback=lambda msg: progress.update(task, description=msg),
            mask_config=mask_config,
            subsample=subsample,
            vector_fasta=vector_fasta,
            reads_per_well=reads_per_well,
            resume=resume,
        )
        progress.update(task, description="Checking each well against its construct...")
        verdicts = verify_plate(
            plate, demux_dir, flank_5p, flank_3p, tools,
            output_dir / "verify", min_reads=min_reads, workers=workers,
        )
        progress.update(task, description="Rendering pileups...")
        links = _render_pileups(plate, verdicts, demux_dir, output_dir,
                                flank_5p, flank_3p, tools, workers)
    console.print(
        f"[green]✓[/green] {results.get('input_reads', 0):,} reads, "
        f"{results.get('demuxed_reads', 0):,} with both barcodes, "
        f"in {(datetime.now() - started).seconds // 60} min "
        f"{(datetime.now() - started).seconds % 60} s"
    )

    # --- Results ------------------------------------------------------------
    csv_path = write_verification(verdicts, output_dir / "verification.csv")
    counts = tally(verdicts)
    strays = stray_read_wells(plate, demux_dir, min_reads)
    summary = {
        "expected_plate": str(expected),
        "fastq": [str(f) for f in fastqs],
        "min_reads": min_reads,
        "wells_expected": n_expected,
        "verdicts": counts,
        "stray_read_wells": strays,
        "reads": {k: results.get(k) for k in
                  ("input_reads", "aligned_reads", "demuxed_reads", "assigned_reads")},
        "read_layout": layout_summary,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2))

    from usortm.report.plate_check import write_plate_check_page

    # A well opens its pileup's summary; the pileup itself is beside it.
    page = write_plate_check_page(
        plate, verdicts, output_dir / "plate_check.html",
        links={k: v["summary"] for k, v in links.items()})

    section(console, "Wells")
    table = Table(box=box.ROUNDED, border_style=BORDER_STYLE, show_header=False)
    table.add_column("Verdict")
    table.add_column("Wells", justify="right")
    for verdict, n in counts.items():
        style = _VERDICT_STYLE.get(verdict, "")
        table.add_row(f"[{style}]{verdict}[/{style}]" if style else verdict, f"{n:,}")
    console.print(table)
    if strays:
        console.print(
            f"[muted]{strays} well(s) expected empty or not listed carry 1–"
            f"{min_reads - 1} reads, below the call; barcode crosstalk at that "
            "level is usual.[/muted]"
        )

    flagged = [v for v in verdicts if v.verdict != "match"]
    if flagged:
        section(console, "Wells to look at")
        t = Table(box=box.SIMPLE_HEAD, border_style=BORDER_STYLE)
        for col in ("Well", "Expected", "Verdict", "Holds", "Reads", "Detail"):
            t.add_column(col, justify="right" if col == "Reads" else "left",
                         overflow="fold")
        for v in flagged[:MAX_LISTED]:
            style = _VERDICT_STYLE.get(v.verdict, "")
            where = v.label if plate.layout == "96" else v.well
            t.add_row(
                f"{v.plate}:{where}", escape(v.expected or "—"),
                f"[{style}]{v.verdict}[/{style}]" if style else v.verdict,
                escape(v.observed or ""), f"{v.reads:,}", escape(_describe(v)),
            )
        console.print(t)
        if len(flagged) > MAX_LISTED:
            console.print(f"[muted]…and {len(flagged) - MAX_LISTED} more in "
                          f"{csv_path.name}.[/muted]")

    commands = _write_commands(
        output_dir, expected=expected, fastq=fastq, vector=vector,
        read_template=read_template if layout_summary is None else None,
        derived_template=(output_dir / "derived_read_template.fasta"
                          if layout_summary is not None else None),
        vector_fasta=vector_fasta if layout_summary is None else None,
        mask_config_file=mask_config_file, min_reads=min_reads,
        threads=threads, workers=workers, subsample=subsample,
        reads_per_well=reads_per_well, tools=tools, columns=columns,
    )

    section(console, "Outputs")
    console.print(f"  {page}   plate map; each well opens its pileup summary")
    console.print(f"  {commands}   how to run this again")
    console.print(f"  {csv_path}   one row per well")
    console.print(f"  {output_dir / 'verification_summary.json'}")
    if layout_summary is not None:
        console.print(f"  {output_dir / 'derived_read_template.fasta'}")
    console.print(f"  {demux_dir}/   pipeline outputs, including per-well reads")
    console.print()
