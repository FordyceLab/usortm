"""Read barcode mask sequences off a run's own reads.

Dorado locates a barcode by the sequences flanking it, which are specific to
the plasmid backbone.  Masks from a different construct classify almost
nothing while alignment still succeeds, which looks like a finished run with
empty wells.

Rather than guess, this finds the LevSeq barcodes in reads that are already
on disk and reports the sequence sitting on either side of them.
"""

from typing import Optional
from pathlib import Path
import collections
import gzip

import typer

from usortm.cli.theme import get_console

console = get_console()

# Bases of context to collect either side of a barcode. Long enough to see the
# constant region, short enough to stay above the noise.
CONTEXT = 30
DEFAULT_MASK_LEN = 22
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _iter_seqs(path: Path, limit: int):
    """Yield sequence lines from a FASTQ, plain or gzipped."""
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i // 4 >= limit:
                break
            if i % 4 == 1:
                yield line.strip().upper()


def derive(
    project_dir: Path = typer.Argument(
        ...,
        help="Path to a uSort-M project that has been demultiplexed at least once.",
        exists=True,
    ),
    reads: Optional[Path] = typer.Option(
        None,
        "--reads",
        help=(
            "FASTQ to read instead of the project's oriented reads. Use when "
            "no demux has run yet."
        ),
    ),
    mask_length: int = typer.Option(
        DEFAULT_MASK_LEN,
        "--mask-length",
        help="Bases either side of the barcode to use as the mask.",
        min=6, max=CONTEXT,
    ),
    max_reads: int = typer.Option(
        40000,
        "--max-reads",
        help="Reads to scan.",
        min=100,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Where to write the config. Defaults to <project>/mask_config.derived.toml.",
    ),
    round_num: int = typer.Option(1, "--round", help="Sequencing round.", min=1),
):
    """
    Derive barcode masks from a run's own reads.

    Finds the LevSeq barcodes in reads already on disk and reports the
    sequence flanking them, which is what Dorado needs to locate barcodes in
    this construct.

    [bold]Example:[/bold]

        usortm masks derive my_project/
    """
    from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC

    if reads is not None:
        fastq = reads
    else:
        demux_output = (project_dir / "rounds" / str(round_num) / "demux_output"
                        if round_num > 1 else project_dir / "demux_output")
        fastq = demux_output / "alignment" / "oriented_reads.fastq"
        if not fastq.exists():
            console.print(
                f"[red]Error:[/red] no reads found at {fastq}\n"
                "Run [cyan]usortm demux[/cyan] once first, or pass "
                "[cyan]--reads <fastq>[/cyan]."
            )
            raise typer.Exit(1)

    console.print(f"Scanning {fastq} for LevSeq barcodes...")

    # mask1 describes what flanks the FORWARD barcode, so only forward
    # barcodes are searched.  Mixing in reverse barcodes would average two
    # different positions in the amplicon into one meaningless consensus.
    forward_barcodes = [s.upper() for s in LEVSEQ_FBC]

    before = collections.Counter()
    after = collections.Counter()
    n_reads = n_hits = 0

    for seq in _iter_seqs(fastq, max_reads):
        n_reads += 1
        # Take the leftmost match, so the context is read at a consistent
        # position rather than wherever iteration order happened to land.
        best_pos, best_bc = None, None
        for bc in forward_barcodes:
            pos = seq.find(bc)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos, best_bc = pos, bc
        if best_pos is None:
            continue
        n_hits += 1
        before[seq[max(0, best_pos - CONTEXT):best_pos]] += 1
        after[seq[best_pos + len(best_bc):best_pos + len(best_bc) + CONTEXT]] += 1

    if not n_hits:
        console.print(
            f"[red]Error:[/red] no exact barcode matches in {n_reads:,} reads.\n"
            "The reads may not carry LevSeq barcodes, or their error rate is "
            "too high for exact matching."
        )
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found barcodes in {n_hits:,} of {n_reads:,} reads "
        f"({100 * n_hits / n_reads:.1f}%)"
    )

    top_before, n_before = before.most_common(1)[0]
    top_after, n_after = after.most_common(1)[0]
    front = top_before[-mask_length:]
    rear = top_after[:mask_length]

    console.print()
    console.print(f"  before barcode: [cyan]{front}[/cyan]  "
                  f"(in {100 * n_before / n_hits:.0f}% of hits)")
    console.print(f"  after barcode : [cyan]{rear}[/cyan]  "
                  f"(in {100 * n_after / n_hits:.0f}% of hits)")

    if n_before / n_hits < 0.25 or n_after / n_hits < 0.25:
        console.print(
            "\n[yellow]⚠[/yellow] The flanking sequence is not very consistent, "
            "so these masks may be unreliable. Check them against your plasmid "
            "map before relying on them."
        )

    out_path = output if output is not None else (
        project_dir / "mask_config.derived.toml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Barcode mask (flanking) sequences for Dorado demultiplexing.\n"
        "#\n"
        f"# Derived from {fastq.name}: the LevSeq barcodes were located in the\n"
        "# reads and the surrounding sequence read off directly. Masks are\n"
        "# specific to a plasmid backbone -- they are not transferable between\n"
        "# constructs.\n"
        "#\n"
        "# Only [fbc] is needed; the reverse-barcode masks are derived as\n"
        "# reverse complements at load time.\n\n"
        "[meta]\n"
        f'description = "Derived from {fastq.name}"\n\n'
        "[fbc]\n"
        f'mask1_front = "{front}"\n'
        f'mask1_rear  = "{rear}"\n'
        f'mask2_front = "{_rc(rear)}"\n'
        f'mask2_rear  = "{_rc(front)}"\n'
    )

    console.print()
    console.print(f"[green]✓[/green] Written to {out_path}")
    console.print()
    console.print("[bold]Next step:[/bold]")
    console.print(
        f"  [cyan]usortm demux {project_dir}/ --mask-config {out_path}[/cyan]"
    )
    console.print(
        "[muted]  Check the classification rate in the summary; if it is still "
        "low, try --mask-length 10 or 30.[/muted]"
    )


masks_app = typer.Typer(
    help="Inspect and derive barcode mask sequences.",
    no_args_is_help=True,
)
masks_app.command(name="derive")(derive)
