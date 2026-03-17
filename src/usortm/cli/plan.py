"""Plan and orchestrate a uSort-M experiment workflow."""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import csv
import json
from datetime import datetime

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()

# Project state file structure
PROJECT_STATE_FILE = "usortm_project.json"


def plan(
    variants_file: Path = typer.Argument(
        ...,
        help="CSV file with variant definitions (columns: name, sequence or mutation).",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("usortm_project"),
        "--output", "-o",
        help="Output directory for project files.",
    ),
    seq_length: Optional[int] = typer.Option(
        None,
        "--seq-length", "-l",
        help="Length of the variable region in base pairs (auto-detected from sequences if omitted).",
    ),
    fold_sampling: float = typer.Option(
        8.0,
        "--fold-sampling", "-f",
        help="Fold oversampling during sorting.",
    ),
    skew: Optional[float] = typer.Option(
        None,
        "--skew", "-s",
        help="Expected library skew (Q90/Q10 ratio). If omitted, you will be prompted to select a synthesis method.",
    ),
    target_coverage: float = typer.Option(
        0.90,
        "--target-coverage",
        help="Target fraction of library to recover.",
    ),
    barcode_kit: str = typer.Option(
        "levseq",
        "--barcodes", "-b",
        help="Barcode kit to use: 'levseq' (recommended), 'evseq', or path to custom CSV.",
    ),
    round_num: int = typer.Option(
        1,
        "--round", "-r",
        help="Sequencing round number (1 for initial sort, 2+ for re-order rounds).",
        min=1,
    ),
):
    """
    Plan a [#4096E3]uSort-M[/#4096E3] experiment from a variant list.
    
    Creates a project directory with:
    
    • Cost estimates and sorting recommendations
    • Barcode plate assignments
    • Template files for downstream steps
    
    [bold]Workflow:[/bold]
    
    1. [cyan]usortm plan[/cyan] variants.csv → Create project, get sorting plan
    2. [muted]Wet lab: synthesize, clone, sort, barcode, sequence[/muted]
    3. [cyan]usortm demux[/cyan] project/ --fastq data.fastq → Demultiplex reads
    4. [cyan]usortm pick[/cyan] project/ → Generate hit-picking list
    5. [cyan]usortm report[/cyan] project/ → Final plate maps
    
    [bold]Example:[/bold]
    
        usortm plan my_variants.csv --output acyp_project/ --seq-length 300
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print()
    console.print(Panel.fit(
        "[brand]uSort-M[/brand] Experiment Planner",
        border_style=BORDER_STYLE,
    ))
    console.print()
    
    # Read variants
    variants = _read_variants(variants_file)
    library_size = len(variants)

    # Validate variant names for problematic characters
    _validate_variant_names(variants)

    # Auto-detect sequence length if not provided
    if seq_length is None:
        seq_length = _infer_seq_length(variants)
        console.print(f"[green]✓[/green] Auto-detected sequence length: [cyan]{seq_length} bp[/cyan]")

    console.print(f"[green]✓[/green] Loaded [cyan]{library_size}[/cyan] variants from {variants_file.name}")

    # Select synthesis method (determines skew for simulation)
    synthesis_method_slug = None
    if skew is None:
        selected_method = _prompt_synthesis_method(seq_length, library_size)
        if selected_method is not None:
            synthesis_method_slug = selected_method.slug
            if selected_method.skew_q90_q10 is not None:
                skew = selected_method.skew_q90_q10
                console.print(
                    f"[green]✓[/green] Using [cyan]{selected_method.name}[/cyan] "
                    f"(skew {skew:.1f}× Q90/Q10)"
                )
            else:
                skew = 1.0  # arrayed synthesis: uniform
                console.print(
                    f"[green]✓[/green] Using [cyan]{selected_method.name}[/cyan] "
                    f"(arrayed synthesis, skew ~1×)"
                )
        else:
            skew = 4.0  # default fallback

    # Calculate sorting requirements
    total_wells = int(library_size * fold_sampling)
    n_plates = max(1, (total_wells + 383) // 384)  # Round up

    # Estimate costs (import here to avoid circular imports)
    from usortm.costs import cost_functions as cf

    synthesis_cost = cf.usortm_synthesis_cost(library_size, seq_length)
    cloning_cost = cf.usortm_cloning_cost(library_size)
    sorting_cost = cf.usortm_sorting_cost(library_size, fold_sampling=fold_sampling)
    barcoding_cost = cf.usortm_barcoding_cost(n_wells=total_wells)
    sequencing_cost = cf.usortm_sequencing_cost(n_wells=total_wells, seq_length=seq_length)
    hitpicking_cost = cf.usortm_hitpicking_cost(library_size, seq_length)
    total_cost = synthesis_cost + cloning_cost + sorting_cost + barcoding_cost + sequencing_cost + hitpicking_cost

    # Predict coverage using simulation
    try:
        import numpy as np
        from usortm.simulate.sortm import sortm
        _simulate_ok = True
    except (ImportError, SystemError):
        _simulate_ok = False

    if _simulate_ok:
        console.print("[muted]Running coverage simulation...[/muted]")
        predicted_variants = sortm(
            n_sims=100,
            lib_size=library_size,
            fold_sampling=fold_sampling,
            skew=skew,
            p_grow=0.67,  # Typical sorting efficiency
            return_correct=True,
            seed=42,
        )
        expected_coverage = np.mean(predicted_variants) / library_size
        coverage_std = np.std(predicted_variants) / library_size
    else:
        # Approximate coverage analytically when simulation unavailable
        import math
        expected_coverage = 1.0 - math.exp(-fold_sampling / skew)
        coverage_std = 0.0

    # Check if fold sampling is sufficient
    if expected_coverage < target_coverage:
        console.print()
        console.print(f"[yellow]⚠ Warning:[/yellow] Expected coverage ([cyan]{expected_coverage:.1%}[/cyan]) "
                     f"is below target ([cyan]{target_coverage:.1%}[/cyan])")
        recommended_fold = fold_sampling * (target_coverage / expected_coverage) * 1.1  # 10% buffer
        console.print(f"  → Consider increasing fold sampling to [cyan]{recommended_fold:.1f}×[/cyan]")
        console.print()

    # Display summary
    console.print()
    summary_table = Table(
        title="Experiment Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Parameter", style="muted")
    summary_table.add_column("Value", justify="right")
    summary_table.add_column("Parameter", style="muted")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row(
        "Library size", f"{library_size:,}",
        "Sequence length", f"{seq_length} bp",
    )
    summary_table.add_row(
        "Fold sampling", f"{fold_sampling}×",
        "Target coverage", f"{target_coverage:.0%}",
    )
    coverage_color = "green" if expected_coverage >= target_coverage else "yellow"
    skew_label = f"{skew:.1f}× (Q90/Q10)"
    if synthesis_method_slug:
        from usortm.costs.method_loader import load_all_methods
        _all = load_all_methods()
        _sm = _all.get(synthesis_method_slug)
        if _sm and _sm.skew_q90_q10 is not None:
            skew_label = f"{skew:.1f}× Q90/Q10 ({_sm.name})"
    summary_table.add_row(
        "Predicted coverage", f"[{coverage_color}]{expected_coverage:.1%}[/{coverage_color}] (±{coverage_std:.1%})",
        "Library skew", skew_label,
    )
    summary_table.add_row(
        "Total wells", f"{total_wells:,}",
        "384-well plates", f"{n_plates}",
    )
    summary_table.add_row(
        "Barcode kit", barcode_kit,
        "Estimated cost", f"[green]${total_cost:,.0f}[/green]",
    )
    
    console.print(summary_table)
    console.print()
    
    # Generate barcode assignments
    barcode_assignments = _generate_barcode_assignments(n_plates, barcode_kit, output_dir)

    # Write default mask config for user to customize
    _write_default_mask_config(output_dir)
    
    # Save project state
    project_state = {
        "created": datetime.now().isoformat(),
        "status": "planned",
        "round": round_num,
        "variants_file": str(variants_file.absolute()),
        "library_size": library_size,
        "seq_length": seq_length,
        "fold_sampling": fold_sampling,
        "skew": skew,
        "synthesis_method": synthesis_method_slug,
        "target_coverage": target_coverage,
        "expected_coverage": round(expected_coverage, 4),
        "coverage_std": round(coverage_std, 4),
        "n_plates": n_plates,
        "total_wells": total_wells,
        "barcode_kit": barcode_kit,
        "costs": {
            "synthesis": round(synthesis_cost, 2),
            "cloning": round(cloning_cost, 2),
            "sorting": round(sorting_cost, 2),
            "barcoding": round(barcoding_cost, 2),
            "sequencing": round(sequencing_cost, 2),
            "hitpicking": round(hitpicking_cost, 2),
            "total": round(total_cost, 2),
        },
        "workflow_steps": {
            "plan": {"completed": True, "timestamp": datetime.now().isoformat()},
            "demux": {"completed": False},
            "pick": {"completed": False},
            "report": {"completed": False},
        }
    }
    
    with open(output_dir / PROJECT_STATE_FILE, "w") as f:
        json.dump(project_state, f, indent=2)
    
    # Copy variants to project
    _save_variants(variants, output_dir / "variants.csv")
    
    # Generate sorting instructions
    _write_sorting_instructions(output_dir, library_size, n_plates, fold_sampling, skew)
    
    # Display generated files
    console.print("[green]✓[/green] Generated project files:")
    console.print(f"  • {output_dir}/usortm_project.json (project state)")
    console.print(f"  • {output_dir}/variants.csv (variant list)")
    console.print(f"  • {output_dir}/sorting_instructions.md (sorting guide)")
    console.print(f"  • {output_dir}/barcodes/ (barcode assignments)")
    console.print(f"  • {output_dir}/mask_config.toml (barcode flanking sequences)")
    console.print()
    
    # Display timeline
    console.print("[bold]Estimated Timeline:[/bold]")
    console.print("  [cyan]Day 1:[/cyan] Pooled assembly + transformation")
    console.print("  [cyan]Day 2:[/cyan] FACS isolation into plates")
    console.print("  [cyan]Day 3:[/cyan] PCR barcoding + pooling")
    console.print("  [cyan]Days 4-6:[/cyan] Sequencing turnaround")
    console.print()
    
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  1. Order oligo pool ({library_size} variants, {seq_length} bp)")
    console.print(f"  2. Follow sorting instructions in [cyan]{output_dir}/sorting_instructions.md[/cyan]")
    console.print(f"  3. After sequencing, run: [cyan]usortm demux {output_dir}/ --fastq <data.fastq>[/cyan]")
    console.print()


def _read_variants(variants_file: Path) -> list[dict]:
    """Read variants from CSV file."""
    variants = []
    with open(variants_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append(dict(row))
    return variants


def _save_variants(variants: list[dict], output_path: Path):
    """Save variants to CSV file."""
    if not variants:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=variants[0].keys())
        writer.writeheader()
        writer.writerows(variants)


def _infer_seq_length(variants: list[dict]) -> int:
    """Infer sequence length from variant sequences.

    Looks for a ``Sequence`` or ``sequence`` column, computes the median
    length of uppercase-only characters (matching
    ``csv_to_reference_fasta(strip_flanking=True)`` behavior).  Falls back
    to 300 bp if no sequence column is found.
    """
    import statistics

    seq_col = None
    for col_name in ("Sequence", "sequence"):
        if variants and col_name in variants[0]:
            seq_col = col_name
            break

    if seq_col is None:
        return 300

    lengths = []
    for v in variants:
        seq = v.get(seq_col, "")
        upper_only = "".join(c for c in seq if c.isupper())
        if upper_only:
            lengths.append(len(upper_only))

    if not lengths:
        return 300

    return int(statistics.median(lengths))


def _validate_variant_names(variants: list[dict]) -> None:
    """Check variant names for characters that break downstream processing.

    Protected characters: ``/`` (file paths), ``|`` (FASTA/cons_check
    delimiter), ``>`` (FASTA header), and whitespace.  Aborts with an
    error message showing up to 10 offending names.
    """
    import re as _re

    name_col = None
    for col_name in ("Name", "name", "variant"):
        if variants and col_name in variants[0]:
            name_col = col_name
            break

    if name_col is None:
        return

    bad_pattern = _re.compile(r'[/|>\s]')
    offenders: list[tuple[int, str]] = []

    for idx, v in enumerate(variants, start=2):  # row 2 = first data row (1-indexed + header)
        name = v.get(name_col, "")
        if bad_pattern.search(name):
            offenders.append((idx, name))

    if offenders:
        console.print()
        console.print("[red]Error:[/red] Variant names contain characters that break downstream processing.")
        console.print("Protected characters: [cyan]/[/cyan]  [cyan]|[/cyan]  [cyan]>[/cyan]  whitespace")
        console.print()
        for row_num, name in offenders[:10]:
            console.print(f"  Row {row_num}: [yellow]{name!r}[/yellow]")
        if len(offenders) > 10:
            console.print(f"  ... and {len(offenders) - 10} more")
        console.print()
        console.print("Please fix the variant names and re-run.")
        raise typer.Exit(1)


def _generate_barcode_assignments(n_plates: int, barcode_kit: str, output_dir: Path) -> dict:
    """Generate barcode assignments for plates."""
    barcode_dir = output_dir / "barcodes"
    barcode_dir.mkdir(exist_ok=True)
    
    if barcode_kit.lower() == "levseq":
        return _generate_levseq_barcodes(n_plates, barcode_dir)
    elif barcode_kit.lower() == "evseq":
        return _generate_evseq_barcodes(n_plates, barcode_dir)
    elif Path(barcode_kit).exists():
        return _load_custom_barcodes(Path(barcode_kit), n_plates, barcode_dir)
    else:
        console.print(f"[yellow]Warning:[/yellow] Unknown barcode kit '{barcode_kit}', using LevSeq defaults")
        return _generate_levseq_barcodes(n_plates, barcode_dir)


def _generate_levseq_barcodes(n_plates: int, barcode_dir: Path) -> dict:
    """Generate LevSeq barcode assignments with actual sequences.

    Uses the standard LevSeq NB/RB barcode set (96 forward, 96 reverse).
    Each 384-well plate uses 96 FBCs and 4 RBCs (one per quadrant).
    Also writes Dorado-ready TOML and FASTA config files.
    """
    from usortm.demux.barcodes import (
        LEVSEQ_FBC,
        LEVSEQ_RBC,
        get_rbc_count_for_plates,
        write_levseq_fbc_fasta,
        write_levseq_fbc_toml,
        write_levseq_rbc_fasta,
        write_levseq_rbc_toml,
    )
    from usortm.demux.utils import barcode_to_well

    n_rbc = get_rbc_count_for_plates(n_plates)

    # Write barcode CSV with actual sequences
    barcode_path = barcode_dir / "levseq_barcodes.csv"

    with open(barcode_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plate", "well", "fbc_id", "rbc_id", "fbc_seq", "rbc_seq",
        ])

        for rbc_idx in range(n_rbc):
            for fbc_idx in range(96):
                fbc_name = f"FB{fbc_idx + 1:02d}"
                rbc_name = f"RB{rbc_idx + 1:02d}"

                well_pos = barcode_to_well(fbc_name, rbc_name)
                if well_pos is None:
                    continue

                # Parse the well position (e.g. "1A1" -> plate=1, well="A1")
                import re
                m = re.match(r"(\d+)([A-P])(\d+)", well_pos)
                if not m:
                    continue
                plate = m.group(1)
                well = f"{m.group(2)}{m.group(3)}"

                writer.writerow([
                    plate,
                    well,
                    fbc_name,
                    rbc_name,
                    LEVSEQ_FBC[fbc_idx],
                    LEVSEQ_RBC[rbc_idx],
                ])

    # Also write Dorado-ready config files
    write_levseq_fbc_toml(barcode_dir)
    write_levseq_rbc_toml(barcode_dir, n_barcodes=n_rbc)
    write_levseq_fbc_fasta(barcode_dir)
    write_levseq_rbc_fasta(barcode_dir, n_barcodes=n_rbc)

    console.print(
        f"[green]\u2713[/green] Generated LevSeq barcodes: {barcode_path}"
    )
    return {"kit": "levseq", "file": str(barcode_path)}


def _generate_evseq_barcodes(n_plates: int, barcode_dir: Path) -> dict:
    """Generate evSeq barcode assignments."""
    # evSeq uses dual indexing: well barcodes + plate (Nextera) barcodes
    
    template_path = barcode_dir / "evseq_template.csv"
    
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "well", "fwd_barcode", "rev_barcode", "plate_i5", "plate_i7"])
        
        # Nextera plate barcodes
        nextera = {
            1: ("TTTAGTCATTGA", "TCTCTTGTGTGG"),
            2: ("AAGATACAAGAG", "GTGGTATCTGAC"),
            3: ("GAGGATACACAT", "CAGAAGGCTTAT"),
            4: ("CCGCCCGCTCTA", "TAGCCACCAGCA"),
            5: ("TCCGCTCGGTAA", "TCCTCGTCCTCT"),
            6: ("CCTCACGCATCG", "GGACAATGCCCA"),
            7: ("TCGGACTCCTCG", "TATCACAATCCA"),
            8: ("CCAGCCATCCCG", "CGTGGTAACTTC"),
        }
        
        for plate in range(1, n_plates + 1):
            plate_bc = nextera.get(plate, nextera[1])
            for row_idx, row in enumerate("ABCDEFGH"):
                for col in range(1, 13):
                    well = f"{row}{col}"
                    writer.writerow([plate, well, "", "", plate_bc[0], plate_bc[1]])
    
    console.print(f"[green]✓[/green] Generated evSeq barcode template: {template_path}")
    return {"kit": "evseq", "template": str(template_path)}


def _load_custom_barcodes(barcode_file: Path, n_plates: int, barcode_dir: Path) -> dict:
    """Load custom barcode assignments from user file."""
    import shutil
    dest = barcode_dir / "custom_barcodes.csv"
    shutil.copy(barcode_file, dest)
    console.print(f"[green]✓[/green] Copied custom barcodes to: {dest}")
    return {"kit": "custom", "file": str(dest)}


def _write_sorting_instructions(output_dir: Path, library_size: int, n_plates: int, 
                                fold_sampling: float, skew: float):
    """Write sorting instructions markdown file."""
    total_wells = int(library_size * fold_sampling)
    sort_hours = (n_plates * 30 + 60) / 60
    
    content = f"""# uSort-M Sorting Instructions

## Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Library size | {library_size:,} variants |
| Fold sampling | {fold_sampling}× |
| Expected skew | {skew}× (Q90/Q10) |
| Total wells | {total_wells:,} |
| 384-well plates | {n_plates} |
| Estimated sort time | {sort_hours:.1f} hours |

## Pre-Sort Checklist

- [ ] Pooled assembly completed
- [ ] Transformation performed (target: {library_size * 10:,}+ CFU)
- [ ] Overnight culture grown
- [ ] 384-well plates prepared with growth medium
- [ ] FACS instrument reserved ({sort_hours:.1f} hours)

## Sorting Protocol

### Day 1: Assembly & Transformation

1. Perform pooled Golden Gate assembly
2. Transform into competent cells
3. Plate on selective media
4. Incubate overnight

### Day 2: FACS Sorting

1. Prepare single-cell suspension from transformation plate
2. Set up FACS for single-cell sorting into 384-well plates
3. Sort settings:
   - Single-cell mode
   - Target: 1 cell per well
   - Gate on live cells (FSC/SSC)
4. Sort {n_plates} plates × 384 wells = {total_wells:,} total events
5. Incubate sorted plates overnight

### Day 3: Barcoding PCR

1. Check plates for growth (OD600 or visual inspection)
2. Expected growth rate: ~67% of wells
3. Perform barcoding PCR according to barcode kit protocol
4. Pool barcoded samples
5. Submit for sequencing

## Plate Layout

```
Plate 1:  Wells A1-P24 (384 wells)
Plate 2:  Wells A1-P24 (384 wells)
...
Plate {n_plates}: Wells A1-P24 ({total_wells - (n_plates-1)*384} wells used)
```

## Expected Outcomes

With {fold_sampling}× sampling and {skew}× skew:
- Expected unique variants recovered: ~{int(library_size * 0.9):,} ({90}% coverage)
- Expected wells with growth: ~{int(total_wells * 0.67):,}
- Expected sequencing depth needed: ~500 reads/well

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Low growth rate (<50%) | Sorting stress | Reduce sort pressure |
| High doublet rate | Poor gating | Tighten single-cell gate |
| Low coverage | High skew | Increase fold sampling |
| Many empty wells | Low CFU input | Increase transformation scale |

## Next Steps

After sequencing is complete:

```bash
usortm demux {output_dir.name}/ --fastq <your_data.fastq>
```
"""
    
    (output_dir / "sorting_instructions.md").write_text(content)


def _prompt_synthesis_method(seq_length: int, library_size: int):
    """Interactively prompt the user to select a synthesis method.

    Filters available methods to those compatible with the given sequence
    length and library size, then presents a questionary select prompt.

    Args:
        seq_length: Length of the variable region in bp.
        library_size: Number of variants in the library.

    Returns:
        A SynthesisMethod if one was selected, or None to use the default skew.
    """
    import questionary
    from usortm.costs.method_loader import find_methods

    compatible = find_methods(seq_length, library_size=library_size)

    if not compatible:
        return None

    def _skew_label(m):
        if m.skew_q90_q10 is None:
            return "uniform (arrayed)"
        return f"Q90/Q10 = {m.skew_q90_q10:.1f}×"

    choices = [
        questionary.Choice(
            title=f"{m.name}  [{_skew_label(m)}]",
            value=m,
        )
        for m in compatible
    ]
    choices.append(questionary.Choice(title="Skip — specify skew manually (--skew)", value=None))

    console.print()
    import sys
    if not sys.stdin.isatty():
        return None
    try:
        answer = questionary.select(
            "Select your pooled synthesis method (used to model library skew):",
            choices=choices,
        ).ask()
    except KeyboardInterrupt:
        return None

    return answer


def _write_default_mask_config(output_dir: Path):
    """Write a default mask_config.toml with cutinase backbone sequences.

    Users can edit this file to match their plasmid backbone before
    running ``usortm demux``.  The demux command auto-detects this
    file in the project directory.

    Only the ``[fbc]`` section is written; the ``[rbc]`` masks are
    automatically derived (reverse-complement swap) at load time.
    """
    from usortm.demux.barcodes import DEFAULT_MASKS, DEFAULT_SCORING

    fbc = DEFAULT_MASKS["fbc"]

    content = f"""# Barcode mask (flanking) sequences for Dorado demultiplexing.
#
# These sequences flank the barcode cassettes in the plasmid backbone
# and help Dorado locate barcodes in each read.  Edit these to match
# YOUR plasmid backbone before running `usortm demux`.
#
# Only the [fbc] section is needed — the reverse-barcode (RBC) masks
# are automatically derived as reverse complements at load time.
#
# The defaults below are for the cutinase expression vector.
# You can also use a built-in preset: usortm config list

[meta]
description = "Cutinase expression vector (T7 backbone)"

[fbc]
mask1_front = "{fbc['mask1_front']}"
mask1_rear  = "{fbc['mask1_rear']}"
mask2_front = "{fbc['mask2_front']}"
mask2_rear  = "{fbc['mask2_rear']}"

# Uncomment and edit to override Dorado barcode scoring parameters.
# These rarely need tuning — adjust only if you see low barcode
# classification rates.
#
# [scoring]
# max_barcode_penalty = {DEFAULT_SCORING['max_barcode_penalty']}
# min_barcode_penalty_dist = {DEFAULT_SCORING['min_barcode_penalty_dist']}
# flank_right_pad = {DEFAULT_SCORING['flank_right_pad']}
# flank_left_pad = {DEFAULT_SCORING['flank_left_pad']}
# min_separation_only_dist = {DEFAULT_SCORING['min_separation_only_dist']}
# min_flank_score = {DEFAULT_SCORING['min_flank_score']}
# barcode_end_proximity = {DEFAULT_SCORING['barcode_end_proximity']}
"""
    (output_dir / "mask_config.toml").write_text(content)
