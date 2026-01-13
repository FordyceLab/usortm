"""Estimate costs and coverage for a uSort-M experiment."""

from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def estimate(
    library_size: int = typer.Option(
        ...,
        "--library-size", "-n",
        help="Number of unique variants in your library.",
        min=1,
        max=20000,
    ),
    seq_length: int = typer.Option(
        300,
        "--seq-length", "-l",
        help="Length of the variable region in base pairs.",
        min=30,
        max=2000,
    ),
    fold_sampling: float = typer.Option(
        8.0,
        "--fold-sampling", "-f",
        help="Fold oversampling during sorting (wells sorted / library size).",
        min=1.0,
        max=50.0,
    ),
    skew: float = typer.Option(
        4.0,
        "--skew", "-s",
        help="Library skew (90th/10th percentile abundance ratio).",
        min=1.0,
        max=100.0,
    ),
    sorting_efficiency: float = typer.Option(
        0.67,
        "--sorting-efficiency",
        help="Fraction of sorted wells that grow successfully.",
        min=0.1,
        max=1.0,
    ),
    machine_rate: float = typer.Option(
        70.0,
        "--machine-rate",
        help="FACS machine hourly rate in USD.",
    ),
    operator_rate: float = typer.Option(
        65.0,
        "--operator-rate", 
        help="FACS operator hourly rate in USD.",
    ),
    compare: bool = typer.Option(
        True,
        "--compare/--no-compare",
        help="Show comparison with traditional gene synthesis costs.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON.",
    ),
):
    """
    Estimate cost and effort for a uSort-M experiment.
    
    Calculates projected costs for synthesis, cloning, sorting, barcoding,
    sequencing, and hit-picking based on your library parameters.
    
    [bold]Example:[/bold]
    
        usortm estimate --library-size 500 --seq-length 300 --fold-sampling 8
    """
    from usortm.costs import cost_functions as cf
    
    # Calculate uSort-M costs
    synthesis_cost = cf.usortm_synthesis_cost(library_size, seq_length)
    cloning_cost = cf.usortm_cloning_cost(library_size)
    sorting_cost = _calculate_sorting_cost(
        library_size, fold_sampling, machine_rate, operator_rate
    )
    barcoding_cost = cf.usortm_barcoding_cost(library_size * fold_sampling / library_size)  
    # Recalculate with actual wells
    total_wells = int(library_size * fold_sampling)
    n_plates = max(1, total_wells // 384)
    barcoding_cost = n_plates * 97.73
    sequencing_cost = cf.usortm_sequencing_cost(library_size, seq_length)
    hitpicking_cost = cf.usortm_hitpicking_cost(library_size, seq_length)
    
    usortm_total = (
        synthesis_cost + cloning_cost + sorting_cost + 
        barcoding_cost + sequencing_cost + hitpicking_cost
    )
    
    # Calculate traditional costs for comparison
    if compare:
        trad_synthesis = library_size * 35  # ~$35 per gene fragment
        trad_cloning = library_size * (2680/250 + 165/(6*200) * 10)  # Per-variant assembly + transform
        trad_sequencing = cf.parsed_genefragments_sequencing_cost(seq_length, library_size)
        trad_total = trad_synthesis + trad_cloning + trad_sequencing
    
    # Calculate effort metrics - 8 min/plate sort + 30 min setup
    sort_minutes = n_plates * 8 + 30
    sort_hours = sort_minutes / 60
    
    # Barcoding time: ~50 min per plate (8-10 plates per 8-hour day)
    barcode_minutes = n_plates * 50
    barcode_hours = barcode_minutes / 60
    
    # Calculate days for each step (assuming 8-hour workday)
    sort_days = max(1, (sort_hours + 2) / 8)  # +2 hours for setup/cleanup
    barcode_days = max(1, barcode_hours / 8)
    
    if json_output:
        import json
        result = {
            "library_size": library_size,
            "seq_length": seq_length,
            "fold_sampling": fold_sampling,
            "costs": {
                "synthesis": round(synthesis_cost, 2),
                "cloning": round(cloning_cost, 2),
                "sorting": round(sorting_cost, 2),
                "barcoding": round(barcoding_cost, 2),
                "sequencing": round(sequencing_cost, 2),
                "hitpicking": round(hitpicking_cost, 2),
                "total": round(usortm_total, 2),
            },
            "effort": {
                "total_wells": total_wells,
                "n_plates": n_plates,
                "sort_hours": round(sort_hours, 1),
                "sort_days": round(sort_days, 1),
                "barcode_hours": round(barcode_hours, 1),
                "barcode_days": round(barcode_days, 1),
            },
        }
        if compare:
            result["traditional"] = {
                "synthesis": round(trad_synthesis, 2),
                "cloning": round(trad_cloning, 2),
                "sequencing": round(trad_sequencing, 2),
                "total": round(trad_total, 2),
            }
            result["savings_fold"] = round(trad_total / usortm_total, 1)
        console.print(json.dumps(result, indent=2))
        return
    
    # Rich output
    console.print()
    console.print(Panel.fit(
        f"[bold blue]uSort-M[/bold blue] Cost Estimate\n"
        f"Library: [cyan]{library_size:,}[/cyan] variants × [cyan]{seq_length}[/cyan] bp",
        border_style="blue",
    ))
    console.print()
    
    # Parameters summary
    param_table = Table(
        title="Simulation Parameters",
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
    )
    param_table.add_column("Parameter", style="dim")
    param_table.add_column("Value", justify="right")
    param_table.add_column("Parameter", style="dim")
    param_table.add_column("Value", justify="right")
    
    param_table.add_row(
        "Fold sampling", f"{fold_sampling}×",
        "Library skew", f"{skew}× (Q90/Q10)",
    )
    param_table.add_row(
        "Sorting efficiency", f"{sorting_efficiency:.0%}",
        "FACS rate", f"${machine_rate + operator_rate:.0f}/hr",
    )
    console.print(param_table)
    console.print()
    
    # Cost breakdown table
    cost_table = Table(
        title="Cost Breakdown",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    cost_table.add_column("Step", style="dim")
    cost_table.add_column("[blue]uSort-M[/blue]", justify="right", style="green")
    if compare:
        cost_table.add_column("Traditional", justify="right", style="yellow")
    
    cost_table.add_row(
        "Synthesis",
        f"${synthesis_cost:,.0f}",
        f"${trad_synthesis:,.0f}" if compare else None,
    )
    cost_table.add_row(
        "Cloning",
        f"${cloning_cost:,.0f}",
        f"${trad_cloning:,.0f}" if compare else None,
    )
    cost_table.add_row(
        "Sorting",
        f"${sorting_cost:,.0f}",
        "N/A" if compare else None,
    )
    cost_table.add_row(
        "Barcoding + Sequencing",
        f"${barcoding_cost + sequencing_cost:,.0f}",
        f"${trad_sequencing:,.0f}" if compare else None,
    )
    cost_table.add_row(
        "Hit-picking",
        f"${hitpicking_cost:,.0f}",
        "N/A" if compare else None,
    )
    cost_table.add_row(
        "[bold]Total[/bold]",
        f"[bold green]${usortm_total:,.0f}[/bold green]",
        f"[bold yellow]${trad_total:,.0f}[/bold yellow]" if compare else None,
    )
    
    console.print(cost_table)
    console.print()
    
    if compare:
        savings = trad_total / usortm_total
        console.print(
            f"  [bold green]{savings:.1f}-fold savings[/bold green] "
            f"with [blue]uSort-M[/blue] (${trad_total - usortm_total:,.0f} saved)"
        )
        console.print()
    
    # Effort summary
    effort_table = Table(
        title="Effort Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    effort_table.add_column("Metric", style="dim")
    effort_table.add_column("Value", justify="right")
    
    effort_table.add_row("Total wells to sort", f"{total_wells:,}")
    effort_table.add_row("384-well plates", f"{n_plates}")
    
    # Format sort time appropriately
    if sort_hours < 1:
        effort_table.add_row("Sorting time", f"{sort_minutes:.0f} min")
    elif sort_days <= 1:
        effort_table.add_row("Sorting time", f"{sort_hours:.1f} hours")
    else:
        effort_table.add_row("Sorting time", f"{sort_days:.1f} days ({sort_hours:.0f} hours)")
    
    # Format barcoding time
    if barcode_hours < 1:
        effort_table.add_row("Barcoding time", f"{barcode_hours*60:.0f} min")
    elif barcode_days <= 1:
        effort_table.add_row("Barcoding time", f"{barcode_hours:.0f} hours")
    else:
        effort_table.add_row("Barcoding time", f"{barcode_days:.1f} days ({barcode_hours:.0f} hours)")
    
    effort_table.add_row("Cost per variant", f"${usortm_total/library_size:.2f}")
    
    console.print(effort_table)
    console.print()
    
    # Dynamic timeline based on experiment size
    console.print("[bold]Estimated Timeline:[/bold]")
    console.print("  Day 1: Pooled assembly + transformation")
    
    if sort_days <= 1:
        console.print("  Day 2: FACS isolation into plates")
        barcode_start = 3
    else:
        console.print(f"  Days 2-{1 + int(sort_days)}: FACS isolation into plates")
        barcode_start = 2 + int(sort_days)
    
    if barcode_days <= 1:
        console.print(f"  Day {barcode_start}: PCR barcoding + pooling")
        seq_start = barcode_start + 1
    else:
        console.print(f"  Days {barcode_start}-{barcode_start + int(barcode_days) - 1}: PCR barcoding + pooling")
        seq_start = barcode_start + int(barcode_days)
    
    console.print(f"  Days {seq_start}-{seq_start + 2}: Sequencing + analysis")
    console.print()


def _calculate_sorting_cost(
    library_size: int,
    fold_sampling: float,
    machine_rate: float,
    operator_rate: float,
) -> float:
    """Calculate sorting costs based on configurable rates."""
    total_wells = int(library_size * fold_sampling)
    n_plates = max(1, total_wells // 384)
    sort_minutes = n_plates * 8 + 30  # 8 min/plate + 30 min setup
    hourly_rate = machine_rate + operator_rate
    return (sort_minutes / 60) * hourly_rate
