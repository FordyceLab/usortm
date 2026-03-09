"""Estimate costs and coverage for a uSort-M experiment."""

from typing import Optional
from pathlib import Path

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()


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
        4.0,
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
    actual_synthesis: Optional[float] = typer.Option(
        None,
        "--actual-synthesis",
        help="Actual synthesis cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    actual_cloning: Optional[float] = typer.Option(
        None,
        "--actual-cloning",
        help="Actual cloning cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    actual_sorting: Optional[float] = typer.Option(
        None,
        "--actual-sorting",
        help="Actual sorting cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    actual_barcoding: Optional[float] = typer.Option(
        None,
        "--actual-barcoding",
        help="Actual barcoding cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    actual_sequencing: Optional[float] = typer.Option(
        None,
        "--actual-sequencing",
        help="Actual sequencing cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    actual_hitpicking: Optional[float] = typer.Option(
        None,
        "--actual-hitpicking",
        help="Actual hit-picking cost paid (USD). Overrides the predicted value.",
        min=0.0,
    ),
    compare: bool = typer.Option(
        True,
        "--compare/--no-compare",
        help="Show comparison with traditional gene synthesis costs.",
    ),
    methods_dir: Optional[str] = typer.Option(
        None,
        "--methods-dir",
        help="Path to custom synthesis pricing TOML files (defaults to built-in methods).",
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
    from usortm.costs.method_loader import load_all_methods

    # Load pricing methods (custom dir or built-in)
    _loaded_methods = load_all_methods(methods_dir)
    # Populate cache so cost_functions uses the same methods
    cf._methods_cache = _loaded_methods

    # Calculate uSort-M costs
    total_wells = int(library_size * fold_sampling)
    n_plates = max(1, total_wells // 384)

    synthesis_cost = actual_synthesis if actual_synthesis is not None else cf.usortm_synthesis_cost(library_size, seq_length, methods_dir=methods_dir)
    cloning_cost = actual_cloning if actual_cloning is not None else cf.usortm_cloning_cost(library_size)
    sorting_cost = actual_sorting if actual_sorting is not None else cf.usortm_sorting_cost(
        library_size, fold_sampling=fold_sampling,
        machine_rate=machine_rate, operator_rate=operator_rate
    )
    barcoding_cost = actual_barcoding if actual_barcoding is not None else cf.usortm_barcoding_cost(n_wells=total_wells)
    sequencing_cost = actual_sequencing if actual_sequencing is not None else cf.usortm_sequencing_cost(n_wells=total_wells, seq_length=seq_length)
    hitpicking_cost = actual_hitpicking if actual_hitpicking is not None else cf.usortm_hitpicking_cost(library_size, seq_length)

    # Track which steps used actual vs. predicted costs
    _actual_flags = {
        "synthesis": actual_synthesis is not None,
        "cloning": actual_cloning is not None,
        "sorting": actual_sorting is not None,
        "barcoding": actual_barcoding is not None,
        "sequencing": actual_sequencing is not None,
        "hitpicking": actual_hitpicking is not None,
    }
    
    usortm_total = (
        synthesis_cost + cloning_cost + sorting_cost + 
        barcoding_cost + sequencing_cost + hitpicking_cost
    )
    
    # Calculate traditional costs for comparison
    if compare:
        trad_synthesis = cf.parsed_genefragments_synthesis_cost(
            seq_length, library_size, method='twist_genefragments'
        )
        trad_cloning = cf.parsed_genefragments_assembly_cost(
            library_size, assembly_method='hifi'
        )
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
            "actual_costs": {k: v for k, v in _actual_flags.items() if v},
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
        f"[brand]uSort-M[/brand] Cost Estimate\n"
        f"Library: [cyan]{library_size:,}[/cyan] variants × [cyan]{seq_length}[/cyan] bp",
        border_style=BORDER_STYLE,
    ))

    # Show pricing dates from loaded methods
    _dates = sorted(set(m.date_collected for m in _loaded_methods.values()))
    console.print(f"  [dim]Pricing date: {', '.join(_dates)}[/dim]")
    console.print()

    # Parameters summary
    param_table = Table(
        title="Simulation Parameters",
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
    )
    param_table.add_column("Parameter", style="muted")
    param_table.add_column("Value", justify="right")
    param_table.add_column("Parameter", style="muted")
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
    cost_table.add_column("Step", style="muted")
    cost_table.add_column("[brand.plain]uSort-M[/brand.plain]", justify="right", style="green")
    if compare:
        cost_table.add_column("Traditional", justify="right", style="yellow")
    
    def _step_label(name: str, *keys: str) -> str:
        if any(_actual_flags.get(k) for k in keys):
            return f"{name} [dim](actual)[/dim]"
        return name

    cost_table.add_row(
        _step_label("Synthesis", "synthesis"),
        f"${synthesis_cost:,.0f}",
        f"${trad_synthesis:,.0f}" if compare else None,
    )
    cost_table.add_row(
        _step_label("Cloning", "cloning"),
        f"${cloning_cost:,.0f}",
        f"${trad_cloning:,.0f}" if compare else None,
    )
    cost_table.add_row(
        _step_label("Sorting", "sorting"),
        f"${sorting_cost:,.0f}",
        "N/A" if compare else None,
    )
    cost_table.add_row(
        _step_label("Barcoding + Sequencing", "barcoding", "sequencing"),
        f"${barcoding_cost + sequencing_cost:,.0f}",
        f"${trad_sequencing:,.0f}" if compare else None,
    )
    cost_table.add_row(
        _step_label("Hit-picking", "hitpicking"),
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
            f"with [brand.plain]uSort-M[/brand.plain] (${trad_total - usortm_total:,.0f} saved)"
        )
        console.print()
    
    # Effort summary
    effort_table = Table(
        title="Effort Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    effort_table.add_column("Metric", style="muted")
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
