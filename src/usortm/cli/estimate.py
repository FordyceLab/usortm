"""Estimate costs and coverage for a uSort-M experiment."""

from typing import Optional
from pathlib import Path

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from usortm.cli.theme import get_console, BORDER_STYLE

console = get_console()


TILED_ASSEMBLY_INSERT_LENGTH = 30  # bp for tiled/substitution assembly inserts


def _prompt_synthesis_cost(seq_length, library_size, methods_dir=None):
    """Prompt user to select a synthesis method when none is auto-detected.

    Returns (cost, method_name) tuple. Falls back to (0, None) if
    non-interactive or user cancels.
    """
    import sys

    if not sys.stdin.isatty():
        return 0, None

    try:
        import questionary
    except ImportError:
        console.print("[yellow]⚠[/yellow] No synthesis pricing for {seq_length} bp sequences. "
                      "Install questionary (`pip install questionary`) for interactive method selection, "
                      "or use --actual-synthesis to specify a cost.")
        return 0, None

    from usortm.costs.method_loader import find_methods, compute_cost, load_all_methods

    console.print()
    console.print(
        f"[yellow]⚠[/yellow] No default pooled synthesis method for [cyan]{seq_length}[/cyan] bp sequences."
    )

    # Find compatible methods
    compatible = find_methods(seq_length, library_size=library_size, methods_dir=methods_dir)

    pooled = [m for m in compatible if m.type == "pooled"] if compatible else []

    # Build entries first so we can align the detail column across all rows.
    entries = []
    if pooled:
        entries.append(("section", "POOLED SYNTHESIS"))
        for m in pooled:
            detail = f"{m.seq_length_min}–{m.seq_length_max} bp"
            if m.skew_q90_q10:
                detail += f", skew {m.skew_q90_q10:.1f}×"
            entries.append(("choice", m.name, detail, m))

    entries.append(("section", "TILED ASSEMBLY"))
    entries.append((
        "choice",
        "Tiled assembly",
        f"{TILED_ASSEMBLY_INSERT_LENGTH} bp inserts, assembled into WT gene",
        "tiled",
    ))

    entries.append(("section", "OTHER"))
    entries.append((
        "choice",
        "Skip",
        "specify cost manually with --actual-synthesis",
        None,
    ))

    name_w = max(len(e[1]) for e in entries if e[0] == "choice")

    choices = []
    for i, e in enumerate(entries):
        if e[0] == "section":
            if i > 0:
                choices.append(questionary.Separator(" "))
            choices.append(questionary.Separator(f"  {e[1]}"))
        else:
            _, name, detail, value = e
            choices.append(questionary.Choice(
                title=f"{name.ljust(name_w)}  │  {detail}",
                value=value,
            ))

    try:
        answer = questionary.select(
            "Select synthesis method for cost estimate:",
            choices=choices,
        ).ask()
    except KeyboardInterrupt:
        return 0, None

    if answer is None:
        return 0, None

    if answer == "tiled":
        # Price the tiled inserts using Twist Oligo Pools at insert length
        methods = load_all_methods(methods_dir)
        m = methods.get("twist_oligo_pools")
        if m is not None:
            cost = compute_cost(m, library_size, TILED_ASSEMBLY_INSERT_LENGTH)
            if cost is not None:
                console.print(
                    f"  [green]✓[/green] Tiled assembly: {library_size:,} × {TILED_ASSEMBLY_INSERT_LENGTH} bp inserts "
                    f"→ [cyan]${cost:,.0f}[/cyan]"
                )
                return cost, f"Tiled assembly ({TILED_ASSEMBLY_INSERT_LENGTH} bp inserts)"
        return 0, None

    # A SynthesisMethod object was selected
    cost = compute_cost(answer, library_size, seq_length)
    if cost is not None:
        console.print(
            f"  [green]✓[/green] {answer.name}: {library_size:,} × {seq_length} bp "
            f"→ [cyan]${cost:,.0f}[/cyan]"
        )
        return cost, answer.name

    return 0, None


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
    fold_sampling: Optional[float] = typer.Option(
        None,
        "--fold-sampling", "-f",
        help="Fold oversampling during sorting. If set, the predicted coverage is simulated from it; if omitted, it is solved for from --target-coverage.",
        min=1.0,
        max=50.0,
    ),
    target_coverage: float = typer.Option(
        0.90,
        "--target-coverage",
        help="Target library coverage (0-1). Used to auto-determine fold-sampling when -f is not set.",
        min=0.5,
        max=1.0,
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
        help="Show comparison with direct gene synthesis costs.",
    ),
    sdm_compare: bool = typer.Option(
        False,
        "--sdm/--no-sdm",
        help="Show SDM cost comparison (for single-mutation libraries).",
    ),
    sdm_include_hifi: bool = typer.Option(
        False,
        "--sdm-hifi/--no-sdm-hifi",
        help="Include HiFi assembly step in SDM cost estimate.",
    ),
    round1_fold: float = typer.Option(
        3.0,
        "--round1-fold",
        help="Fold-sampling for round 1 in resynthesis strategy comparison.",
        min=1.0,
        max=20.0,
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
    report: bool = typer.Option(
        False,
        "--report/--no-report",
        help="Generate a PDF cost summary report.",
    ),
):
    """
    Estimate cost and effort for a uSort-M experiment.
    
    Calculates projected costs for synthesis, cloning, sorting, barcoding,
    sequencing, and hit-picking based on your library parameters.

    Library size, skew, and fold-sampling set the predicted coverage, which is
    simulated and reported alongside the costs.

    [bold]Example:[/bold]

        usortm estimate --library-size 500 --seq-length 300 --fold-sampling 8
    """
    from usortm.costs import cost_functions as cf
    from usortm.costs.method_loader import load_all_methods

    # Load pricing methods (custom dir or built-in)
    _loaded_methods = load_all_methods(methods_dir)
    # Populate cache so cost_functions uses the same methods
    cf._methods_cache = _loaded_methods

    from usortm.simulate.sortm import expected_coverage as _predict_coverage

    # Fold-sampling is either given (-f) or searched for from --target-coverage.
    fold_sampling_auto = fold_sampling is None

    status = None
    if not json_output:
        console.print()
        _status_msg = (
            f"Simulating fold-sampling for {target_coverage:.0%} coverage..."
            if fold_sampling_auto
            else f"Simulating coverage at {fold_sampling:g}× fold-sampling..."
        )
        status = console.status(f"[muted]{_status_msg}[/muted]")
        status.start()

    if fold_sampling_auto:
        from usortm.simulate.sortm import find_fold_sampling

        def _progress(iteration, fs, cov):
            if status is not None:
                status.update(
                    f"[muted]Simulating... {fs:.1f}× → {cov:.1%} coverage[/muted]"
                )

        fold_sampling, _ = find_fold_sampling(
            target_coverage=target_coverage,
            lib_size=library_size,
            skew=skew,
            p_grow=sorting_efficiency,
            n_sims=100,
            seed=42,
            progress_callback=_progress,
        )

    # Predict coverage at the fold-sampling we will use. Library size, skew,
    # fold-sampling and sorting efficiency fix this, so an explicit -f gets the
    # same prediction the search reports when it lands on the same value.
    _prediction = _predict_coverage(
        fold_sampling=fold_sampling,
        lib_size=library_size,
        skew=skew,
        p_grow=sorting_efficiency,
        n_sims=100,
        seed=42,
    )
    expected_coverage = _prediction["coverage"]
    coverage_sd = _prediction["coverage_sd"]
    coverage_p10 = _prediction["coverage_p10"]
    coverage_p90 = _prediction["coverage_p90"]

    if status is not None:
        status.stop()
        _fold_str = f"{fold_sampling:g}×"
        if fold_sampling_auto:
            _fold_str += " [dim](auto)[/dim]"
        console.print(
            f"  [dim]Simulation:[/dim] [cyan]{_fold_str}[/cyan] fold-sampling "
            f"→ [cyan]{expected_coverage:.1%}[/cyan] expected coverage "
            f"[dim]({coverage_p10:.0%}–{coverage_p90:.0%} across "
            f"{_prediction['n_sims']} sims)[/dim]"
        )

    # Calculate uSort-M costs
    total_wells = int(library_size * fold_sampling)
    n_plates = max(1, (total_wells + 383) // 384)

    synthesis_cost = actual_synthesis if actual_synthesis is not None else cf.usortm_synthesis_cost(library_size, seq_length, methods_dir=methods_dir)
    synthesis_method_name = None

    # If no auto-detected synthesis method (e.g. >350 bp), prompt the user
    if synthesis_cost == 0 and actual_synthesis is None and not json_output:
        synthesis_cost, synthesis_method_name = _prompt_synthesis_cost(
            seq_length, library_size, methods_dir
        )

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
        trad_barcoding = cf.parsed_genefragments_barcoding_cost(library_size)
        trad_sequencing = cf.parsed_genefragments_sequencing_cost(seq_length, library_size)
        trad_total = trad_synthesis + trad_cloning + trad_barcoding + trad_sequencing

    # Calculate SDM costs for comparison
    if sdm_compare:
        sdm_primers = cf.sdm_primer_cost(library_size)
        sdm_kit = cf.sdm_kit_cost(library_size, include_hifi=sdm_include_hifi)
        sdm_transformation = cf.sdm_transformation_cost(library_size)
        sdm_consumables = cf.sdm_consumables_cost(library_size)
        sdm_barcoding = cf.parsed_genefragments_barcoding_cost(library_size)
        sdm_sequencing = cf.parsed_genefragments_sequencing_cost(seq_length, library_size)
        sdm_total = (
            sdm_primers + sdm_kit + sdm_transformation + sdm_consumables
            + sdm_barcoding + sdm_sequencing
        )
    
    # Resynthesis strategy simulation
    resynth = None
    two_round_total = None
    if fold_sampling_auto:
        from usortm.simulate.sortm import simulate_resynthesis_strategy

        if not json_output:
            resynth_status = console.status(
                "[muted]Simulating resynthesis strategy...[/muted]"
            )
            resynth_status.start()

        def _resynth_progress(step, msg):
            if not json_output:
                resynth_status.update(f"[muted]{msg}[/muted]")

        resynth = simulate_resynthesis_strategy(
            target_coverage=target_coverage,
            lib_size=library_size,
            skew=skew,
            round1_fold=round1_fold,
            p_grow=sorting_efficiency,
            n_sims=100,
            seed=42,
            progress_callback=_resynth_progress,
        )

        if not json_output:
            resynth_status.stop()

        # Calculate two-round costs
        dropout_n = resynth["dropout_count"]

        # Use the already-determined synthesis_cost (which may be user-selected via
        # interactive prompt) rather than re-calling the auto-detect function, which
        # returns $0 for sequences without a default pooled method (e.g. >350 bp).
        _cost_per_variant = synthesis_cost / library_size if library_size > 0 else 0
        r1_synthesis = synthesis_cost  # same library, same method
        r1_cloning = cf.usortm_cloning_cost(library_size)
        r1_sorting = cf.usortm_sorting_cost(library_size, fold_sampling=round1_fold,
                                             machine_rate=machine_rate, operator_rate=operator_rate)
        r1_barcoding = cf.usortm_barcoding_cost(n_wells=resynth["round1_wells"])
        r1_sequencing = cf.usortm_sequencing_cost(n_wells=resynth["round1_wells"], seq_length=seq_length)
        r1_hitpicking = cf.usortm_hitpicking_cost(resynth["round1_recovered"], seq_length)
        r1_total = r1_synthesis + r1_cloning + r1_sorting + r1_barcoding + r1_sequencing + r1_hitpicking

        resynth_synthesis = _cost_per_variant * dropout_n if dropout_n > 0 else 0

        r2_cloning = cf.usortm_cloning_cost(dropout_n) if dropout_n > 0 else 0
        r2_sorting = cf.usortm_sorting_cost(dropout_n, fold_sampling=resynth["round2_fold"],
                                             machine_rate=machine_rate, operator_rate=operator_rate) if dropout_n > 0 else 0
        r2_barcoding = cf.usortm_barcoding_cost(n_wells=resynth["round2_wells"]) if resynth["round2_wells"] > 0 else 0
        r2_sequencing = cf.usortm_sequencing_cost(n_wells=resynth["round2_wells"], seq_length=seq_length) if resynth["round2_wells"] > 0 else 0
        r2_hitpicking = cf.usortm_hitpicking_cost(resynth.get("round2_recovered", 0), seq_length) if dropout_n > 0 else 0
        r2_total = resynth_synthesis + r2_cloning + r2_sorting + r2_barcoding + r2_sequencing + r2_hitpicking

        two_round_total = r1_total + r2_total

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
            "fold_sampling_auto": fold_sampling_auto,
            "expected_coverage": round(expected_coverage, 4),
            "coverage_sd": round(coverage_sd, 4),
            "coverage_p10": round(coverage_p10, 4),
            "coverage_p90": round(coverage_p90, 4),
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
                "barcoding": round(trad_barcoding, 2),
                "sequencing": round(trad_sequencing, 2),
                "total": round(trad_total, 2),
            }
            result["savings_fold"] = round(trad_total / usortm_total, 1)
        if sdm_compare:
            result["sdm"] = {
                "primers": round(sdm_primers, 2),
                "q5_sdm_kit": round(sdm_kit, 2),
                "transformation": round(sdm_transformation, 2),
                "consumables": round(sdm_consumables, 2),
                "barcoding": round(sdm_barcoding, 2),
                "sequencing": round(sdm_sequencing, 2),
                "total": round(sdm_total, 2),
            }
            result["sdm_savings_fold"] = round(sdm_total / usortm_total, 1)
        if resynth is not None:
            result["resynthesis"] = {
                "round1_fold": resynth["round1_fold"],
                "round1_coverage": round(resynth["round1_coverage"], 4),
                "round1_recovered": resynth["round1_recovered"],
                "dropout_count": resynth["dropout_count"],
                "round2_fold": resynth["round2_fold"],
                "round2_coverage": round(resynth["round2_coverage"], 4),
                "total_coverage": round(resynth["total_coverage"], 4),
                "total_wells": resynth["total_wells"],
                "round1_cost": round(r1_total, 2),
                "resynthesis_cost": round(resynth_synthesis, 2),
                "round2_cost": round(r2_total, 2),
                "two_round_total": round(two_round_total, 2),
                "single_round_total": round(usortm_total, 2),
                "savings": round(usortm_total - two_round_total, 2),
            }
        console.print(json.dumps(result, indent=2))
        return
    
    # Rich output
    console.print()
    console.print(Panel.fit(
        f"[brand]uSort-M[/brand] Cost Estimate\n"
        f"Library: [cyan]{library_size:,}[/cyan] variants × [cyan]{seq_length}[/cyan] bp",
        border_style=BORDER_STYLE,
    ))

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
    
    _fold_label = f"{fold_sampling}×"
    if fold_sampling_auto:
        _fold_label += f" [dim](auto, {target_coverage:.0%} target)[/dim]"
    param_table.add_row(
        "Fold sampling", _fold_label,
        "Library skew", f"{skew}× (Q90/Q10)",
    )
    _eff_label = f"{sorting_efficiency:.0%}"
    if expected_coverage is not None:
        _eff_label += f" → {expected_coverage:.0%} coverage"
    param_table.add_row(
        "Sorting efficiency", _eff_label,
        "FACS rate", f"${machine_rate + operator_rate:.0f}/hr",
    )
    console.print(param_table)
    console.print()
    
    import math as _math
    from usortm.costs.time_functions import calculate_total_timeline

    def _step_label(name: str, *keys: str) -> str:
        if any(_actual_flags.get(k) for k in keys):
            return f"{name} [dim](actual)[/dim]"
        return name

    def _format_time(hours):
        days = hours / 8
        if hours < 1:
            return f"{hours*60:.0f} min"
        elif days <= 1:
            return f"{hours:.1f} hours"
        else:
            return f"{days:.1f} days ({hours:.0f} hrs)"

    # ── Section 1: uSort-M Cost Breakdown ──
    _breakdown_show_two_round = (
        resynth is not None
        and two_round_total is not None
        and two_round_total < usortm_total
    )
    _breakdown_title = "uSort-M Cost Breakdown"
    if _breakdown_show_two_round:
        _breakdown_title += " (with resynthesis)"
    cost_table = Table(
        title=_breakdown_title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    cost_table.add_column("Step", style="muted")
    cost_table.add_column("Total", justify="right", style="green")
    cost_table.add_column("Per Sequence", justify="right", style="green")

    def _per_seq(v):
        return f"${v / library_size:,.2f}"

    cost_table.add_row(_step_label("Synthesis", "synthesis"), f"${synthesis_cost:,.0f}", _per_seq(synthesis_cost))
    cost_table.add_row(_step_label("Cloning", "cloning"), f"${cloning_cost:,.0f}", _per_seq(cloning_cost))
    cost_table.add_row(_step_label("Sorting", "sorting"), f"${sorting_cost:,.0f}", _per_seq(sorting_cost))
    cost_table.add_row(_step_label("Barcoding", "barcoding"), f"${barcoding_cost:,.0f}", _per_seq(barcoding_cost))
    cost_table.add_row(_step_label("Sequencing", "sequencing"), f"${sequencing_cost:,.0f}", _per_seq(sequencing_cost))
    cost_table.add_row(_step_label("Hit-picking", "hitpicking"), f"${hitpicking_cost:,.0f}", _per_seq(hitpicking_cost))

    # When resynthesis is simulated and cheaper, show both single-round and
    # 2-round totals so downstream comparisons (alt methods) can reference
    # whichever strategy the user would actually run.
    show_two_round = _breakdown_show_two_round
    if show_two_round:
        cost_table.add_row(
            "Total (single-round)",
            f"${usortm_total:,.0f}",
            _per_seq(usortm_total),
        )
        cost_table.add_row(
            "[bold]Total (with resynthesis)[/bold]",
            f"[bold green]${two_round_total:,.0f}[/bold green]",
            f"[bold green]{_per_seq(two_round_total)}[/bold green]",
        )
    else:
        cost_table.add_row(
            "[bold]Total[/bold]",
            f"[bold green]${usortm_total:,.0f}[/bold green]",
            f"[bold green]{_per_seq(usortm_total)}[/bold green]",
        )

    # Best uSort-M total — used by the Alt Methods comparison row below.
    best_usortm_total = two_round_total if show_two_round else usortm_total

    console.print(cost_table)
    console.print()

    # ── Section 2: Strategy Comparison (single vs 2-round) ──
    sr_timeline = calculate_total_timeline(library_size, seq_length, fold_sampling=fold_sampling)

    if resynth is not None:
        dropout_n = resynth["dropout_count"]

        r1_tl = calculate_total_timeline(library_size, seq_length, fold_sampling=round1_fold)
        resynth_wait_days = 7
        r2_tl = calculate_total_timeline(dropout_n, seq_length, fold_sampling=resynth["round2_fold"])
        two_round_days = r1_tl["total_days"] + resynth_wait_days + r2_tl["total_days"]

        r1_wells_n = resynth["round1_wells"]
        r2_wells_n = resynth["round2_wells"]
        r1_plates_n = max(1, -(-r1_wells_n // 384))
        r2_plates_n = max(1, -(-r2_wells_n // 384)) if r2_wells_n > 0 else 0

        strat_table = Table(
            title="Strategy Comparison",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        strat_table.add_column("", style="muted")
        strat_table.add_column("Single-round", justify="right")
        strat_table.add_column("2-round", justify="right")

        strat_table.add_row(
            "[bold]Total cost[/bold]",
            f"[bold green]${usortm_total:,.0f}[/bold green]",
            f"[bold green]${two_round_total:,.0f}[/bold green]",
        )
        strat_table.add_row(
            "  R1 cost",
            f"[dim]—[/dim]",
            f"${r1_total:,.0f} [dim]({round1_fold}× → {resynth['round1_coverage']:.0%})[/dim]",
        )
        strat_table.add_row(
            "  Resynthesize",
            f"[dim]—[/dim]",
            f"${resynth_synthesis:,.0f} [dim]({dropout_n:,} dropouts)[/dim]",
        )
        strat_table.add_row(
            "  R2 cost",
            f"[dim]—[/dim]",
            f"${r2_total - resynth_synthesis:,.0f} [dim]({resynth['round2_fold']}× → {resynth['round2_coverage']:.0%} of dropouts)[/dim]",
        )
        strat_table.add_row(
            "Coverage",
            f"{expected_coverage:.0%}",
            f"{resynth['total_coverage']:.0%}",
        )
        strat_table.add_row(
            "Wells",
            f"{total_wells:,}",
            f"{r1_wells_n + r2_wells_n:,} [dim]({r1_wells_n:,} + {r2_wells_n:,})[/dim]",
        )
        strat_table.add_row(
            "Plates",
            f"{n_plates}",
            f"{r1_plates_n + r2_plates_n} [dim]({r1_plates_n} + {r2_plates_n})[/dim]",
        )
        strat_table.add_row(
            "Working days",
            f"{sr_timeline['total_days']}",
            f"{two_round_days} [dim]({r1_tl['total_days']} + {resynth_wait_days} wait + {r2_tl['total_days']})[/dim]",
        )
        strat_table.add_row(
            "Cost/variant",
            f"${usortm_total/library_size:.2f}",
            f"${two_round_total/library_size:.2f}",
        )

        # Recovery curve plot
        try:
            import plotext as plt

            # Calibrate analytical rate from simulation endpoints.
            # Both curves sample the same pool in round 1, so use a
            # single rate calibrated from the round-1 simulation point.
            r1_cov_end = resynth["round1_coverage"]
            r1_max = resynth["round1_wells"]
            pool_rate = -r1_max / (library_size * _math.log(1 - min(r1_cov_end, 0.999)))

            r2_cov_end = resynth["round2_coverage"]
            r2_max = resynth["round2_wells"]
            r2_rate = -r2_max / (dropout_n * _math.log(1 - min(r2_cov_end, 0.999))) if dropout_n > 0 and r2_cov_end < 0.999 else 1.5

            max_wells = int(library_size * fold_sampling * 1.1)
            step = max(1, max_wells // 80)
            sr_wells = list(range(0, max_wells + 1, step))
            sr_cov = [1 - _math.exp(-w / (library_size * pool_rate)) for w in sr_wells]

            r1_wells_list = list(range(0, r1_max + 1, step))
            r1_cov_list = [1 - _math.exp(-w / (library_size * pool_rate)) for w in r1_wells_list]

            r2_wells_list = list(range(0, r2_max + 1, max(1, r2_max // 40)))
            r2_cov_list = [
                r1_cov_end + (1 - r1_cov_end) * (1 - _math.exp(-w / (dropout_n * r2_rate)))
                for w in r2_wells_list
            ]

            combined_wells = r1_wells_list + [r1_max + w for w in r2_wells_list[1:]]
            combined_cov = r1_cov_list + r2_cov_list[1:]

            # Mark the actual well counts each strategy uses (matches the
            # "Wells" row in Strategy Comparison and the savings footnote).
            sr_cross = total_wells
            resynth_cross = resynth["total_wells"]

            plt.clear_figure()
            plt.plot(sr_wells, sr_cov, color="blue")
            plt.plot(combined_wells, combined_cov, color="green")
            plt.hline(target_coverage, color="red")

            # Top ticks showing where each curve hits target.
            # Use left alignment so the ▼ glyph itself sits at the x-intercept,
            # with the well-count label trailing to the right.
            if resynth_cross is not None:
                plt.text(f"▼{int(resynth_cross):,}", resynth_cross, 0.99, color="green", background="default", alignment="left")
            if sr_cross is not None:
                plt.text(f"▼{int(sr_cross):,}", sr_cross, 0.99, color="blue", background="default", alignment="left")

            legend_x = max_wells * 0.95
            plt.text("── Single-round", legend_x, 0.12, color="blue", background="default", alignment="right")
            plt.text("── With resynthesis", legend_x, 0.04, color="green", background="default", alignment="right")

            plt.title("Recovery Curve")
            plt.xlabel("Wells sorted")
            plt.ylabel("Coverage")
            plt.ylim(0, 1)
            plt.plotsize(50, 18)
            plt.theme("clear")
            plot_str = plt.build()

            from rich.columns import Columns
            from rich.text import Text
            console.print(Columns([strat_table, Text.from_ansi(plot_str)], padding=(0, 2)))

        except ImportError:
            console.print(strat_table)

        diff = usortm_total - two_round_total
        if diff > 0:
            console.print(
                f"\n  [bold green]Resynthesis saves ${diff:,.0f}[/bold green] "
                f"({diff / usortm_total:.0%} less) with fewer wells ({resynth['total_wells']:,} vs {total_wells:,})"
            )
        else:
            console.print(
                f"\n  [dim]Single-round is ${-diff:,.0f} cheaper than resynthesis for this library[/dim]"
            )
        console.print()

    else:
        # No resynthesis data — show simple effort summary
        effort_table = Table(
            title="Effort Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        effort_table.add_column("Metric", style="muted")
        effort_table.add_column("Value", justify="right")

        effort_table.add_row(
            "Predicted coverage",
            f"[green]{expected_coverage:.1%}[/green] "
            f"[dim]({coverage_p10:.0%}–{coverage_p90:.0%})[/dim]",
        )
        effort_table.add_row(
            "Variants recovered",
            f"{round(expected_coverage * library_size):,} of {library_size:,}",
        )
        effort_table.add_row("Total wells to sort", f"{total_wells:,}")
        effort_table.add_row("384-well plates", f"{n_plates}")
        effort_table.add_row("Sorting time", _format_time(sort_hours))
        effort_table.add_row("Barcoding time", _format_time(barcode_hours))
        effort_table.add_row("Working days", f"{sr_timeline['total_days']}")
        effort_table.add_row("Cost per variant", f"${usortm_total/library_size:.2f}")

        console.print(effort_table)
        console.print()

    # ── Section 3: Alternative Methods ──
    if compare or sdm_compare:
        alt_table = Table(
            title="Alternative Methods",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        alt_table.add_column("Step", style="muted")
        if compare:
            alt_table.add_column("Direct Synthesis", justify="right", style="yellow")
            alt_table.add_column("Per Sequence", justify="right", style="yellow")
        if sdm_compare:
            alt_table.add_column("SDM", justify="right", style="magenta")
            alt_table.add_column("Per Sequence", justify="right", style="magenta")

        def _fmt_per_seq(v):
            return f"${v / library_size:,.2f}" if isinstance(v, (int, float)) else ""

        def _alt_row(label, trad_val=None, sdm_val=None, trad_num=None, sdm_num=None):
            """Add a row with only the columns that exist.

            *_val is the display string; *_num is the raw numeric cost used
            for the per-sequence column (None for N/A / meta rows).
            """
            cols = [label]
            if compare:
                cols.append(trad_val or "")
                cols.append(_fmt_per_seq(trad_num))
            if sdm_compare:
                cols.append(sdm_val or "")
                cols.append(_fmt_per_seq(sdm_num))
            alt_table.add_row(*cols)

        _alt_row("Synthesis",
                 f"${trad_synthesis:,.0f}" if compare else None,
                 f"${sdm_primers:,.0f}" if sdm_compare else None,
                 trad_num=trad_synthesis if compare else None,
                 sdm_num=sdm_primers if sdm_compare else None)
        # SDM cloning = Q5 SDM kit + transformation + consumables, shown as
        # one row so the breakdown mirrors the uSort-M / Direct Synthesis layout.
        sdm_cloning_combined = (sdm_kit + sdm_transformation + sdm_consumables) if sdm_compare else None
        _alt_row("Cloning",
                 f"${trad_cloning:,.0f}" if compare else None,
                 f"${sdm_cloning_combined:,.0f}" if sdm_compare else None,
                 trad_num=trad_cloning if compare else None,
                 sdm_num=sdm_cloning_combined if sdm_compare else None)
        _alt_row("Barcoding",
                 f"${trad_barcoding:,.0f}" if compare else None,
                 f"${sdm_barcoding:,.0f}" if sdm_compare else None,
                 trad_num=trad_barcoding if compare else None,
                 sdm_num=sdm_barcoding if sdm_compare else None)
        _alt_row("Sequencing",
                 f"${trad_sequencing:,.0f}" if compare else None,
                 f"${sdm_sequencing:,.0f}" if sdm_compare else None,
                 trad_num=trad_sequencing if compare else None,
                 sdm_num=sdm_sequencing if sdm_compare else None)
        _alt_row("[bold]Total[/bold]",
                 f"[bold yellow]${trad_total:,.0f}[/bold yellow]" if compare else None,
                 f"[bold magenta]${sdm_total:,.0f}[/bold magenta]" if sdm_compare else None,
                 trad_num=trad_total if compare else None,
                 sdm_num=sdm_total if sdm_compare else None)
        _vs_label = "[bold]vs uSort-M[/bold]"
        if best_usortm_total < usortm_total:
            _vs_label = "[bold]vs uSort-M[/bold] [dim](with resynthesis)[/dim]"
        _alt_row(_vs_label,
                 f"[bold green]{trad_total / best_usortm_total:.1f}× savings[/bold green]" if compare else None,
                 f"[bold green]{sdm_total / best_usortm_total:.1f}× savings[/bold green]" if sdm_compare else None)

        if sdm_compare:
            n_failures = _math.ceil(library_size * cf.SDM_FAILURE_RATE)
            console.print(alt_table)
            console.print(f"  [dim]SDM assumes {cf.SDM_FAILURE_RATE:.0%} failure rate ({n_failures:,} reclones for {library_size:,} genes)[/dim]")
            console.print()
        else:
            console.print(alt_table)
            console.print()

    # ── PDF Report ──
    if report and not json_output:
        from usortm.costs.report import generate_estimate_report
        from usortm.costs.time_functions import calculate_total_timeline as _calc_tl

        _sr_tl = _calc_tl(library_size, seq_length, fold_sampling=fold_sampling)

        report_data = {
            "library_size": library_size,
            "seq_length": seq_length,
            "fold_sampling": fold_sampling,
            "fold_sampling_auto": fold_sampling_auto,
            "expected_coverage": expected_coverage,
            "target_coverage": target_coverage,
            "skew": skew,
            "sorting_efficiency": sorting_efficiency,
            "synthesis_cost": synthesis_cost,
            "synthesis_method_name": synthesis_method_name,
            "cloning_cost": cloning_cost,
            "sorting_cost": sorting_cost,
            "barcoding_cost": barcoding_cost,
            "sequencing_cost": sequencing_cost,
            "hitpicking_cost": hitpicking_cost,
            "usortm_total": usortm_total,
            "resynth": resynth,
            "round1_fold": round1_fold,
            "r1_total": r1_total if resynth is not None else None,
            "resynth_synthesis": resynth_synthesis if resynth is not None else None,
            "r2_total": r2_total if resynth is not None else None,
            "two_round_total": two_round_total,
            "sr_timeline_days": _sr_tl["total_days"],
            "two_round_days": two_round_days if resynth is not None else None,
            "compare": compare,
            "trad_total": trad_total if compare else None,
            "trad_synthesis": trad_synthesis if compare else None,
            "trad_cloning": trad_cloning if compare else None,
            "trad_barcoding": trad_barcoding if compare else None,
            "trad_sequencing": trad_sequencing if compare else None,
            "sdm_compare": sdm_compare,
            "sdm_total": sdm_total if sdm_compare else None,
            "sdm_primers": sdm_primers if sdm_compare else None,
            "sdm_kit": sdm_kit if sdm_compare else None,
            "sdm_transformation": sdm_transformation if sdm_compare else None,
            "sdm_consumables": sdm_consumables if sdm_compare else None,
            "sdm_barcoding": sdm_barcoding if sdm_compare else None,
            "sdm_sequencing": sdm_sequencing if sdm_compare else None,
            "sdm_include_hifi": sdm_include_hifi,
            "pricing_dates": ", ".join(sorted(set(
                m.date_collected for m in _loaded_methods.values()
            ))),
        }

        path = generate_estimate_report(report_data)
        console.print(f"  [green]✓[/green] Report saved to [cyan]{path}[/cyan]")
        console.print()
