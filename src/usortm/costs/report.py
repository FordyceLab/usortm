"""Generate PDF cost estimate reports using matplotlib."""

import math
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages


# Palette
_BLUE = "#4ba5e2"
_GREY = "#6b7280"
_GREEN = "#22c55e"
_HEADER_COLOR = "#e5e7eb"
_ALT_ROW = "#f9fafb"

# Layout constants (figure coords; figure is 8.5 × 11 in)
_ROW_H = 0.021    # figure-coord height per table row at scale(1, 1.4) fontsize 8
_TITLE_H = 0.022  # vertical space used by a section title + gap below it
_SEC_GAP = 0.015  # gap between one section's bottom and the next title


def _fmt_cost(v):
    """Format a cost value for display."""
    if v is None or v == 0:
        return "$0"
    if v >= 1000:
        return f"${v:,.0f}"
    return f"${v:.0f}"


def _table_height(n_data_rows):
    """Approximate figure-coord height for a table (header + data rows)."""
    return (n_data_rows + 1) * _ROW_H


def _col_widths_from_content(col_labels, data):
    """Proportional column widths from max character count per column."""
    n = len(col_labels)
    chars = [len(str(h)) for h in col_labels]
    for row in data:
        for j, cell in enumerate(row):
            chars[j] = max(chars[j], len(str(cell)))
    total = sum(chars)
    return [c / total for c in chars] if total else [1.0 / n] * n


def _add_table(ax, col_labels, data, col_widths=None):
    """Render a table on a matplotlib axes with clean formatting."""
    ax.axis("off")

    n_cols = len(col_labels)
    if col_widths is None:
        col_widths = _col_widths_from_content(col_labels, data)

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="upper center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(_HEADER_COLOR)
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_edgecolor("#d1d5db")

    # Style data rows — left-align first column
    for i in range(1, len(data) + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor("#e5e7eb")
            if j == 0:
                cell.set_text_props(ha="left")
                cell._loc = "left"
            if i % 2 == 0:
                cell.set_facecolor(_ALT_ROW)

        # Bold the last row (Total)
        if i == len(data):
            for j in range(n_cols):
                table[i, j].set_text_props(fontweight="bold")

    return table


def _section_title(fig, x, y, text):
    """Place a bold section title at (x, y); return y after the title."""
    fig.text(x, y, text, fontsize=9.5, fontweight="bold", color="#111827", va="top")
    return y - _TITLE_H


def _draw_recovery_curve(ax, data):
    """Draw the recovery curve on a matplotlib axes."""
    lib_size = data["library_size"]
    fold_sampling = data["fold_sampling"]
    expected_coverage = data["expected_coverage"]
    resynth = data.get("resynth")
    target_coverage = data.get("target_coverage", 0.90)

    if expected_coverage is None or resynth is None:
        ax.text(0.5, 0.5, "No simulation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=_GREY)
        ax.set_axis_off()
        return

    dropout_n = resynth["dropout_count"]
    r1_cov_end = resynth["round1_coverage"]
    r1_max = resynth["round1_wells"]
    r2_cov_end = resynth["round2_coverage"]
    r2_max = resynth["round2_wells"]

    # Calibrate rates from simulation endpoints
    pool_rate = -r1_max / (lib_size * math.log(1 - min(r1_cov_end, 0.999)))
    r2_rate = (-r2_max / (dropout_n * math.log(1 - min(r2_cov_end, 0.999)))
               if dropout_n > 0 and r2_cov_end < 0.999 else 1.5)

    max_wells = int(lib_size * fold_sampling * 1.1)
    step = max(1, max_wells // 200)

    # Single-round curve
    sr_wells = list(range(0, max_wells + 1, step))
    sr_cov = [1 - math.exp(-w / (lib_size * pool_rate)) for w in sr_wells]

    # Two-round curve
    r1_wells_list = list(range(0, r1_max + 1, step))
    r1_cov_list = [1 - math.exp(-w / (lib_size * pool_rate)) for w in r1_wells_list]

    r2_wells_list = list(range(0, r2_max + 1, max(1, r2_max // 100)))
    r2_cov_list = [
        r1_cov_end + (1 - r1_cov_end) * (1 - math.exp(-w / (dropout_n * r2_rate)))
        for w in r2_wells_list
    ]

    combined_wells = r1_wells_list + [r1_max + w for w in r2_wells_list[1:]]
    combined_cov = r1_cov_list + r2_cov_list[1:]

    ax.plot(sr_wells, sr_cov, color=_BLUE, linewidth=1.5, label="Single-round")
    ax.plot(combined_wells, combined_cov, color=_GREEN, linewidth=1.5, label="With resynthesis")
    ax.axhline(target_coverage, color="red", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_wells)
    ax.set_xlabel("Wells sorted", fontsize=7)
    ax.set_ylabel("Coverage", fontsize=7, labelpad=3)
    ax.set_title("Recovery Curve", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=6, loc="lower right")
    ax.set_facecolor("none")


def generate_estimate_report(data):
    """Generate a PDF cost estimate report.

    Args:
        data: Dict with all computed estimate values. Expected keys:
            library_size, seq_length, fold_sampling, fold_sampling_auto,
            expected_coverage, target_coverage, skew, sorting_efficiency,
            synthesis_cost, synthesis_method_name, cloning_cost, sorting_cost,
            barcoding_cost, sequencing_cost, hitpicking_cost, usortm_total,
            resynth (dict or None), r1_total, resynth_synthesis, r2_total,
            two_round_total, compare, trad_total, trad_synthesis, trad_cloning,
            trad_sequencing, sdm_compare, sdm_total, sdm_primers, sdm_kit,
            sdm_transformation, sdm_consumables, sdm_sequencing,
            sdm_include_hifi, pricing_dates, sr_timeline, two_round_days

    Returns:
        Path to the generated PDF file.
    """
    lib_size = data["library_size"]
    seq_length = data["seq_length"]
    output_path = Path(f"usortm_estimate_{lib_size}x{seq_length}bp.pdf")

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        LEFT = 0.06
        RIGHT = 0.94
        WIDTH = RIGHT - LEFT

        y = 0.968

        # ── Header ──
        fig.text(LEFT, y, "uSort-M Cost Estimate",
                 fontsize=15, fontweight="bold", color=_BLUE, va="top")
        fig.text(RIGHT, y, date.today().isoformat(),
                 fontsize=8, color=_GREY, ha="right", va="top")
        y -= 0.034

        fig.text(LEFT, y, f"Library: {lib_size:,} variants × {seq_length} bp",
                 fontsize=9, color="#111827", va="top")
        y -= 0.023

        # All params on one line, including synthesis method
        params = []
        fold_str = f"{data['fold_sampling']}×"
        if data.get("fold_sampling_auto"):
            fold_str += f" (auto, {data['target_coverage']:.0%})"
        params.append(f"Fold: {fold_str}")
        params.append(f"Skew: {data['skew']}×")
        params.append(f"Efficiency: {data['sorting_efficiency']:.0%}")
        if data.get("expected_coverage") is not None:
            params.append(f"Coverage: {data['expected_coverage']:.0%}")
        if data.get("synthesis_method_name"):
            params.append(f"Synthesis: {data['synthesis_method_name']}")
        fig.text(LEFT, y, "  ·  ".join(params), fontsize=7.5, color=_GREY, va="top")
        y -= 0.018

        # Thin separator line
        fig.add_artist(Line2D(
            [LEFT, RIGHT], [y, y],
            transform=fig.transFigure, color="#d1d5db", linewidth=0.6,
        ))
        y -= 0.016

        # ── Table 1: uSort-M Cost Breakdown ──
        y = _section_title(fig, LEFT, y, "uSort-M Cost Breakdown")

        def _per_seq(v):
            return f"${v / lib_size:,.2f}" if lib_size > 0 else "—"

        usortm_total = data["usortm_total"]
        two_round_total = data.get("two_round_total")
        show_two_round = (
            data.get("resynth") is not None
            and two_round_total is not None
            and two_round_total < usortm_total
        )

        cost_data = [
            ["Synthesis", _fmt_cost(data["synthesis_cost"]), _per_seq(data["synthesis_cost"])],
            ["Cloning", _fmt_cost(data["cloning_cost"]), _per_seq(data["cloning_cost"])],
            ["Sorting", _fmt_cost(data["sorting_cost"]), _per_seq(data["sorting_cost"])],
            ["Barcoding", _fmt_cost(data["barcoding_cost"]), _per_seq(data["barcoding_cost"])],
            ["Sequencing", _fmt_cost(data["sequencing_cost"]), _per_seq(data["sequencing_cost"])],
            ["Hit-picking", _fmt_cost(data["hitpicking_cost"]), _per_seq(data["hitpicking_cost"])],
        ]
        if show_two_round:
            cost_data.append(["Total (single-round)", _fmt_cost(usortm_total), _per_seq(usortm_total)])
            cost_data.append(["Total (with resynthesis)", _fmt_cost(two_round_total), _per_seq(two_round_total)])
        else:
            cost_data.append(["Total", _fmt_cost(usortm_total), _per_seq(usortm_total)])

        best_usortm_total = two_round_total if show_two_round else usortm_total
        t1_h = _table_height(len(cost_data))
        ax1 = fig.add_axes([LEFT, y - t1_h, 0.52, t1_h])
        _add_table(ax1, ["Step", "Total", "Per Sequence"], cost_data,
                   col_widths=[0.50, 0.25, 0.25])
        y -= t1_h + _SEC_GAP

        # ── Table 2: Strategy Comparison ──
        resynth = data.get("resynth")
        if resynth is not None:
            r1_cov = resynth["round1_coverage"]
            r2_cov = resynth["round2_coverage"]
            dropout_n = resynth["dropout_count"]

            y = _section_title(fig, LEFT, y, "Strategy Comparison")

            strat_data = [
                ["Total cost", _fmt_cost(data["usortm_total"]), _fmt_cost(data["two_round_total"])],
                ["  R1 cost", "—", f"{_fmt_cost(data['r1_total'])} ({data.get('round1_fold', 3.0)}× → {r1_cov:.0%})"],
                ["  Resynthesize", "—", f"{_fmt_cost(data['resynth_synthesis'])} ({dropout_n:,} dropouts)"],
                ["  R2 cost", "—", f"{_fmt_cost(data['r2_total'] - data['resynth_synthesis'])} ({resynth['round2_fold']}× → {r2_cov:.0%})"],
                ["Coverage", f"{data['expected_coverage']:.0%}", f"{resynth['total_coverage']:.0%}"],
                ["Wells", f"{int(lib_size * data['fold_sampling']):,}", f"{resynth['total_wells']:,}"],
                ["Working days", str(data.get("sr_timeline_days", "—")), str(data.get("two_round_days", "—"))],
                ["Cost/variant", f"${data['usortm_total']/lib_size:.2f}", f"${data['two_round_total']/lib_size:.2f}"],
            ]
            t2_h = _table_height(len(strat_data))

            # Strategy table (left) + recovery curve (right).
            # CURVE_X leaves enough room for the curve's rotated ylabel.
            TABLE_W = 0.44
            CURVE_X = LEFT + TABLE_W + 0.10
            CURVE_W = RIGHT - CURVE_X

            ax2 = fig.add_axes([LEFT, y - t2_h, TABLE_W, t2_h])
            _add_table(ax2, ["", "Single-round", "2-round"], strat_data,
                       col_widths=[0.28, 0.30, 0.42])

            ax_plot = fig.add_axes([CURVE_X, y - t2_h + 0.005, CURVE_W, t2_h - 0.010])
            _draw_recovery_curve(ax_plot, data)

            y -= t2_h + 0.010

            # Savings note
            diff = data["usortm_total"] - data["two_round_total"]
            if diff > 0:
                fig.text(LEFT, y,
                         f"Resynthesis saves {_fmt_cost(diff)} ({diff/data['usortm_total']:.0%} less) "
                         f"with fewer wells ({resynth['total_wells']:,} vs {int(lib_size * data['fold_sampling']):,})",
                         fontsize=7, color="#16a34a", fontweight="bold", va="top")
            else:
                fig.text(LEFT, y,
                         f"Single-round is {_fmt_cost(-diff)} cheaper than resynthesis for this library",
                         fontsize=7, color=_GREY, va="top")
            y -= 0.022

        # ── Table 3: Alternative Methods ──
        compare = data.get("compare", False)
        sdm_compare = data.get("sdm_compare", False)

        if compare or sdm_compare:
            y = _section_title(fig, LEFT, y, "Alternative Methods")

            col_labels = ["Step"]
            if compare:
                col_labels.extend(["Direct Synthesis", "Per Sequence"])
            if sdm_compare:
                col_labels.extend(["SDM", "Per Sequence"])

            rows = []

            def _per_seq(v):
                return f"${v / lib_size:,.2f}" if isinstance(v, (int, float)) and lib_size > 0 else ""

            def _row(label, trad_val=None, sdm_val=None, trad_num=None, sdm_num=None):
                r = [label]
                if compare:
                    r.append(trad_val or "")
                    r.append(_per_seq(trad_num))
                if sdm_compare:
                    r.append(sdm_val or "")
                    r.append(_per_seq(sdm_num))
                rows.append(r)

            _row("Synthesis",
                 _fmt_cost(data.get("trad_synthesis")) if compare else None,
                 _fmt_cost(data.get("sdm_primers")) if sdm_compare else None,
                 trad_num=data.get("trad_synthesis") if compare else None,
                 sdm_num=data.get("sdm_primers") if sdm_compare else None)
            sdm_cloning_combined = (
                (data.get("sdm_kit") or 0)
                + (data.get("sdm_transformation") or 0)
                + (data.get("sdm_consumables") or 0)
            ) if sdm_compare else None
            _row("Cloning",
                 _fmt_cost(data.get("trad_cloning")) if compare else None,
                 _fmt_cost(sdm_cloning_combined) if sdm_compare else None,
                 trad_num=data.get("trad_cloning") if compare else None,
                 sdm_num=sdm_cloning_combined if sdm_compare else None)
            _row("Barcoding",
                 _fmt_cost(data.get("trad_barcoding")) if compare else None,
                 _fmt_cost(data.get("sdm_barcoding")) if sdm_compare else None,
                 trad_num=data.get("trad_barcoding") if compare else None,
                 sdm_num=data.get("sdm_barcoding") if sdm_compare else None)
            _row("Sequencing",
                 _fmt_cost(data.get("trad_sequencing")) if compare else None,
                 _fmt_cost(data.get("sdm_sequencing")) if sdm_compare else None,
                 trad_num=data.get("trad_sequencing") if compare else None,
                 sdm_num=data.get("sdm_sequencing") if sdm_compare else None)
            _row("Total",
                 _fmt_cost(data.get("trad_total")) if compare else None,
                 _fmt_cost(data.get("sdm_total")) if sdm_compare else None,
                 trad_num=data.get("trad_total") if compare else None,
                 sdm_num=data.get("sdm_total") if sdm_compare else None)

            vs_label = "vs uSort-M"
            if best_usortm_total < usortm_total:
                vs_label = "vs uSort-M (with resynthesis)"
            savings_row = [vs_label]
            if compare:
                fold = data["trad_total"] / best_usortm_total
                savings_row.extend([f"{fold:.1f}× savings", ""])
            if sdm_compare:
                fold = data["sdm_total"] / best_usortm_total
                savings_row.extend([f"{fold:.1f}× savings", ""])
            rows.append(savings_row)

            n_cols = len(col_labels)

            # Column widths: cap "Step" at 30%, distribute rest equally
            raw_widths = _col_widths_from_content(col_labels, rows)
            if n_cols > 1:
                step_frac = min(raw_widths[0], 0.30)
                val_frac = (1.0 - step_frac) / (n_cols - 1)
                col_widths = [step_frac] + [val_frac] * (n_cols - 1)
            else:
                col_widths = raw_widths

            # Table width scales with number of value columns, not full-page
            table_w = min(WIDTH, 0.30 + 0.24 * (n_cols - 1))
            t3_h = _table_height(len(rows))
            ax3 = fig.add_axes([LEFT, y - t3_h, table_w, t3_h])
            _add_table(ax3, col_labels, rows, col_widths=col_widths)

        # ── Footer ──
        pricing_dates = data.get("pricing_dates", "")
        fig.text(0.05, 0.025,
                 f"Pricing: {pricing_dates}  ·  Generated by usortm estimate",
                 fontsize=6, color="#9ca3af")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path
