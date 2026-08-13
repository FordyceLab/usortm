"""Plots for measured library skew.

Two views of the same measurement: a rank-abundance plot, which shows the
shape of the distribution and where variants fall below detection, and a
cumulative-abundance (Lorenz) curve, which makes unevenness legible at a
glance against the diagonal a perfectly even library would trace.

Bokeh lives in the ``viz`` extra, so every entry point here degrades to
None rather than raising when it is not installed.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

_BRAND = "#4096E3"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"
_UNDETECTED = "#e07b39"

# Zero counts have no place on a log axis; draw them on a floor line below
# the smallest real count so they stay visible and obviously distinct.
_ZERO_FLOOR = 0.4

__all__ = [
    "bokeh_available",
    "make_abundance_histogram_figure",
    "make_rank_abundance_figure",
    "make_cumulative_figure",
    "write_skew_html",
]


def bokeh_available() -> bool:
    """Whether Bokeh is importable."""
    try:
        import bokeh  # noqa: F401
        return True
    except ImportError:
        return False


def _style_figure(fig):
    """Apply the uSort-M dashboard look to a Bokeh figure."""
    fig.background_fill_color = None
    fig.border_fill_color = None
    fig.outline_line_color = None
    fig.toolbar_location = None
    fig.axis.axis_label_text_font_size = "12px"
    fig.axis.major_label_text_font_size = "11px"
    fig.axis.axis_label_text_color = _MUTED
    fig.axis.major_label_text_color = _MUTED
    fig.axis.axis_line_color = _GRID
    fig.grid.grid_line_color = _GRID
    fig.grid.grid_line_alpha = 0.5


def make_abundance_histogram_figure(counts, stats, n_bins: int = 24):
    """Histogram of log10 abundance, the standard uniformity view.

    A uniform library is a tight bell; skew shows up as width.  Two curves
    are drawn over the bars:

    * **fitted, with counting noise** — what the model expects to *observe*.
      It should track the bars; where it does not, the log-normal
      assumption is a poor description of this library.
    * **underlying abundance** — the same fit with Poisson noise removed.
      The gap between the two curves is the counting noise, i.e. exactly
      the width that would be mistaken for skew by reading the histogram
      at face value.

    Variants with zero reads cannot go on a log axis and are reported in
    the subtitle instead.

    Args:
        counts: VariantCounts.
        stats: SkewStats from `measure_skew`.
        n_bins: Histogram bins across the observed range.

    Returns:
        A Bokeh figure, or None if Bokeh is unavailable.
    """
    try:
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import ColumnDataSource, HoverTool, Label, Span
    except ImportError:
        logger.info("Bokeh not installed; skipping abundance histogram")
        return None

    from usortm.qc.skew import log10_histogram

    try:
        edges, observed, predicted, underlying = log10_histogram(
            counts, stats, n_bins=n_bins
        )
    except ValueError:
        return None

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    source = ColumnDataSource(data=dict(
        center=centers, top=observed, width=widths * 0.96,
        reads_lo=10 ** edges[:-1], reads_hi=10 ** edges[1:],
        variants=observed,
    ))

    fig = bokeh_figure(
        width=460, height=300,
        x_axis_label="Reads per variant (log₁₀)",
        y_axis_label="Variants",
        sizing_mode="stretch_width",
    )
    fig.vbar(
        x="center", top="top", width="width", source=source,
        fill_color=_BRAND, fill_alpha=0.35, line_color=_BRAND, line_width=0.5,
    )
    fig.line(centers, predicted, line_color=_MUTED, line_width=2,
             legend_label="Fitted + counting noise")
    fig.line(centers, underlying, line_color=_BRAND, line_width=2.5,
             line_dash="dashed", legend_label="Underlying abundance")

    # Mean of the underlying distribution, for a sense of scale.
    mean_log10 = stats.mu_log / np.log(10.0)
    if edges[0] < mean_log10 < edges[-1]:
        fig.add_layout(Span(
            location=mean_log10, dimension="height",
            line_color=_MUTED, line_width=1, line_dash="dotted", line_alpha=0.7,
        ))

    sd_log10 = stats.sigma_log / np.log(10.0)
    fig.add_layout(Label(
        x=edges[0] + 0.04 * (edges[-1] - edges[0]),
        y=max(observed.max(), predicted.max()) * 0.92,
        text=f"width σ = {sd_log10:.2f} log₁₀  →  {stats.q90_q10_corrected:.1f}× Q90/Q10",
        text_font_size="10px", text_color=_MUTED,
    ))

    fig.add_tools(HoverTool(
        tooltips=[("Reads", "@reads_lo{0.0}–@reads_hi{0.0}"),
                  ("Variants", "@variants")],
        mode="vline",
    ))
    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "10px"
    fig.legend.background_fill_alpha = 0.0
    fig.legend.border_line_color = None
    fig.y_range.start = 0
    _style_figure(fig)
    return fig


def make_rank_abundance_figure(counts, stats):
    """Rank-abundance plot: observed counts, fitted curve, detection floor.

    Args:
        counts: VariantCounts.
        stats: SkewStats from `measure_skew`.

    Returns:
        A Bokeh figure, or None if Bokeh is unavailable.
    """
    try:
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import ColumnDataSource, HoverTool, Label, Span
    except ImportError:
        logger.info("Bokeh not installed; skipping rank-abundance plot")
        return None

    names = np.array(counts.names)
    values = counts.as_array()
    order = np.argsort(-values)
    values, names = values[order], names[order]
    lib_size = len(values)
    ranks = np.arange(1, lib_size + 1)

    detected = values > 0
    plot_y = np.where(detected, values, _ZERO_FLOOR)

    source = ColumnDataSource(data=dict(
        rank=ranks,
        reads=values,
        plot_y=plot_y,
        name=names,
        color=[_BRAND if d else _UNDETECTED for d in detected],
        status=["detected" if d else "undetected" for d in detected],
    ))

    fig = bokeh_figure(
        width=460, height=300,
        x_axis_label="Variant rank (most to least abundant)",
        y_axis_label="Reads",
        y_axis_type="log",
        sizing_mode="stretch_width",
    )
    fig.scatter(
        x="rank", y="plot_y", source=source,
        size=5, fill_color="color", line_color=None, fill_alpha=0.7,
    )

    # Fitted log-normal quantile curve, drawn over the variants the model
    # treats as present.
    n_present = max(1, int(round(lib_size * (1.0 - stats.dropout_fraction))))
    n_assigned = values.sum()
    if n_assigned > 0 and stats.sigma_log > 0:
        mean_present = n_assigned / n_present
        mu = np.log(mean_present) - 0.5 * stats.sigma_log**2
        curve_ranks = np.arange(1, n_present + 1)
        upper_tail = 1.0 - (curve_ranks - 0.5) / n_present
        fitted = np.exp(mu + stats.sigma_log * norm.ppf(upper_tail))
        fig.line(
            curve_ranks, np.clip(fitted, _ZERO_FLOOR, None),
            line_color=_MUTED, line_width=2, line_dash="dashed", line_alpha=0.8,
        )

    fig.add_layout(Span(
        location=1, dimension="width",
        line_color=_MUTED, line_width=1, line_dash="dotted", line_alpha=0.6,
    ))
    fig.add_layout(Label(
        x=lib_size * 0.02, y=1.15,
        text="1 read", text_font_size="10px", text_color=_MUTED, text_alpha=0.8,
    ))

    fig.add_tools(HoverTool(tooltips=[
        ("Variant", "@name"), ("Rank", "@rank"),
        ("Reads", "@reads{0}"), ("", "@status"),
    ]))
    _style_figure(fig)
    return fig


def make_cumulative_figure(counts):
    """Lorenz curve of read share against an even-library diagonal.

    Args:
        counts: VariantCounts.

    Returns:
        A Bokeh figure, or None if Bokeh is unavailable.
    """
    try:
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import HoverTool
    except ImportError:
        return None

    values = np.sort(counts.as_array())[::-1]
    total = values.sum()
    if total <= 0:
        return None

    lib_frac = np.arange(1, len(values) + 1) / len(values)
    read_frac = np.cumsum(values) / total

    fig = bokeh_figure(
        width=460, height=300,
        x_axis_label="Fraction of library (most abundant first)",
        y_axis_label="Fraction of reads",
        sizing_mode="stretch_width",
    )
    fig.line([0, 1], [0, 1], line_color=_MUTED, line_width=1.5,
             line_dash="dashed", line_alpha=0.7, legend_label="Perfectly even")
    line = fig.line(lib_frac, read_frac, line_color=_BRAND, line_width=2.5,
                    legend_label="Measured")
    fig.varea(x=lib_frac, y1=lib_frac, y2=read_frac,
              fill_color=_BRAND, fill_alpha=0.12)

    fig.add_tools(HoverTool(
        renderers=[line],
        tooltips=[("Library", "@x{0.0%}"), ("Reads", "@y{0.0%}")],
        mode="vline",
    ))
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10px"
    fig.legend.background_fill_alpha = 0.0
    fig.legend.border_line_color = None
    _style_figure(fig)
    return fig


def _summary_rows(profile) -> list:
    """Headline numbers for the HTML summary table."""
    stats, rec, counts = profile.stats, profile.recommendation, profile.counts
    ci_low, ci_high = stats.q90_q10_ci
    ci_text = (
        f" (95% CI {ci_low:.1f}–{ci_high:.1f})"
        if np.isfinite(ci_low) and np.isfinite(ci_high) else ""
    )
    observed = (
        f"{stats.q90_q10_observed:.1f}×"
        if stats.q90_q10_observed is not None else "undefined"
    )
    return [
        ("Library size", f"{counts.library_size:,} variants"),
        ("Reads assigned", f"{counts.assigned_reads:,} of {counts.total_reads:,}"),
        ("Mean depth", f"{stats.mean_depth:.1f} reads/variant"),
        ("Skew, raw Q90/Q10", f"{observed} (inflated by counting noise)"),
        ("Skew, corrected", f"{stats.q90_q10_corrected:.1f}×{ci_text}"),
        ("Effective library size", f"{stats.effective_library_size:,.0f} variants"),
        ("Gini coefficient", f"{stats.gini:.2f}"),
        ("Undetected variants", f"{stats.n_undetected:,}"),
        ("Estimated dropout", f"{stats.dropout_fraction:.1%} of the library"),
        ("Coverage ceiling", f"{stats.coverage_ceiling:.1%}"),
        ("Recommended sorting", f"{rec.fold_sampling:g}× fold-sampling"),
        ("Wells to sort", f"{rec.n_wells:,} ({rec.n_plates} × 384-well plates)"),
        ("Predicted coverage", f"{rec.expected_coverage:.1%} of the full library"),
    ]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>uSort-M library skew &mdash; {title}</title>
{bokeh_css}
{bokeh_js}
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
         Arial, sans-serif; margin: 0; padding: 2rem 1.5rem; background: #f8fafc;
         color: #1f2937; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 0.25rem; }}
  h1 span {{ color: {brand}; }}
  .sub {{ color: #6b7280; font-size: 0.85rem; margin: 0 0 1.5rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem; margin-bottom: 1rem; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
           padding: 1rem 1.1rem; }}
  .card h3 {{ margin: 0 0 0.6rem; font-size: 0.95rem; }}
  .card.wide {{ margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  td {{ padding: 0.4rem 0.2rem; border-bottom: 1px solid #f1f5f9; }}
  td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .note {{ color: #6b7280; font-size: 0.8rem; margin: 0.6rem 0 0; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f172a; color: #e2e8f0; }}
    .card {{ background: #1e293b; border-color: #334155; }}
    td {{ border-bottom-color: #334155; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1><span>uSort-M</span> library skew</h1>
  <p class="sub">{title}</p>
  <div class="card wide">
    <h3>Abundance distribution</h3>
    {histogram_div}
    <p class="note">A uniform library is a tight bell; skew is width. The solid
    grey curve is the fit <em>including</em> counting noise and should track the
    bars. The dashed curve is the underlying abundance with that noise removed
    &mdash; the gap between them is the spread that reading the histogram at face
    value would mistake for skew. {undetected_note}</p>
  </div>
  <div class="cards">
    <div class="card">
      <h3>Rank abundance</h3>
      {rank_div}
      <p class="note">Dashed line: fitted log-normal. Orange points saw no
      reads &mdash; at this depth that means undetected, not necessarily absent.</p>
    </div>
    <div class="card">
      <h3>Cumulative abundance</h3>
      {cumulative_div}
      <p class="note">Distance from the diagonal is the skew. Gini = {gini:.2f}.</p>
    </div>
  </div>
  <div class="card">
    <h3>Summary</h3>
    <table>{rows}</table>
    <p class="note">{caveat}</p>
  </div>
</div>
{bokeh_script}
</body>
</html>
"""


def write_skew_html(profile, output_path, title="library") -> bool:
    """Write a standalone HTML summary of a library profile.

    Args:
        profile: LibraryProfile from `usortm.qc.profile_library`.
        output_path: Destination .html path.
        title: Label shown under the heading.

    Returns:
        True if the file was written, False if Bokeh is unavailable.
    """
    if not bokeh_available():
        logger.info("Bokeh not installed; skipping HTML report")
        return False

    from bokeh.embed import components
    from bokeh.resources import INLINE

    named_figures = [
        (key, fig) for key, fig in (
            ("histogram", make_abundance_histogram_figure(
                profile.counts, profile.stats)),
            ("rank", make_rank_abundance_figure(profile.counts, profile.stats)),
            ("cumulative", make_cumulative_figure(profile.counts)),
        ) if fig is not None
    ]
    if not named_figures:
        return False

    script, divs = components([fig for _, fig in named_figures])
    div_map = dict(zip([key for key, _ in named_figures], divs))

    rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in _summary_rows(profile)
    )

    stats = profile.stats
    if not stats.depth_sufficient:
        caveat = (
            f"Only {stats.mean_depth:.1f} reads per variant. Counting noise "
            "dominates at this depth, so treat the corrected skew as a rough "
            "estimate and the sorting depth as a lower bound."
        )
    else:
        caveat = (
            "Corrected skew deconvolves Poisson counting noise from the raw "
            "count spread; the raw ratio above it is always the more "
            "pessimistic of the two."
        )

    html = _HTML_TEMPLATE.format(
        title=title,
        brand=_BRAND,
        bokeh_css=INLINE.render_css(),
        bokeh_js=INLINE.render_js(),
        bokeh_script=script,
        histogram_div=div_map.get("histogram", ""),
        rank_div=div_map.get("rank", ""),
        cumulative_div=div_map.get("cumulative", ""),
        gini=stats.gini,
        rows=rows,
        caveat=caveat,
        undetected_note=(
            f"{stats.n_undetected:,} variant(s) with zero reads are off the log "
            "axis and not shown."
            if stats.n_undetected else ""
        ),
    )

    with open(output_path, "w") as fh:
        fh.write(html)
    return True
