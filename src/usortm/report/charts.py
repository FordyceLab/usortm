"""The summary page's figures.

Each returns HTML.  Marks are drawn as inline SVG and every label is HTML at
its own size: text inside an SVG scales with the SVG, which at the widths these
sit at lands around seven pixels and goes soft.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH)

#: Read depth at which a well is drawn at the top of the colour ramp.  Twice
#: the tier-A threshold, so the tiers land at recognisable heights on it.
DEPTH_CEILING = 200

TIER_READS = {"A": 100, "B": 50, "C": 20}


def cmap_hex(t: float) -> str:
    """Sample the white-yellow-green ramp the plate maps use, at *t* in [0, 1].

    The same stops as ``get_custom_cmap()``, interpolated here rather than
    through matplotlib, which is an optional extra and cannot be required to
    render a report.  The duplication is checked against the original by
    test_summary_page, since a ramp copied by hand is a ramp that can drift.
    """
    t = max(0.0, min(1.0, t))

    def lerp(stops, x):
        for i in range(len(stops) - 1):
            x0, v0 = stops[i]
            x1, v1 = stops[i + 1]
            if x0 <= x <= x1:
                f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                return v0 + f * (v1 - v0)
        return stops[-1][1]

    r = lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 1.0), (0.40, 0.5), (1.0, 0.0)], t)
    g = lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 0.95), (0.40, 0.98), (1.0, 0.39)], t)
    b = lerp([(0.0, 1.0), (0.05, 1.0), (0.20, 0.35), (0.40, 0.6), (1.0, 0.0)], t)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def depth_colour(reads: float) -> str:
    """A well's fill for *reads*; white when it returned none."""
    if not reads:
        return "#ffffff"
    return cmap_hex(min(reads / DEPTH_CEILING, 1.0))


def colorbar() -> str:
    """The depth ramp, with each tier's tick at the height it sits at.

    Spacing the ticks evenly would point each label at a colour that is not
    its own: on a 0-200 ramp, tier C is at a tenth of the way up, not a
    quarter.
    """
    ramp = ", ".join(
        f"{cmap_hex(i / 20)} {i * 5}%" for i in range(21)
    )
    ticks = "".join(
        f'<span style="bottom:{min(d / DEPTH_CEILING, 1) * 100:.0f}%">{lab}</span>'
        for d, lab in [(0, "0"), (TIER_READS["C"], f"C {TIER_READS['C']}"),
                       (TIER_READS["B"], f"B {TIER_READS['B']}"),
                       (TIER_READS["A"], f"A {TIER_READS['A']}"),
                       (DEPTH_CEILING, f"&ge;{DEPTH_CEILING}")]
    )
    return (f'<div class="cbar"><div class="ramp" style="background:'
            f'linear-gradient(to top, {ramp})"></div>'
            f'<div class="ticks">{ticks}</div></div>')


def colorbar_h() -> str:
    """The same ramp laid along the page, for plates that share one.

    Two maps drawn on one scale need one key: a bar beside each invites the
    reading that they are scaled separately.
    """
    ramp = ", ".join(f"{cmap_hex(i / 20)} {i * 5}%" for i in range(21))
    ticks = "".join(
        f'<span style="left:{min(d / DEPTH_CEILING, 1) * 100:.0f}%">{lab}</span>'
        for d, lab in [(0, "0"), (TIER_READS["C"], f"C {TIER_READS['C']}"),
                       (TIER_READS["B"], f"B {TIER_READS['B']}"),
                       (TIER_READS["A"], f"A {TIER_READS['A']}"),
                       (DEPTH_CEILING, f"&ge;{DEPTH_CEILING}")]
    )
    return (f'<div class="cbarh"><div class="ramp" style="background:'
            f'linear-gradient(to right, {ramp})"></div>'
            f'<div class="ticks">{ticks}</div>'
            f'<div class="cblab">reads per well</div></div>')


def bar(pct: float, tone: str = "") -> str:
    """A share drawn as a bar as well as a number."""
    cls = f"bar {tone}" if tone else "bar"
    return f'<span class="{cls}"><i style="width:{max(0.0, min(100.0, pct)):.1f}%"></i></span>'


def _bars_svg(counts: Sequence[int], colours: Optional[Sequence[str]] = None,
              fill: str = "var(--series-1)") -> str:
    """A histogram's bars, drawn against the tallest bin.

    *colours* gives a fill per bin; *fill* is the one every bin takes when it
    is not given.  Both are CSS colours, so a token follows the theme.
    """
    if not counts:
        return ""
    peak = max(counts) or 1
    width = 640 / len(counts)
    out = []
    for i, c in enumerate(counts):
        if not c:
            continue
        h = round(c / peak * 94)
        if h < 1:
            h = 1
        paint = colours[i] if colours else fill
        out.append(
            f'<rect x="{i * width:.2f}" y="{96 - h}" '
            f'width="{max(width - 1, 0.5):.2f}" height="{h}" fill="{paint}"></rect>'
        )
    out.append('<line x1="0" y1="95.5" x2="640" y2="95.5" '
               'stroke="var(--rule)" stroke-width="1"></line>')
    return (f'<svg viewBox="0 0 640 96" preserveAspectRatio="none" '
            f'class="chart">{"".join(out)}</svg>')


def read_length_chart(hist: dict, run_reads: int) -> str:
    """Read length, with what the histogram covers stated.

    A run demultiplexed in segments produces one histogram per segment, and
    those are only additive when the segments chose the same bin size.  When
    they did not, one segment's stands for the run, so the count it covers is
    given rather than left to look like the whole.
    """
    counts = hist.get("counts") or []
    if not counts:
        return ""
    bin_size = hist.get("bin_size") or 1
    top = bin_size * len(counts)
    n = hist.get("n_reads") or 0
    over = hist.get("n_over") or 0
    parts = [f"median {hist.get('median', 0):,} bp", f"{n:,} reads"]
    if over:
        parts.append(f"{over:,} longer, to {hist.get('longest', 0):,} bp")
    if run_reads and n and n < run_reads * 0.95:
        parts.append(f"one segment of {run_reads:,}")
    return (
        f'{_bars_svg(counts)}'
        f'<div class="axis"><span>0</span><span>{top // 2:,}</span>'
        f'<span>{top:,}{"+" if over else ""} bp</span></div>'
        f'<div class="hint">{" &middot; ".join(parts)}</div>'
    )


def read_depth_chart(depths: Sequence[int]) -> str:
    """Per-well depth, in one green from the plate maps' ramp.

    Filling each bin with the ramp colour for its own depth hid the left of
    the distribution: the ramp opens at white, so the bins holding wells that
    returned few reads were painted the colour of the page behind them, and
    only the deep end of the histogram could be seen.  Those bins are the ones
    a reader is checking for.  A single fill keeps the tie to the plate maps
    and leaves the depth to the axis, which already states it.
    """
    depths = [d for d in depths if d and d > 0]
    if not depths:
        return ""
    n_bins = 40
    counts = [0] * n_bins
    for d in depths:
        counts[min(int(d / DEPTH_CEILING * n_bins), n_bins - 1)] += 1
    ordered = sorted(depths)
    median = ordered[len(ordered) // 2]
    n20 = sum(1 for d in depths if d >= TIER_READS["C"])
    n100 = sum(1 for d in depths if d >= TIER_READS["A"])
    return (
        f'{_bars_svg(counts, fill="var(--good)")}'
        f'<div class="axis"><span>0</span><span>{DEPTH_CEILING // 2}</span>'
        f'<span>&ge;{DEPTH_CEILING} reads</span></div>'
        f'<div class="hint">median {median:,} reads &middot; {n20:,} wells '
        f'&ge;{TIER_READS["C"]} &middot; {n100:,} &ge;{TIER_READS["A"]}</div>'
    )


def recovery_chart(curves: dict, info: str = "") -> str:
    """Simulated recovery against fold sampling, with this run marked.

    *curves* carries ``fold_samplings`` and, under ``measured``, the mean and
    standard deviation at each; ``sampling`` and ``observed`` place the run's
    own point.  *info* is the conditions the curve was computed under, folded
    into the key rather than set beside the figure.
    """
    folds = curves.get("fold_samplings") or []
    if not folds:
        return ""
    x_max = max(folds)
    xs = [0.0] + list(folds)

    def px(v):
        return 100 * v / x_max

    def py(v):
        return 100 - v

    # --grid, not --rule: the rules that separate content sit on top of the
    # page, and at this density they read as a mesh the curve has to compete
    # with.
    out = []
    for pct in (25, 50, 75):
        out.append(f'<line x1="0" y1="{py(pct):.2f}" x2="100" y2="{py(pct):.2f}" '
                   f'stroke="var(--grid)" stroke-width="1" '
                   f'vector-effect="non-scaling-stroke"></line>')
    for v in range(2, int(x_max) + 1, 2):
        out.append(f'<line x1="{px(v):.2f}" y1="0" x2="{px(v):.2f}" y2="100" '
                   f'stroke="var(--grid)" stroke-width="1" '
                   f'vector-effect="non-scaling-stroke"></line>')

    def series(key, colour, dashed, band):
        c = curves.get(key) or {}
        means = [0.0] + list(c.get("means") or [])
        stds = [0.0] + list(c.get("stds") or [])
        if len(means) < 2:
            return ""
        marks = []
        if band and any(stds):
            pts = ([f"{px(x):.2f},{py(min(m + s, 100)):.2f}"
                    for x, m, s in zip(xs, means, stds)]
                   + [f"{px(x):.2f},{py(max(m - s, 0)):.2f}"
                      for x, m, s in reversed(list(zip(xs, means, stds)))])
            marks.append(f'<polygon points="{" ".join(pts)}" fill="{colour}" '
                         f'fill-opacity="0.18"></polygon>')
        line = " ".join(f"{px(x):.2f},{py(m):.2f}" for x, m in zip(xs, means))
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        marks.append(f'<polyline points="{line}" fill="none" stroke="{colour}" '
                     f'stroke-width="2" stroke-linejoin="round"{dash} '
                     f'vector-effect="non-scaling-stroke"></polyline>')
        return "".join(marks)

    out.append(series("measured", "var(--series-1)", False, True))

    ox = curves.get("sampling")
    oy = curves.get("observed")
    marker = ""
    if ox is not None and oy is not None and ox <= x_max:
        # The point alone.  A line dropped to the axis restated a coordinate
        # the gridlines already give, in the one colour on the figure that
        # means something is wrong.
        # An HTML dot: the SVG is stretched to its box, which would draw a
        # circle as an ellipse.
        marker = (f'<i class="obs" style="left:{px(ox):.2f}%;'
                  f'top:{py(oy):.2f}%"></i>')
    out.append('<line x1="0" y1="100" x2="100" y2="100" '
               'stroke="var(--text-muted)" stroke-width="1" '
               'vector-effect="non-scaling-stroke"></line>')

    ticks = "".join(f'<span style="left:{px(v):.2f}%">{v}</span>'
                    for v in range(0, int(x_max) + 1, 2))
    return (
        f'<div class="plotwrap">'
        f'<div class="yaxis"><span>100%</span><span>75%</span><span>50%</span>'
        f'<span>25%</span><span>0%</span></div>'
        f'<div class="plotbox">'
        f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="sq">'
        f'{"".join(out)}</svg>{marker}'
        f'<div class="plotkey">'
        f'<div class="keyrows">'
        f'<i class="k1"></i><span>simulated</span>'
        f'<i class="k3"></i><span>recovered</span>'
        f'</div>{info}</div>'
        f'<div class="xticks">{ticks}</div></div></div>'
        # Which wells are counted is stated in the note above the plot, so
        # the axis names the quantity and leaves the qualifier there.
        f'<div class="xlab">fold sampling</div>'
    )
