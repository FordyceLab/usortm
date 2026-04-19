import re, string, numpy as np, pandas as pd

import bionumpy as bnp

# `include_groups=False` is only a valid kwarg for DataFrameGroupBy.apply
# on pandas >= 2.2. On older pandas it's forwarded to the user function
# and raises TypeError. Gate it behind a version check.
_PD_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2] if x.isdigit())
_APPLY_KWARGS = {"include_groups": False} if _PD_VERSION >= (2, 2) else {}

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, RadioButtonGroup, CustomJS,
    LinearColorMapper, ColorBar, CustomJSTickFormatter, TapTool
)
from bokeh.layouts import column
from bokeh.embed import file_html
from bokeh.resources import INLINE

def get_custom_cmap():
    cdict = {
        'red':   [(0.0,   1.0, 1.0),   # white
                  (0.05,  1.0, 1.0),   # white
                  (0.20, 1.0, 1.0),   # pale yellow (~75)
                  (0.40,  0.5, 0.5),   # spring green transition
                  (1.0,   0.0, 0.0)],  # deep green

        'green': [(0.0,   1.0, 1.0),   # white
                  (0.05,  1.0, 1.0),   # white
                  (0.20, 0.95, 0.95), # pale yellow (~75)
                  (0.40,  0.98, 0.98), # spring green
                  (1.0,   0.39, 0.39)],# deep green

        'blue':  [(0.0,   1.0, 1.0),   # white
                  (0.05,  1.0, 1.0),   # white
                  (0.20, 0.35, 0.35),   # pale yellow — blue doesn't fully drop (~75)
                  (0.40,  0.6, 0.6),   # spring green
                  (1.0,   0.0, 0.0)],  # deep green
    }

    base_cmap = mcolors.LinearSegmentedColormap('custom_summer', cdict, N=512)

    lut = base_cmap(np.linspace(0, 1, 512))
    lut[:, 3] = 1.0

    custom_cmap = mcolors.ListedColormap(lut, name='custom_summer_opaque')
    return custom_cmap

def plot_length_hist(fastq, ax=None):
    """
    """
    # Parse lengths
    lengths = []
    with open(fastq, 'r') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                lengths.append(len(line.strip()))
    lengths = np.array(lengths)

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    # ax.hist(lengths,bins=50,alpha=0.4,)
    sns.histplot(lengths, bins=50, kde=False, color='C0', ax=ax, element='step')
    ax.set_xlabel('Read Length (bp)')
    ax.set_ylabel('Count')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])

    # Get N reads string:
    if (len(lengths) >= 1000) and (len(lengths) < 1000000):
        n_reads_str = f'N reads = {len(lengths)/1000:.1f}k'
    elif len(lengths) >= 1000000:
        n_reads_str = f'N reads = {len(lengths)/1000000:.1f}M'
    else:
        n_reads_str = f'N reads = {len(lengths)}'

    ax.text(s=n_reads_str,x=0.95,y=0.9,fontdict={'fontsize':10}, ha='right', transform=plt.gca().transAxes)

    # Calculate median and add triangle above it
    median_len = np.median(lengths)
    ax.plot([median_len], [ax.get_ylim()[1]*0.99], marker='v', color='red')
    ax.text(s=f'Median = {int(median_len)} bp',x=0.95,y=0.85,fontdict={'fontsize':9}, color='red', ha='right', transform=plt.gca().transAxes)   

    return ax


def plot_quality_hist(reads, means=None, ax=None):
    """Plot histogram of mean quality scores per read
    """

    # If reads is a filepath, load as a bionumpy array
    if reads.endswith('.fastq') or reads.endswith('.fq'):
        reads = bnp.open(reads).read()

    if means is None:
        # Parse mean qualities
        means = np.mean(reads.quality, axis=1)

    # Compute the 10th percentile of the mean qualities
    plt_q_10 = np.quantile(means, 0.1)

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    sns.histplot(means,bins=50, element='step', color='C0', ax=ax, kde=False)
    ax.axvspan(plt_q_10,max(means),0,870,color='green',zorder=-10, alpha=0.2)
    ax.set_xlabel('Mean Q Score')
    ax.set_xlim(0,50)
    ax.set_ylabel('Count')
    ax.set_yticklabels([f"{int(x):,}" for x in ax.get_yticks()])

    # Get N reads string:
    if (len(reads) >= 1000) and (len(reads) < 1000000):
        n_reads_str = f'N reads = {len(reads)/1000:.1f}k'
    elif len(reads) >= 1000000:
        n_reads_str = f'N reads = {len(reads)/1000000:.1f}M'
    else:
        n_reads_str = f'N reads = {len(reads)}'

    ax.text(s=n_reads_str,x=0.05,y=0.9,fontdict={'fontsize':10}, transform=plt.gca().transAxes)
    ax.text(s=f'90% above Q{int(plt_q_10)}',x=0.05,y=0.85,fontdict={'fontsize':9}, color='green', transform=plt.gca().transAxes)

    return ax

def _parse_well(w):
    m = re.match(r"(\d+)([A-P]+)(\d+)", str(w))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else (None, None, None)

def _well_label(r, c):
    return f"{r}{int(c)}"

def make_plate_map_bokeh_reads(df, well_col="well_pos", ref_col="ref_name",
                               min_reads=100, max_lines=6,
                               well_size=26, plot_width=800,
                               streakout_wells=None,
                               mutation_wells=None,
                               silent_mutation_wells=None):
    
    ROWS = list(string.ascii_uppercase[:16])  # A–P

    # Merge fwd/rev orientations so both strands count as the same species
    df = df.copy()
    df[ref_col] = df[ref_col].str.replace(r'^(fwd|rev):', '', regex=True)

    # --- aggregate ---
    g = df.groupby([well_col, ref_col]).size().reset_index(name="n")
    if g.empty:
        raise ValueError(
            f"No reads have both '{well_col}' and '{ref_col}' assigned — "
            "plate map cannot be generated."
        )
    g["plate"], g["row"], g["col"] = zip(*g[well_col].map(_parse_well))
    g = g.dropna(subset=["plate"])
    g["frac"] = g.groupby(well_col)["n"].transform(lambda x: x/x.sum())

    comp = (g.sort_values([well_col,"frac"], ascending=[True,False])
          .groupby(well_col, group_keys=False)
          .apply(lambda x: "<br/>".join(
              [f"<b>{_well_label(x.iloc[0].row, x.iloc[0].col)}</b>"] +
              [f"{r} {p:.0%}" for r,p in zip(x[ref_col], x["frac"])][:max_lines]
          ), **_APPLY_KWARGS)
          .rename("tooltip").reset_index())

    dom = g.sort_values([well_col,"n"], ascending=[True,False]).groupby(well_col).head(1)
    dom = dom.merge(comp, on=well_col)
    totals = g.groupby(well_col)["n"].sum().reset_index().rename(columns={"n":"reads"})
    dom = dom.merge(totals, on=well_col)

    dom["RowCat"] = pd.Categorical(dom["row"], categories=ROWS[::-1], ordered=True)
    dom["well"] = dom.apply(lambda r: _well_label(r["row"], r["col"]), axis=1)

    # --- full layout ---
    full_layout = pd.DataFrame(
        [(r, c) for r in ROWS for c in range(1, 25)], columns=["row", "col"]
    )
    full_layout["RowCat"] = pd.Categorical(full_layout["row"], categories=ROWS[::-1], ordered=True)
    full_layout["well"] = full_layout.apply(lambda r: _well_label(r["row"], r["col"]), axis=1)

    _streakout_set = streakout_wells or set()
    _mutation_set = mutation_wells or set()
    _silent_mut_set = silent_mutation_wells or set()

    def fill_plate(p):
        merged = full_layout.copy()
        sub = dom[dom["plate"] == p]
        merged = merged.merge(sub[["row","col","plate","tooltip","reads","frac"]],
                              on=["row","col"], how="left")
        merged["plate"] = p
        merged["tooltip"] = merged["tooltip"].fillna("empty")
        merged["reads"] = merged["reads"].fillna(0)
        merged["frac"] = merged["frac"].fillna(0)

        # Flag potential doublets and precompute lower-right triangle overlays.
        # Triangle is split by diagonal from bottom-left to top-right.
        tri_offset = 0.37
        def _doublet_overlay(r):
            if (r["reads"] > 20) and (r["frac"] < 0.9):
                return (
                    [r["col"] - tri_offset, r["col"] + tri_offset, r["col"] + tri_offset],
                    [(r["row"], -tri_offset), (r["row"], -tri_offset), (r["row"], tri_offset)],
                )
            return ([], [])

        merged["doublet_xs"], merged["doublet_ys"] = zip(
            *merged.apply(_doublet_overlay, axis=1)
        )

        # Blue top-left corner tab for streak-out candidates (verified mixed
        # wells where both subpopulations have correct consensus).
        so_offset = 0.37
        def _streakout_overlay(r):
            key = f"{int(p)}_{_well_label(r['row'], r['col'])}"
            if key in _streakout_set and r["reads"] >= 20:
                return (
                    [r["col"] - so_offset, r["col"] - so_offset, r["col"] + so_offset],
                    [(r["row"], so_offset), (r["row"], -so_offset), (r["row"], so_offset)],
                )
            return ([], [])

        merged["streakout_xs"], merged["streakout_ys"] = zip(
            *merged.apply(_streakout_overlay, axis=1)
        )

        def _streakout_url(r):
            key = f"{int(p)}_{_well_label(r['row'], r['col'])}"
            if key in _streakout_set:
                return f"streakout/well_{key}.html"
            return ""

        merged["streakout_url"] = merged.apply(_streakout_url, axis=1)
        merged["streakout_hint"] = merged["streakout_url"].apply(
            lambda u: '<div style="font-size:11px;color:#2563eb;margin-top:2px;">'
                      '→ Multiple colonies — click to view pileup</div>' if u else ""
        )

        # Red top-left corner tab for mutation wells (mapped to a library member
        # but consensus has a non-synonymous mutation or indel).
        mut_offset = 0.37
        def _mutation_overlay(r):
            key = f"{int(p)}_{_well_label(r['row'], r['col'])}"
            if key in _mutation_set and r["reads"] >= 20:
                return (
                    [r["col"] - mut_offset, r["col"] - mut_offset, r["col"] + mut_offset],
                    [(r["row"], mut_offset), (r["row"], -mut_offset), (r["row"], mut_offset)],
                )
            return ([], [])

        merged["mutation_xs"], merged["mutation_ys"] = zip(
            *merged.apply(_mutation_overlay, axis=1)
        )

        def _mutation_url(r):
            key = f"{int(p)}_{_well_label(r['row'], r['col'])}"
            if key in _mutation_set:
                return f"mutation/pileup/well_{key}.html"
            return ""

        merged["mutation_url"] = merged.apply(_mutation_url, axis=1)
        merged["mutation_hint"] = merged["mutation_url"].apply(
            lambda u: '<div style="font-size:11px;color:#dc2626;margin-top:2px;">'
                      '⚠ Mutation — click to view pileup</div>' if u else ""
        )

        # Amber top-left corner tab for silent mutation wells (synonymous DNA
        # change — correct protein, not flagged as a real mutation).
        sm_offset = 0.37
        def _silent_mut_overlay(r):
            key = f"{int(p)}_{_well_label(r['row'], r['col'])}"
            if key in _silent_mut_set and r["reads"] >= 20:
                return (
                    [r["col"] - sm_offset, r["col"] - sm_offset, r["col"] + sm_offset],
                    [(r["row"], sm_offset), (r["row"], -sm_offset), (r["row"], sm_offset)],
                )
            return ([], [])

        merged["silent_mut_xs"], merged["silent_mut_ys"] = zip(
            *merged.apply(_silent_mut_overlay, axis=1)
        )
        merged["silent_mut_hint"] = merged.apply(
            lambda r: (
                '<div style="font-size:11px;color:#d97706;margin-top:2px;">'
                '~ Silent mutation — synonymous DNA change</div>'
                if f"{int(p)}_{_well_label(r['row'], r['col'])}" in _silent_mut_set
                else ""
            ),
            axis=1,
        )
        return merged

    # Filter out ghost plates with fewer than 250 total reads
    _MIN_PLATE_READS = 250
    _plate_read_totals = dom.groupby("plate")["reads"].sum()
    plates = sorted(p for p in dom["plate"].unique()
                    if _plate_read_totals.get(p, 0) >= _MIN_PLATE_READS)
    if not plates:
        plates = sorted(dom["plate"].unique())  # fallback: show all if none pass
    plate_dict = {str(p): fill_plate(p).to_dict(orient="list") for p in plates}

    custom_cmap = get_custom_cmap()
    palette = [mcolors.rgb2hex(custom_cmap(i / 255)[:3]) for i in range(256)]
    mapper = LinearColorMapper(palette=palette, low=0, high=min_reads * 2)

    TOOLTIPS = """
    <div style="line-height:1.2">
      <div style="font-size:13px;">Plate @plate · <b>@well</b></div>
      <div style="margin-top:4px;">@tooltip{safe}</div>
      <div style="font-size:11px;color:#666;margin-top:4px;">
        Reads: @reads &nbsp;|&nbsp; Top frac: @frac{0.0%}
      </div>
      @streakout_hint{safe}
      @mutation_hint{safe}
      @silent_mut_hint{safe}
    </div>
    """

    start_plate = plates[0]
    src = ColumnDataSource(plate_dict[str(start_plate)])

    fig = figure(x_range=(0.5, 24.5), y_range=ROWS[::-1],
                 width=900, height=560, tools="reset",
                 title=f"Plate {start_plate}")
    well_renderer = fig.rect(
        "col", "RowCat", width=0.74, height=0.74, source=src,
        fill_color={'field': 'reads', 'transform': mapper},
        line_color="darkgray", line_width=1.2,
        nonselection_fill_alpha=1.0, nonselection_line_alpha=1.0,
    )
    fig.patches(
        "doublet_xs", "doublet_ys", source=src,
        fill_color="#C6C6C6", fill_alpha=1.0, line_color=None
    )
    fig.patches(
        "streakout_xs", "streakout_ys", source=src,
        fill_color="#2563eb", fill_alpha=1.0, line_color=None
    )
    fig.patches(
        "mutation_xs", "mutation_ys", source=src,
        fill_color="#dc2626", fill_alpha=1.0, line_color=None
    )
    fig.patches(
        "silent_mut_xs", "silent_mut_ys", source=src,
        fill_color="#d97706", fill_alpha=1.0, line_color=None
    )
    fig.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[well_renderer]))
    fig.add_tools(TapTool(renderers=[well_renderer], callback=CustomJS(args=dict(src=src), code="""
        const indices = src.selected.indices;
        if (indices.length > 0) {
            const so_url = src.data['streakout_url'][indices[0]];
            const mut_url = src.data['mutation_url'][indices[0]];
            const url = so_url || mut_url;
            if (url && url.length > 0) {
                (window.top || window).open(url, '_blank');
            }
            src.selected.indices = [];
        }
    """)))
    fig.xaxis.ticker = list(range(1, 25))
    fig.grid.grid_line_color = None

    # colorbar with tier labels at A (100), B (50), C (20)
    from bokeh.models import FixedTicker
    color_bar = ColorBar(color_mapper=mapper,
                         label_standoff=8, width=12, location=(0,0),
                         title="Read Count", title_text_font_size="14pt",
                         bar_line_color="black", major_tick_line_color="black",
                         major_label_text_font_size="12pt", major_tick_line_width=2,
                         ticker=FixedTicker(ticks=[0, 20, 50, 100, min_reads * 2]))
    color_bar.formatter = CustomJSTickFormatter(code=f"""
        if (tick == 100) {{ return "A (100)"; }}
        else if (tick == 50) {{ return "B (50)"; }}
        else if (tick == 20) {{ return "C (20)"; }}
        else if (tick == {min_reads * 2}) {{ return "\u2265{min_reads * 2}"; }}
        else {{ return tick.toString(); }}
    """)
    fig.add_layout(color_bar, 'right')

    if len(plates) > 1:
        btn_group = RadioButtonGroup(
            labels=[str(p) for p in plates],
            active=0,
            stylesheets=[".bk-btn { min-width: 40px; font-size: 14px; }"],
        )
        btn_group.js_on_change("active", CustomJS(
            args=dict(src=src, figs=fig, data=plate_dict, labels=[str(p) for p in plates]),
            code="""
            const p = labels[cb_obj.active];
            const new_data = {};
            for (let k in data[p]) { new_data[k] = data[p][k].slice(); }
            src.data = new_data;
            figs.title.text = "Plate " + p;
            src.change.emit();
            """,
        ))
        layout = column(btn_group, fig)
    else:
        layout = column(fig)
    return layout


def save_plate_map_html(df, output_path, title="Plate Map",
                        streakout_wells=None, mutation_wells=None,
                        silent_mutation_wells=None, **kwargs):
    """Generate an interactive plate map and save as standalone HTML.

    Wraps :func:`make_plate_map_bokeh_reads` and writes a self-contained
    HTML file using Bokeh's inline resources.

    Args:
        df: DataFrame with ``well_pos`` and ``ref_name`` columns.
        output_path: Path to write the HTML file.
        title: HTML page title.
        streakout_wells: Optional set of ``"{plate}_{well}"`` keys for
            streak-out candidates (shown as blue corner tabs).
        mutation_wells: Optional set of ``"{plate}_{well}"`` keys for
            wells with consensus mutations (shown as red corner tabs).
        silent_mutation_wells: Optional set of ``"{plate}_{well}"`` keys for
            wells with synonymous DNA changes (shown as amber corner tabs).
        **kwargs: Forwarded to :func:`make_plate_map_bokeh_reads`.
    """
    from pathlib import Path

    layout = make_plate_map_bokeh_reads(df, streakout_wells=streakout_wells,
                                        mutation_wells=mutation_wells,
                                        silent_mutation_wells=silent_mutation_wells,
                                        **kwargs)
    html = file_html(layout, INLINE, title)
    html = _inject_usortm_theme(html)
    Path(output_path).write_text(html)


def _inject_usortm_theme(html: str) -> str:
    """Inject summary-style light/dark page background sync into HTML.

    Standalone plate map pages are generated by Bokeh and default to a white
    background. This adds lightweight CSS/JS so they match the summary report
    background and follow the saved ``usortm-theme`` preference.
    """
    if "usortm-theme-bridge" in html:
        return html

    style_block = """
<style id="usortm-theme-bridge">
:root { --usortm-bg: #fafafa; }
[data-theme="dark"] { --usortm-bg: #1a1a2e; }
html, body { background: var(--usortm-bg) !important; }
</style>
""".strip()

    script_block = """
<script id="usortm-theme-sync">
(function () {
  try {
    var stored = localStorage.getItem('usortm-theme');
    if (stored === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  } catch (e) {
    document.documentElement.removeAttribute('data-theme');
  }
})();
</script>
""".strip()

    if "<head>" in html:
        html = html.replace("<head>", "<head>\n" + style_block, 1)

    if "</body>" in html:
        html = html.replace("</body>", script_block + "\n</body>", 1)

    return html


def _tier_colors(min_reads=100):
    """Sample tier colors from the custom colormap at fixed read counts.

    Tier A → color at 115 reads, B → 70 reads, C → 40 reads, matching
    the same 0–(min_reads*2) scale used by the demux plate map colorbar.
    """
    cmap = get_custom_cmap()
    high = min_reads * 2
    tier_reads = {"A": 150, "B": 70, "C": 40}
    colors = {
        tier: mcolors.rgb2hex(cmap(min(reads / high, 1.0))[:3])
        for tier, reads in tier_reads.items()
    }
    colors[""] = "#FFFFFF"
    colors["Streakout"] = "#3B82F6"  # blue
    return colors

def _well_tier(reads, cons_frac):
    """Return quality tier (A/B/C) or '' for a well based on read/consensus thresholds."""
    if reads >= 100 and cons_frac > 0.9:
        return "A"
    if reads >= 50 and cons_frac > 0.9:
        return "B"
    if reads >= 20 and cons_frac > 0.9:
        return "C"
    return ""


def make_pick_plate_map_bokeh(pick_list, target_format=384,
                               min_reads=100, well_size=26, plot_width=800,
                               pileup_url_map=None):
    """Create an interactive Bokeh plate map for a cherry-pick list.

    Each well shows the picked variant with hover details including source
    plate/well, read count, and consensus fraction.

    Args:
        pick_list: List of dicts from ``pick.py._generate_pick_list()``.
            Each dict has keys: variant, source_plate, source_well,
            target_plate, target_well, reads, consensus_fraction.
        target_format: Target plate format (96 or 384).
        well_size: Size of well markers in pixels.
        plot_width: Width of the Bokeh figure.

    Returns:
        Bokeh layout (column with optional plate slider).
    """
    if target_format == 96:
        n_rows, n_cols = 8, 12
        ROWS = list(string.ascii_uppercase[:8])
    else:
        n_rows, n_cols = 16, 24
        ROWS = list(string.ascii_uppercase[:16])

    # Parse target wells and group by plate
    for hit in pick_list:
        m = re.match(r"([A-P])(\d+)", hit["target_well"])
        if m:
            hit["_row"] = m.group(1)
            hit["_col"] = int(m.group(2))
        hit["_plate"] = int(hit.get("target_plate", 0))

    # Build full layout for each plate
    full_layout = pd.DataFrame(
        [(r, c) for r in ROWS for c in range(1, n_cols + 1)],
        columns=["row", "col"],
    )
    full_layout["RowCat"] = pd.Categorical(
        full_layout["row"], categories=ROWS[::-1], ordered=True
    )
    full_layout["well"] = full_layout.apply(
        lambda r: f"{r['row']}{int(r['col'])}", axis=1
    )

    plates = sorted(set(h["_plate"] for h in pick_list))
    if not plates:
        plates = [0]

    TIER_COLORS = _tier_colors(min_reads)
    _pileup_map = pileup_url_map or {}

    def fill_plate(p):
        merged = full_layout.copy()
        plate_pileups = _pileup_map.get(str(p), {})
        sub_rows = []
        for h in pick_list:
            if h["_plate"] != p:
                continue
            # Entry-level pileup_url (e.g. streakout) takes precedence over map lookup
            pileup_url = h.get("pileup_url") or plate_pileups.get(h.get("target_well", ""), "")
            pileup_hint = (
                '<div style="font-size:11px;color:#2563eb;margin-top:2px;">'
                '→ Click to view pileup</div>'
                if pileup_url else ""
            )
            tier_override = h.get("tier_override", "")
            if tier_override == "Streakout":
                tier_label = "Streakout"
                tooltip_extra = (
                    f"<div style='color:#2563eb;font-weight:bold;margin-top:2px;'>"
                    f"Streakout recoverable</div>"
                    f"Source: {h['source_plate']}:{h['source_well']}<br/>"
                    f"Reads in source: {h['reads']:,} ({h['consensus_fraction']:.0%} of well)"
                )
            else:
                tier_label = _well_tier(h["reads"], h["consensus_fraction"]) or "N/A"
                tooltip_extra = (
                    f"Source: {h['source_plate']}:{h['source_well']}<br/>"
                    f"Reads: {h['reads']:,}<br/>"
                    f"Consensus: {h['consensus_fraction']:.0%}"
                )
            sub_rows.append({
                "row": h["_row"],
                "col": h["_col"],
                "variant": h["variant"],
                "source": f"{h['source_plate']}:{h['source_well']}",
                "reads": h["reads"],
                "cons_frac": h["consensus_fraction"],
                "pileup_url": pileup_url,
                "tier_override": tier_override,
                "tooltip": (
                    f"<b>{h['target_well']}</b><br/>"
                    f"{h['variant']}<br/>"
                    f"{tooltip_extra}<br/>"
                    f"Tier: {tier_label}"
                    f"{pileup_hint}"
                ),
            })
        sub = pd.DataFrame(sub_rows)
        if len(sub) > 0:
            merged = merged.merge(
                sub[["row", "col", "variant", "source", "reads", "cons_frac",
                     "tooltip", "pileup_url", "tier_override"]],
                on=["row", "col"], how="left",
            )
        else:
            merged["variant"] = ""
            merged["source"] = ""
            merged["reads"] = 0
            merged["cons_frac"] = 0.0
            merged["tooltip"] = "empty"
            merged["pileup_url"] = ""
            merged["tier_override"] = ""
        merged["plate"] = p
        merged["variant"] = merged["variant"].fillna("")
        merged["source"] = merged["source"].fillna("")
        merged["reads"] = merged["reads"].fillna(0)
        merged["cons_frac"] = merged["cons_frac"].fillna(0)
        merged["tooltip"] = merged["tooltip"].fillna("empty")
        merged["pileup_url"] = merged["pileup_url"].fillna("")
        merged["tier_override"] = merged["tier_override"].fillna("")
        merged["tier"] = merged.apply(
            lambda r: r["tier_override"] if r["tier_override"] else _well_tier(r["reads"], r["cons_frac"]), axis=1
        )
        merged["tier_color"] = merged["tier"].map(TIER_COLORS)
        return merged

    plate_dict = {str(p): fill_plate(p).to_dict(orient="list") for p in plates}

    TOOLTIPS = """
    <div style="line-height:1.4">
      <div>@tooltip{safe}</div>
    </div>
    """

    start_plate = plates[0]
    src = ColumnDataSource(plate_dict[str(start_plate)])

    fig_obj = figure(
        x_range=(0.5, n_cols + 0.5), y_range=ROWS[::-1],
        width=900, height=560 if target_format == 384 else 340,
        tools="reset",
        title=f"Pick Plate {start_plate}",
    )
    well_renderer = fig_obj.scatter(
        "col", "RowCat", size=well_size, source=src, marker="square",
        fill_color="tier_color",
        line_color="darkgray", line_width=1.2,
        nonselection_fill_alpha=1.0, nonselection_line_alpha=1.0,
    )
    fig_obj.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[well_renderer]))
    fig_obj.add_tools(TapTool(renderers=[well_renderer], callback=CustomJS(args=dict(src=src), code="""
        const indices = src.selected.indices;
        if (indices.length > 0) {
            const url = src.data['pileup_url'][indices[0]];
            if (url && url.length > 0) {
                (window.top || window).open(url, '_blank');
            }
            src.selected.indices = [];
        }
    """)))
    fig_obj.xaxis.ticker = list(range(1, n_cols + 1))
    fig_obj.grid.grid_line_color = None

    if len(plates) > 1:
        btn_group = RadioButtonGroup(
            labels=[str(p) for p in plates],
            active=0,
            stylesheets=[".bk-btn { min-width: 40px; font-size: 14px; }"],
        )
        btn_group.js_on_change("active", CustomJS(
            args=dict(src=src, figs=fig_obj, data=plate_dict, labels=[str(p) for p in plates]),
            code="""
            const p = labels[cb_obj.active];
            const new_data = {};
            for (let k in data[p]) { new_data[k] = data[p][k].slice(); }
            src.data = new_data;
            figs.title.text = "Pick Plate " + p;
            src.change.emit();
            """,
        ))
        layout = column(btn_group, fig_obj)
    else:
        layout = column(fig_obj)

    return layout


def save_pick_plate_map_html(pick_list, output_path, title="Pick Plate Map",
                              pileup_url_map=None, **kwargs):
    """Generate a pick plate map and save as standalone HTML.

    Args:
        pick_list: Pick list from ``pick.py._generate_pick_list()``.
        output_path: Path to write the HTML file.
        title: HTML page title.
        pileup_url_map: Optional nested dict ``{plate: {well: url}}`` mapping
            target wells to their pileup HTML paths (relative to the HTML file).
        **kwargs: Forwarded to :func:`make_pick_plate_map_bokeh`.
    """
    from pathlib import Path

    layout = make_pick_plate_map_bokeh(pick_list, pileup_url_map=pileup_url_map, **kwargs)
    html = file_html(layout, INLINE, title)
    html = _inject_usortm_theme(html)
    Path(output_path).write_text(html)
