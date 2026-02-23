import re, string, numpy as np, pandas as pd

import bionumpy as bnp

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, RadioButtonGroup, CustomJS,
    LinearColorMapper, ColorBar, CustomJSTickFormatter
)
from bokeh.layouts import column
from bokeh.embed import file_html
from bokeh.resources import INLINE

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
                               well_size=26, plot_width=800):
    
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
          ), include_groups=False)  # suppress future warning
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

    def fill_plate(p):
        merged = full_layout.copy()
        sub = dom[dom["plate"] == p]
        merged = merged.merge(sub[["row","col","plate","tooltip","reads","frac"]],
                              on=["row","col"], how="left")
        merged["plate"] = p
        merged["tooltip"] = merged["tooltip"].fillna("empty")
        merged["reads"] = merged["reads"].fillna(0)
        merged["frac"] = merged["frac"].fillna(0)
        return merged

    plates = sorted(dom["plate"].unique())
    plate_dict = {str(p): fill_plate(p).to_dict(orient="list") for p in plates}

    # cool-warm diverging colormap: blue → white → red, centered at min_reads
    def make_diverging_gradient(hex_low, hex_mid, hex_high, n=256):
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", [hex_low, hex_mid, hex_high]
        )
        return [mcolors.rgb2hex(cmap(i / n)[:3]) for i in range(n)]

    palette = make_diverging_gradient("#4575B4", "#FFFFFF", "#D73027", 256)
    mapper = LinearColorMapper(palette=palette, low=0, high=min_reads * 2)

    TOOLTIPS = """
    <div style="line-height:1.2">
      <div style="font-size:13px;">Plate @plate · <b>@well</b></div>
      <div style="margin-top:4px;">@tooltip{safe}</div>
      <div style="font-size:11px;color:#666;margin-top:4px;">
        Reads: @reads &nbsp;|&nbsp; Top frac: @frac{0.0%}
      </div>
    </div>
    """

    start_plate = plates[0]
    src = ColumnDataSource(plate_dict[str(start_plate)])

    fig = figure(x_range=(0.5, 24.5), y_range=ROWS[::-1],
                 width=900, height=560, tools="reset",
                 title=f"Plate {start_plate}")
    fig.scatter("col", "RowCat", size=well_size, source=src, marker="square",
                fill_color={'field': 'reads', 'transform': mapper},
                line_color="darkgray", line_width=1.2)
    fig.add_tools(HoverTool(tooltips=TOOLTIPS))
    fig.xaxis.ticker = list(range(1, 25))
    fig.grid.grid_line_color = None

    # colorbar with custom top tick
    color_bar = ColorBar(color_mapper=mapper, 
                         label_standoff=8, width=12, location=(0,0),
                         title="Read Count", title_text_font_size="14pt",
                         bar_line_color="black", major_tick_line_color="black", 
                         major_label_text_font_size="12pt", major_tick_line_width=2)
    color_bar.formatter = CustomJSTickFormatter(code=f"""
        if (tick == {min_reads}) {{
            return "{min_reads} (threshold)";
        }} else if (tick == {min_reads * 2}) {{
            return "\u2265{min_reads * 2}";
        }} else {{
            return tick.toString();
        }}
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


def save_plate_map_html(df, output_path, title="Plate Map", **kwargs):
    """Generate an interactive plate map and save as standalone HTML.

    Wraps :func:`make_plate_map_bokeh_reads` and writes a self-contained
    HTML file using Bokeh's inline resources.

    Args:
        df: DataFrame with ``well_pos`` and ``ref_name`` columns.
        output_path: Path to write the HTML file.
        title: HTML page title.
        **kwargs: Forwarded to :func:`make_plate_map_bokeh_reads`.
    """
    from pathlib import Path

    layout = make_plate_map_bokeh_reads(df, **kwargs)
    html = file_html(layout, INLINE, title)
    Path(output_path).write_text(html)


def make_pick_plate_map_bokeh(pick_list, target_format=384,
                               well_size=26, plot_width=800):
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

    max_reads = max((h["reads"] for h in pick_list), default=100)

    def fill_plate(p):
        merged = full_layout.copy()
        sub = pd.DataFrame([
            {
                "row": h["_row"],
                "col": h["_col"],
                "variant": h["variant"],
                "source": f"{h['source_plate']}:{h['source_well']}",
                "reads": h["reads"],
                "cons_frac": h["consensus_fraction"],
                "tooltip": (
                    f"<b>{h['target_well']}</b><br/>"
                    f"{h['variant']}<br/>"
                    f"Source: {h['source_plate']}:{h['source_well']}<br/>"
                    f"Reads: {h['reads']:,}<br/>"
                    f"Consensus: {h['consensus_fraction']:.0%}"
                ),
            }
            for h in pick_list if h["_plate"] == p
        ])
        if len(sub) > 0:
            merged = merged.merge(
                sub[["row", "col", "variant", "source", "reads", "cons_frac", "tooltip"]],
                on=["row", "col"], how="left",
            )
        else:
            merged["variant"] = ""
            merged["source"] = ""
            merged["reads"] = 0
            merged["cons_frac"] = 0.0
            merged["tooltip"] = "empty"
        merged["plate"] = p
        merged["variant"] = merged["variant"].fillna("")
        merged["source"] = merged["source"].fillna("")
        merged["reads"] = merged["reads"].fillna(0)
        merged["cons_frac"] = merged["cons_frac"].fillna(0)
        merged["tooltip"] = merged["tooltip"].fillna("empty")
        return merged

    plate_dict = {str(p): fill_plate(p).to_dict(orient="list") for p in plates}

    # Gradient white → green for pick maps
    def make_gradient(hex1, hex2, n=256):
        cmap = mcolors.LinearSegmentedColormap.from_list("", [hex1, hex2])
        return [mcolors.rgb2hex(cmap(i / n)[:3]) for i in range(n)]

    palette = make_gradient("#FFFFFF", "#059669", 256)
    mapper = LinearColorMapper(palette=palette, low=0, high=max_reads)

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
    fig_obj.scatter(
        "col", "RowCat", size=well_size, source=src, marker="square",
        fill_color={"field": "reads", "transform": mapper},
        line_color="darkgray", line_width=1.2,
    )
    fig_obj.add_tools(HoverTool(tooltips=TOOLTIPS))
    fig_obj.xaxis.ticker = list(range(1, n_cols + 1))
    fig_obj.grid.grid_line_color = None

    color_bar = ColorBar(
        color_mapper=mapper,
        label_standoff=8, width=12, location=(0, 0),
        title="Read Count", title_text_font_size="14pt",
        bar_line_color="black", major_tick_line_color="black",
        major_label_text_font_size="12pt", major_tick_line_width=2,
    )
    fig_obj.add_layout(color_bar, "right")

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
                              **kwargs):
    """Generate a pick plate map and save as standalone HTML.

    Args:
        pick_list: Pick list from ``pick.py._generate_pick_list()``.
        output_path: Path to write the HTML file.
        title: HTML page title.
        **kwargs: Forwarded to :func:`make_pick_plate_map_bokeh`.
    """
    from pathlib import Path

    layout = make_pick_plate_map_bokeh(pick_list, **kwargs)
    html = file_html(layout, INLINE, title)
    Path(output_path).write_text(html)