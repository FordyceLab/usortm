import re, string, numpy as np, pandas as pd

import bionumpy as bnp

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, HoverTool, Slider, CustomJS,
    LinearColorMapper, ColorBar, CustomJSTickFormatter
)
from bokeh.layouts import column
from bokeh.io import output_notebook

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

    # --- aggregate ---
    g = df.groupby([well_col, ref_col]).size().reset_index(name="n")
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

    # gradient white → blue
    def make_gradient(hex1, hex2, n=256):
        cmap = mcolors.LinearSegmentedColormap.from_list("", [hex1, hex2])
        return [mcolors.rgb2hex(cmap(i/n)[:3]) for i in range(n)]

    palette = make_gradient("#FFFFFF", "#005DCE", 256)
    mapper = LinearColorMapper(palette=palette, low=0, high=min_reads)

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
                 width=plot_width, height=500, tools="reset",
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
            return ">{min_reads}";
        }} else {{
            return tick.toString();
        }}
    """)
    fig.add_layout(color_bar, 'right')

    slider = Slider(start=min(plates), end=max(plates), step=1, value=start_plate,
                    title="Plate")
    slider.js_on_change("value", CustomJS(args=dict(src=src, figs=fig, data=plate_dict),
        code="""
        const p = cb_obj.value.toString();
        const new_data = {};
        for (let k in data[p]) {
            new_data[k] = data[p][k].slice();
        }
        src.data = new_data;
        figs.title.text = "Plate " + p;
        src.change.emit();
    """))

    layout = column(slider, fig)
    show(layout)
    return layout