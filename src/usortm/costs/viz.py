import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_total_cost_singleFrag_varyLib(frag_len,
                                 commercial_costs_df,
                                 usortm_costs_df,
                                 plot_export_dir,
                                 fold_savings_lib_size=1000,
                                 sdm_costs_df=None,
                                 ):
    """Plot total cost comparison between commercial and uSort-M methods.

    Creates two plots: a full-range comparison and a zoomed view near the crossover point.

    Args:
        frag_len: Fragment/sequence length (bp) to plot.
        commercial_costs_df: DataFrame from generate_commercial_cost_dict().
        usortm_costs_df: DataFrame from get_usortm_costs().
        plot_export_dir: Directory path to save output plots.
        fold_savings_lib_size: Library size at which to annotate fold savings (default: 1000).
        sdm_costs_df: Optional DataFrame from generate_sdm_costs(). If provided,
            adds an SDM trace (yellow) to both panels.

    Returns:
        None. Saves two PDF files to plot_export_dir.
    """

    # =======================
    # Format costs
    # =======================

    # --- Extract commercial costs ---
    # Filter for specific fragment length and Total rows only
    commercial_df = commercial_costs_df[(commercial_costs_df['Length'] == frag_len) &
                              (commercial_costs_df['Step'] == 'Total')]
    # Group by library size to get min/mean/max across vendors
    commercial_grouped = commercial_df.groupby('Library Size')['Cost'].agg(['min', 'mean', 'max'])
    sizes = np.array(commercial_grouped.index)
    mins = np.array(commercial_grouped['min'])
    means = np.array(commercial_grouped['mean'])
    maxs = np.array(commercial_grouped['max'])

    # --- Extract uSort-M costs ---
    # Filter for specific length and Total rows only
    usortm_df = usortm_costs_df[(usortm_costs_df['Length'] == frag_len) &
                                (usortm_costs_df['Step'] == 'Total')]
    usort_sizes = np.array(sorted(usortm_df['Library Size'].unique()))
    usort_costs = np.array([usortm_df[usortm_df['Library Size'] == s]['Cost'].values[0]
                           for s in usort_sizes])

    # --- Extract SDM costs (optional) ---
    sdm_sizes = sdm_costs = None
    if sdm_costs_df is not None:
        sdm_df = sdm_costs_df[(sdm_costs_df['Length'] == frag_len) &
                              (sdm_costs_df['Step'] == 'Total')]
        if not sdm_df.empty:
            sdm_sizes = np.array(sorted(sdm_df['Library Size'].unique()))
            sdm_costs = np.array([sdm_df[sdm_df['Library Size'] == s]['Cost'].values[0]
                                  for s in sdm_sizes])

    # --- Find crossover point ---
    mean_interp = np.interp(usort_sizes, sizes, means)
    diff = mean_interp - usort_costs
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) > 0:
        idx = sign_changes[0]
        x0, x1 = usort_sizes[idx], usort_sizes[idx+1]
        y0, y1 = diff[idx], diff[idx+1]
        crossover_x = x0 - y0 * (x1 - x0) / (y1 - y0)
        crossover_y = np.interp(crossover_x, usort_sizes, usort_costs)
    else:
        crossover_x, crossover_y = None, None

    # --- Shared figure settings ---
    FIGSIZE = (2.6, 2.6)
    DPI = 150

    # =======================
    # Panel 1: Full range
    # =======================
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.fill_between(sizes, mins, maxs, color='grey', alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(sizes, means, color='grey', zorder=1, linewidth=2, label="Commercial\nGene Fragments")
    if sdm_sizes is not None:
        ax.plot(sdm_sizes, sdm_costs, color='#eab308', zorder=1, linewidth=2, label="SDM")
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2, label="uSort-M")

    ax.set_xlim(xmax=2500)

    # Add a bit of padding to top
    ax.set_ylim(ymax=max(maxs)*0.8)
    
    # Add commas to x-axis labels
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_yticklabels([f"${int(x/1000)}k" if x != 0 else f"${int(x)}" for x in ax.get_yticks()])
    ax.set_xlabel(f"Library Size", fontsize=12)
    ax.set_ylabel("Total Projected Cost (USD)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{frag_len:,} bp fragments")

    # Annotate crossover point
    if crossover_x is not None:
        ax.scatter([crossover_x], [crossover_y], s=20, color='black', zorder=3)

    # --- Add dashed line + savings annotation at specified library size ---
    if fold_savings_lib_size:
        lib_target = fold_savings_lib_size
        # Check if target is within the range of both datasets
        if (sizes.min() <= lib_target <= sizes.max() and
            usort_sizes.min() <= lib_target <= usort_sizes.max()):
            # Use interpolation to get costs at target library size
            grey_y = np.interp(lib_target, sizes, means)
            blue_y = np.interp(lib_target, usort_sizes, usort_costs)
            fold_savings = grey_y / blue_y

            # Dashed connector line with endpoints and centered label
            mid_y = (grey_y + blue_y) / 2
            ax.plot([lib_target, lib_target], [blue_y, grey_y],
                    color='black', linestyle='--', linewidth=1, zorder=2)
            print(lib_target)
            ax.scatter([lib_target, lib_target], [blue_y, grey_y],
                    color='black', s=6, zorder=3)
            ax.text(lib_target * 1.1, mid_y,
                    f"{fold_savings:.1f}-fold savings\n@ {lib_target:,}",
                    va='center', ha='left', fontsize=8)
            
    # Add minor yticks every 1000
    

    # Set faceolor to none
    ax.set_facecolor('none')

    full_path = os.path.join(plot_export_dir, f"Cost_comparison_{frag_len}bp_full.pdf")
    plt.savefig(full_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)

    # =======================
    # Panel 2: Zoom near crossover
    # =======================
    FIGSIZE = (1,1)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.fill_between(sizes, mins, maxs, color='grey', alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(sizes, means, color='grey', zorder=1, linewidth=2)
    if sdm_sizes is not None:
        ax.plot(sdm_sizes, sdm_costs, color='#eab308', zorder=1, linewidth=2)
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2)

    if crossover_x is not None:
        zoom_xmin = max(0, crossover_x - 50)
        zoom_xmax = crossover_x + 50
        zoom_ymax = crossover_y * 1.5
        ax.set_xlim(zoom_xmin, zoom_xmax)
        ax.set_ylim(0, zoom_ymax)
    else:
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 5000)

    ax.tick_params(labelsize=9)
    ax.set_yticklabels([f"${int(y/1000)}k" if y >= 1000 else f"${int(y)}"
                        for y in ax.get_yticks()])

    # Annotate crossover point
    if crossover_x is not None:
        ax.scatter([crossover_x], [crossover_y], s=20, color='black', zorder=3)
        ax.annotate(f"{int(crossover_x)} seq",
                    xy=(crossover_x, crossover_y),
                    xytext=(crossover_x * 1.25, crossover_y - (0.35 * crossover_y)),
                    fontsize=10,
                    ha='left')

    # Set faceolor to none
    ax.set_facecolor('none')

    zoom_path = os.path.join(plot_export_dir, f"Cost_comparison_{frag_len}bp_zoom.pdf")
    plt.savefig(zoom_path, bbox_inches='tight', transparent=True)

    plt.show()
    plt.close(fig)

    print(f"Saved:\n - Full: {full_path}\n - Zoom: {zoom_path}")


def plot_cost_per_variant_singleFrag_varyLib(frag_len,
                                              commercial_costs_df,
                                              usortm_costs_df,
                                              plot_export_dir,
                                              fold_savings_lib_size=1000,
                                              sdm_costs_df=None,
                                              ):
    """Plot cost per variant comparison between commercial and uSort-M methods.

    Creates two plots: a full-range comparison and a zoomed view near the crossover point.

    Args:
        frag_len: Fragment/sequence length (bp) to plot.
        commercial_costs_df: DataFrame from generate_commercial_cost_dict().
        usortm_costs_df: DataFrame from get_usortm_costs().
        plot_export_dir: Directory path to save output plots.
        fold_savings_lib_size: Library size at which to annotate fold savings (default: 1000).
        sdm_costs_df: Optional DataFrame from generate_sdm_costs(). If provided,
            adds an SDM trace (yellow) to both panels.

    Returns:
        None. Saves two PDF files to plot_export_dir.
    """

    # =======================
    # Format costs
    # =======================

    # --- Extract commercial costs ---
    # Filter for specific fragment length and Total rows only
    commercial_df = commercial_costs_df[(commercial_costs_df['Length'] == frag_len) &
                              (commercial_costs_df['Step'] == 'Total')]
    # Group by library size to get min/mean/max CPV across vendors
    commercial_grouped = commercial_df.groupby('Library Size')['CPV'].agg(['min', 'mean', 'max'])
    sizes = np.array(commercial_grouped.index)
    mins = np.array(commercial_grouped['min'])
    means = np.array(commercial_grouped['mean'])
    maxs = np.array(commercial_grouped['max'])

    # Filter out any NaN or Inf values
    valid_mask = np.isfinite(mins) & np.isfinite(means) & np.isfinite(maxs)
    sizes = sizes[valid_mask]
    mins = mins[valid_mask]
    means = means[valid_mask]
    maxs = maxs[valid_mask]

    # --- Extract uSort-M costs ---
    # Filter for specific length and Total rows only
    usortm_df = usortm_costs_df[(usortm_costs_df['Length'] == frag_len) &
                                (usortm_costs_df['Step'] == 'Total')]
    usort_sizes = np.array(sorted(usortm_df['Library Size'].unique()))
    usort_costs = np.array([usortm_df[usortm_df['Library Size'] == s]['CPV'].values[0]
                           for s in usort_sizes])

    # --- Extract SDM CPV (optional) ---
    sdm_sizes = sdm_costs = None
    if sdm_costs_df is not None:
        sdm_df = sdm_costs_df[(sdm_costs_df['Length'] == frag_len) &
                              (sdm_costs_df['Step'] == 'Total')]
        if not sdm_df.empty:
            sdm_sizes = np.array(sorted(sdm_df['Library Size'].unique()))
            sdm_costs = np.array([sdm_df[sdm_df['Library Size'] == s]['CPV'].values[0]
                                  for s in sdm_sizes])

    # --- Find crossover point ---
    mean_interp = np.interp(usort_sizes, sizes, means)
    diff = mean_interp - usort_costs
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_changes) > 0:
        idx = sign_changes[0]
        x0, x1 = usort_sizes[idx], usort_sizes[idx+1]
        y0, y1 = diff[idx], diff[idx+1]
        crossover_x = x0 - y0 * (x1 - x0) / (y1 - y0)
        crossover_y = np.interp(crossover_x, usort_sizes, usort_costs)
    else:
        crossover_x, crossover_y = None, None

    # --- Shared figure settings ---
    FIGSIZE = (2.6, 2.6)
    DPI = 150

    # =======================
    # Panel 1: Full range
    # =======================
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.fill_between(sizes, mins, maxs, color='grey', alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(sizes, means, color='grey', zorder=1, linewidth=2, label="Commercial\nGene Fragments")
    if sdm_sizes is not None:
        ax.plot(sdm_sizes, sdm_costs, color='#eab308', zorder=1, linewidth=2, label="SDM")
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2, label="uSort-M")

    # --- Add final cost labels at the end of each trace ---
    # Commercial gene fragments (grey)
    final_commercial_cost = means[-1]
    ax.text(sizes[-1]*1.02, final_commercial_cost, 
            f"${final_commercial_cost:.2f}" if final_commercial_cost < 1 else f"${final_commercial_cost:.1f}",
            color='grey', fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
            )
    
    # uSort-M (blue)
    final_usortm_cost = usort_costs[-1]
    ax.text(usort_sizes[-1]*1.02, final_usortm_cost,
            f"${final_usortm_cost:.2f}" if final_usortm_cost < 1 else f"${final_usortm_cost:.1f}",
            color='#4ba5e2', fontsize=9, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
            )

    ax.set_xlim(xmax=sizes[-1])
    
    # Add a bit of padding to top (with safety check)
    y_max = max(maxs) if len(maxs) > 0 and np.isfinite(max(maxs)) else 100
    ax.set_ylim(ymax=y_max*1.4)
    
    ax.set_xticklabels([f"{int(x):,}" if x != 0 else f"{int(x)}" for x in ax.get_xticks()])
    
    # Format y-axis for cost per variant
    ax.set_yticklabels([f"${int(x)}" if x >= 1 else f"${x:.2f}" for x in ax.get_yticks()])
    
    ax.set_xlabel(f"Library Size", fontsize=12)
    ax.set_ylabel("Cost per Variant (USD)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{frag_len:,} bp fragments")

    # Annotate crossover point
    if crossover_x is not None:
        ax.scatter([crossover_x], [crossover_y], s=20, color='black', zorder=3)

    # --- Add dashed line + savings annotation at specified library size ---
    if fold_savings_lib_size:
        lib_target = fold_savings_lib_size
        # Check if target is within the range of both datasets
        if (sizes.min() <= lib_target <= sizes.max() and
            usort_sizes.min() <= lib_target <= usort_sizes.max()):
            # Use interpolation to get costs at target library size
            grey_y = np.interp(lib_target, sizes, means)
            blue_y = np.interp(lib_target, usort_sizes, usort_costs)
            fold_savings = grey_y / blue_y

            # Dashed connector line with endpoints and centered label
            mid_y = (grey_y + blue_y) / 2
            ax.plot([lib_target, lib_target], [blue_y, grey_y],
                    color='black', linestyle='--', linewidth=1, zorder=2)
            ax.scatter([lib_target, lib_target], [blue_y, grey_y],
                    color='black', s=6, zorder=3)
            ax.text(lib_target * 1.05, mid_y,
                    f"{fold_savings:.1f}-fold savings\n@{lib_target:,}",
                    va='center', ha='left', fontsize=8)

    # Set facecolor to none
    ax.set_facecolor('none')

    full_path = os.path.join(plot_export_dir, f"Cost_per_variant_{frag_len}bp_full.pdf")
    plt.savefig(full_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)

    # =======================
    # Panel 2: Zoom near crossover
    # =======================
    FIGSIZE = (1, 1)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.fill_between(sizes, mins, maxs, color='grey', alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(sizes, means, color='grey', zorder=1, linewidth=2)
    if sdm_sizes is not None:
        ax.plot(sdm_sizes, sdm_costs, color='#eab308', zorder=1, linewidth=2)
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2)

    if crossover_x is not None:
        zoom_xmin = max(0, crossover_x - 50)
        zoom_xmax = crossover_x + 50
        zoom_ymax = crossover_y * 1.5
        ax.set_xlim(zoom_xmin, zoom_xmax)
        ax.set_ylim(0, zoom_ymax)
    else:
        ax.set_xlim(0, 200)
        # Safe default for y-limit
        default_ylim = max(means[:200]) * 1.2 if len(means) > 200 else (max(means) * 1.2 if len(means) > 0 else 10)
        ax.set_ylim(0, default_ylim)

    ax.tick_params(labelsize=9)
    ax.set_yticklabels([f"${int(y)}" if y >= 1 else f"${y:.2f}"
                        for y in ax.get_yticks()])

    # Set facecolor to none
    ax.set_facecolor('none')

    zoom_path = os.path.join(plot_export_dir, f"Cost_per_variant_{frag_len}bp_zoom.pdf")
    plt.savefig(zoom_path, bbox_inches='tight', transparent=True)

    plt.show()
    plt.close(fig)

    print(f"Saved:\n - Full: {full_path}\n - Zoom: {zoom_path}")

def _draw_cost_panel(ax, frag_len,
                     ref_label, ref_sizes, ref_mins, ref_means, ref_maxs, ref_color,
                     usort_sizes, usort_costs,
                     fold_savings_lib_size=2000,
                     show_inset=True,
                     xmax=2500,
                     show_xlabel=True,
                     show_ylabel=False,
                     show_legend=False,
                     ):
    """Draw a single cost-vs-library-size panel.

    ref_* describes the reference comparator (commercial gene fragments or SDM).
    If ref_mins/ref_maxs are None the comparator is plotted as a single line
    (no shaded min/max range) — used for SDM where there's a single estimate.
    """
    has_range = ref_mins is not None and ref_maxs is not None

    if has_range:
        ax.fill_between(ref_sizes, ref_mins, ref_maxs,
                        color=ref_color, alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(ref_sizes, ref_means, color=ref_color, zorder=1, linewidth=2,
            label=ref_label)
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2,
            label="uSort-M")

    # Crossover (reference mean vs uSort-M)
    crossover_x = crossover_y = None
    if len(ref_sizes) > 1 and len(usort_sizes) > 1:
        ref_interp = np.interp(usort_sizes, ref_sizes, ref_means)
        diff = ref_interp - usort_costs
        signs = np.sign(diff)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        if sign_changes.size > 0:
            i = sign_changes[0]
            x0, x1 = usort_sizes[i], usort_sizes[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            if (y1 - y0) != 0:
                crossover_x = float(x0 - y0 * (x1 - x0) / (y1 - y0))
                crossover_y = float(np.interp(crossover_x, usort_sizes, usort_costs))

    if crossover_x is not None:
        ax.scatter([crossover_x], [crossover_y], s=20, color='black', zorder=3)

    ax.set_xlim(0, xmax)
    if has_range and len(ref_maxs) > 0:
        ymax = float(np.nanmax(ref_maxs)) * 0.85
    else:
        ymax = float(np.nanmax(ref_means)) * 1.2 if len(ref_means) else 1.0
    ax.set_ylim(0, ymax)

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_yticklabels([f"${int(y/1000)}k" if y >= 1000 else f"${int(y)}"
                        for y in ax.get_yticks()])
    if show_xlabel:
        ax.set_xlabel("Library Size", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Total Projected Cost (USD)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title(f"{frag_len:,} bp ORF", fontsize=11)

    # Dashed savings annotation
    if (fold_savings_lib_size is not None
            and ref_sizes.min() <= fold_savings_lib_size <= ref_sizes.max()
            and usort_sizes.min() <= fold_savings_lib_size <= usort_sizes.max()):
        ref_y = float(np.interp(fold_savings_lib_size, ref_sizes, ref_means))
        u_y = float(np.interp(fold_savings_lib_size, usort_sizes, usort_costs))
        if u_y > 0 and ref_y > u_y:
            fold = ref_y / u_y
            ax.plot([fold_savings_lib_size, fold_savings_lib_size], [u_y, ref_y],
                    color='black', linestyle='--', linewidth=1, zorder=2)
            ax.scatter([fold_savings_lib_size, fold_savings_lib_size], [u_y, ref_y],
                       color='black', s=6, zorder=3)
            label_x = fold_savings_lib_size + xmax * 0.015
            ax.text(label_x, ref_y + (ymax * 0.02),
                    f"{fold:.1f}-fold savings\n@{fold_savings_lib_size:,}",
                    va='bottom', ha='left', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="none", alpha=0.85))

    # Inset: zoom near crossover
    if show_inset and crossover_x is not None and crossover_y is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axin = inset_axes(ax, width="35%", height="40%", loc='upper left',
                          borderpad=1.0)
        if has_range:
            axin.fill_between(ref_sizes, ref_mins, ref_maxs,
                              color=ref_color, alpha=0.3, edgecolor='none')
        axin.plot(ref_sizes, ref_means, color=ref_color, linewidth=1.2)
        axin.plot(usort_sizes, usort_costs, color='#4ba5e2', linewidth=1.2)
        axin.scatter([crossover_x], [crossover_y], s=12, color='black', zorder=3)
        axin.annotate(f"{int(round(crossover_x))} seq",
                      xy=(crossover_x, crossover_y),
                      xytext=(crossover_x * 1.25, crossover_y * 0.45),
                      fontsize=8, ha='left')
        zoom_xmax = max(50, crossover_x * 2.0)
        zoom_ymax = max(crossover_y * 3.0, 1.0)
        axin.set_xlim(0, zoom_xmax)
        axin.set_ylim(0, zoom_ymax)
        axin.tick_params(labelsize=7)
        axin.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda x, _: f"${int(x/1000)}k" if x >= 1000 else f"${int(x)}"))
        axin.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        for spine in axin.spines.values():
            spine.set_linewidth(0.6)

    if show_legend:
        ax.legend(fontsize=8, loc='lower right', frameon=False)


def _extract_total_curve(df, frag_len):
    """Return (sizes, costs) arrays from a Total-cost dataframe filtered to frag_len."""
    sub = df[(df['Length'] == frag_len) & (df['Step'] == 'Total')]
    sizes = np.array(sorted(sub['Library Size'].unique()))
    costs = np.array([sub[sub['Library Size'] == s]['Cost'].values[0] for s in sizes])
    return sizes, costs


def _extract_total_stats(df, frag_len):
    """Return (sizes, mins, means, maxs) from a multi-vendor Total dataframe."""
    sub = df[(df['Length'] == frag_len) & (df['Step'] == 'Total')]
    grouped = sub.groupby('Library Size')['Cost'].agg(['min', 'mean', 'max'])
    sizes = np.array(grouped.index)
    mins = np.array(grouped['min'])
    means = np.array(grouped['mean'])
    maxs = np.array(grouped['max'])
    valid = np.isfinite(mins) & np.isfinite(means) & np.isfinite(maxs)
    return sizes[valid], mins[valid], means[valid], maxs[valid]


def plot_cost_grid_2row(
    frag_lens,
    tiled_sdm_df,
    tiled_usortm_df,
    full_commercial_df,
    full_usortm_df,
    plot_export_dir,
    fold_savings_lib_size=2000,
    filename="SI_cost_grid_2row.pdf",
    xmax=2500,
    panel_subtitles=None,
):
    """Generate the SI Figure S1 grid: tiled (row 1) over total-synthesis (row 2).

    Row 1 (Tiled assembly):  uSort-M priced via 30 bp insert pooled synthesis
                             vs SDM (single yellow line).
    Row 2 (Total synthesis): uSort-M priced via full-length pooled synthesis
                             vs commercial gene fragments (shaded grey range).

    Y-axes are shared per row. One y-label appears on the leftmost column.
    """
    n_cols = len(frag_lens)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(2.6 * n_cols, 5.0), dpi=150, sharey='row',
    )

    panel_subtitles = panel_subtitles or {}

    for col, frag_len in enumerate(frag_lens):
        # Row 1 (top): Tiled assembly — SDM vs uSort-M (30 bp insert)
        sdm_sub = tiled_sdm_df[(tiled_sdm_df['Length'] == frag_len) &
                               (tiled_sdm_df['Step'] == 'Total')]
        sdm_sizes = np.array(sorted(sdm_sub['Library Size'].unique()))
        sdm_costs = np.array([sdm_sub[sdm_sub['Library Size'] == s]['Cost'].values[0]
                              for s in sdm_sizes])
        u1_sizes, u1_costs = _extract_total_curve(tiled_usortm_df, frag_len)
        _draw_cost_panel(
            axes[0, col], frag_len,
            ref_label="SDM",
            ref_sizes=sdm_sizes, ref_mins=None, ref_means=sdm_costs, ref_maxs=None,
            ref_color='#eab308',
            usort_sizes=u1_sizes, usort_costs=u1_costs,
            fold_savings_lib_size=fold_savings_lib_size,
            xmax=xmax,
            show_xlabel=False,
            show_ylabel=(col == 0),
            show_legend=(col == n_cols - 1),
        )

        # Row 2 (bottom): Total synthesis — commercial vs uSort-M (full gene)
        com_sizes, com_mins, com_means, com_maxs = _extract_total_stats(full_commercial_df, frag_len)
        u2_sizes, u2_costs = _extract_total_curve(full_usortm_df, frag_len)
        _draw_cost_panel(
            axes[1, col], frag_len,
            ref_label="Commercial\nGene Fragments",
            ref_sizes=com_sizes, ref_mins=com_mins, ref_means=com_means, ref_maxs=com_maxs,
            ref_color='grey',
            usort_sizes=u2_sizes, usort_costs=u2_costs,
            fold_savings_lib_size=fold_savings_lib_size,
            xmax=xmax,
            show_xlabel=True,
            show_ylabel=(col == 0),
            show_legend=(col == n_cols - 1),
        )

        # Optional per-column subtitle (e.g. synthesis-platform note) —
        # appended to the panel title as a small second line.
        if frag_len in panel_subtitles:
            for row in (0, 1):
                t = axes[row, col]
                base = t.get_title()
                t.set_title(
                    f"{base}\n({panel_subtitles[frag_len]})",
                    fontsize=10, pad=4,
                )

    # Per-row mode label on the left margin
    fig.text(0.005, 0.78, "Tiled assembly", rotation=90,
             va='center', ha='left', fontsize=11, fontweight='bold')
    fig.text(0.005, 0.30, "Total synthesis", rotation=90,
             va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    os.makedirs(plot_export_dir, exist_ok=True)
    out_path = os.path.join(plot_export_dir, filename)
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_cost_panel(
    series,
    plot_export_dir,
    filename,
    title=None,
    subtitle=None,
    fold_savings_lib_size=None,
    fold_savings_baseline=None,
    fold_savings_target=None,
    xmax=2500,
    figsize=(3.4, 3.4),
    dpi=200,
    xlabel="Library Size",
    ylabel="Total Projected Cost (USD)",
    legend_loc="lower right",
    square_axes=True,
):
    """Single-panel cost-vs-library-size plot, designed for figure assembly in Illustrator.

    Each entry in ``series`` is a dict describing one trace:

        {
          'name':   'uSort-M (Oligo Pools)',     # legend label
          'sizes':  np.ndarray,                   # x values (library sizes)
          'costs':  np.ndarray,                   # mean / single-line y values
          'color':  '#4ba5e2',
          'mins':   np.ndarray | None,            # optional shaded range lower bound
          'maxs':   np.ndarray | None,            # optional shaded range upper bound
          'linestyle': '-',                       # optional, default solid
          'linewidth': 2,                         # optional
          'fill_alpha': 0.3,                      # optional shaded alpha
        }

    A dashed fold-savings annotation can be drawn between two named series via
    ``fold_savings_baseline`` and ``fold_savings_target`` at ``fold_savings_lib_size``.

    Saves a single transparent PDF to ``plot_export_dir/filename`` and returns the path.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if square_axes:
        ax.set_box_aspect(1)

    by_name = {s['name']: s for s in series}

    for s in series:
        sizes = np.asarray(s['sizes'])
        costs = np.asarray(s['costs'])
        mins = s.get('mins'); maxs = s.get('maxs')
        if mins is not None and maxs is not None:
            ax.fill_between(sizes, np.asarray(mins), np.asarray(maxs),
                            color=s['color'],
                            alpha=s.get('fill_alpha', 0.3),
                            zorder=0, edgecolor='none')
        ax.plot(sizes, costs,
                color=s['color'],
                linewidth=s.get('linewidth', 2),
                linestyle=s.get('linestyle', '-'),
                zorder=1,
                label=s['name'])

    ax.set_xlim(0, xmax)
    # ymax is sized off the highest mean-line value (not the shaded max), with
    # 40 % headroom above so the upper-left of the panel stays clear for an
    # externally placed inset overlay.
    mean_candidates = []
    for s in series:
        mean_candidates.extend(np.asarray(s['costs']).tolist())
    finite_means = [c for c in mean_candidates if np.isfinite(c)]
    if finite_means:
        ax.set_ylim(0, max(finite_means) * 1.4)

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, _: f"${int(y/1000)}k" if y >= 1000 else f"${int(y)}"))

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=17)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=17)
    ax.tick_params(labelsize=16)

    if title:
        if subtitle:
            ax.set_title(f"{title}\n({subtitle})", fontsize=16, pad=6)
        else:
            ax.set_title(title, fontsize=17, pad=6)

    ax.set_facecolor('none')

    # Fold-savings annotation
    if (fold_savings_lib_size is not None
            and fold_savings_baseline in by_name
            and fold_savings_target in by_name):
        b = by_name[fold_savings_baseline]
        t = by_name[fold_savings_target]
        b_sizes = np.asarray(b['sizes']); b_costs = np.asarray(b['costs'])
        t_sizes = np.asarray(t['sizes']); t_costs = np.asarray(t['costs'])
        if (b_sizes.min() <= fold_savings_lib_size <= b_sizes.max()
                and t_sizes.min() <= fold_savings_lib_size <= t_sizes.max()):
            b_y = float(np.interp(fold_savings_lib_size, b_sizes, b_costs))
            t_y = float(np.interp(fold_savings_lib_size, t_sizes, t_costs))
            if t_y > 0 and b_y > t_y:
                fold = b_y / t_y
                ax.plot([fold_savings_lib_size, fold_savings_lib_size], [t_y, b_y],
                        color='black', linestyle='--', linewidth=1, zorder=2)
                ax.scatter([fold_savings_lib_size, fold_savings_lib_size], [t_y, b_y],
                           color='black', s=8, zorder=3)

                # Always right-justified, at a fixed (tight) horizontal
                # distance from the dashed vline. Vertical position is chosen
                # dynamically *within* the dashed-vline span (between target
                # and baseline curves) so the label tracks the savings region.
                # The label's *bottom* is anchored just above the lower curve
                # of the largest fitting gap (va='bottom'), guaranteeing no
                # curve overlap even with multi-line labels.
                label_x = fold_savings_lib_size - xmax * 0.015

                y_low = min(t_y, b_y)
                y_high = max(t_y, b_y)
                curve_ys = [y_low, y_high]
                for s in series:
                    sx = np.asarray(s['sizes']); sy = np.asarray(s['costs'])
                    if sx.size and sx.min() <= label_x <= sx.max():
                        cy = float(np.interp(label_x, sx, sy))
                        if y_low <= cy <= y_high:
                            curve_ys.append(cy)
                curve_ys = sorted(set(curve_ys))

                # Estimate total label height in data coordinates so we can
                # require gaps to fit the text without crashing into either
                # bounding curve.
                _y_range = y_high - y_low if y_high > y_low else 1.0
                _full_y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                _line_h_pts = 14 * 1.25  # fontsize * line spacing
                _line_h_data = (_line_h_pts / 72.0 * dpi) / (figsize[1] * dpi * 0.78) * _full_y_range
                _label_lines = 2  # "X-fold\n@ N"
                _text_h_data = _line_h_data * _label_lines
                _pad = _line_h_data * 0.25

                # Pick the LARGEST gap that can comfortably fit the text.
                best_gap = 0.0
                best_bottom = y_low + _pad
                for i in range(len(curve_ys) - 1):
                    gap = curve_ys[i + 1] - curve_ys[i]
                    if gap >= (_text_h_data + 2 * _pad) and gap > best_gap:
                        best_gap = gap
                        best_bottom = curve_ys[i] + _pad
                # If no gap can fit, fall back to the largest gap and center
                # the label within it (last-resort placement).
                if best_gap == 0.0:
                    for i in range(len(curve_ys) - 1):
                        gap = curve_ys[i + 1] - curve_ys[i]
                        if gap > best_gap:
                            best_gap = gap
                            best_bottom = curve_ys[i] + max(0.0, (gap - _text_h_data) / 2)

                ax.text(label_x, best_bottom,
                        f"{fold:.1f}-fold\n@ {fold_savings_lib_size:,}",
                        va='bottom', ha='right', fontsize=14)

    if legend_loc:
        ax.legend(fontsize=14, loc=legend_loc, frameon=False)

    os.makedirs(plot_export_dir, exist_ok=True)
    out_path = os.path.join(plot_export_dir, filename)
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    return out_path


def _find_crossover(b_x, b_y, t_x, t_y):
    """Return (cx, cy) where baseline-mean crosses target curve, or (None, None)."""
    b_interp = np.interp(t_x, b_x, b_y)
    diff = b_interp - t_y
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if sign_changes.size == 0:
        return None, None
    i = sign_changes[0]
    x0, x1 = t_x[i], t_x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    if (y1 - y0) == 0:
        return None, None
    cx = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    cy = float(np.interp(cx, t_x, t_y))
    return cx, cy


def plot_cost_inset(
    series,
    plot_export_dir,
    filename,
    crossover_baseline=None,
    crossover_target=None,
    crossovers=None,
    figsize=(1.2, 1.2),
    dpi=200,
    margin_frac=1.0,    # how far (in units of largest crossover_x) to extend xlim
    yscale=1.3,         # ylim multiplier on largest crossover_y
    square_axes=True,
    label_fontsize=16,
    tick_fontsize=18,
):
    """Standalone inset PDF — zoomed view near one or more crossovers of named series.

    Two ways to specify crossovers:
      • single pair via ``crossover_baseline`` + ``crossover_target`` (back-compat)
      • a list via ``crossovers=[{'baseline': ..., 'target': ...,
                                  'marker_color': 'black',
                                  'label': '@N seq' (optional)}, ...]``

    Marker is filled with ``marker_color`` (default black). Saves a transparent PDF.
    Returns the path or ``None`` if no crossover is found in the data range.
    """
    if crossovers is None:
        if crossover_baseline is None or crossover_target is None:
            return None
        crossovers = [{
            'baseline': crossover_baseline,
            'target': crossover_target,
            'marker_color': 'black',
        }]

    by_name = {s['name']: s for s in series}

    found = []
    for spec in crossovers:
        b_name = spec['baseline']; t_name = spec['target']
        if b_name not in by_name or t_name not in by_name:
            continue
        b = by_name[b_name]; t = by_name[t_name]
        cx, cy = _find_crossover(np.asarray(b['sizes']), np.asarray(b['costs']),
                                 np.asarray(t['sizes']), np.asarray(t['costs']))
        if cx is None:
            continue
        found.append({
            'baseline': b_name, 'target': t_name,
            'marker_color': spec.get('marker_color', 'black'),
            'label': spec.get('label', f"{int(round(cx))} seq"),
            'cx': cx, 'cy': cy,
        })

    if not found:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if square_axes:
        ax.set_box_aspect(1)

    for s in series:
        sx = np.asarray(s['sizes']); sy = np.asarray(s['costs'])
        mins = s.get('mins'); maxs = s.get('maxs')
        if mins is not None and maxs is not None:
            ax.fill_between(sx, np.asarray(mins), np.asarray(maxs),
                            color=s['color'],
                            alpha=s.get('fill_alpha', 0.3),
                            zorder=0, edgecolor='none')
        ax.plot(sx, sy, color=s['color'],
                linewidth=s.get('linewidth', 1.8),
                linestyle=s.get('linestyle', '-'),
                zorder=1)

    # Preserve the caller's order: the FIRST crossover in `crossovers`
    # gets the upper label slot (k=0), the LAST gets the bottom (k=last).
    cxs = [f['cx'] for f in found]; cys = [f['cy'] for f in found]
    max_cx = max(cxs); max_cy = max(cys)

    # All labels go BELOW their markers, in the empty region beneath the lowest
    # curve (uSort-M). Each marker sits on uSort-M, so any y below the marker is
    # guaranteed clear of every curve at the same x.
    #
    # Use data coordinates with explicit clamping so labels never escape the
    # axes box — vertical positions are computed against ymax, then bounded.
    zoom_xmax_estimate = max(50.0, max_cx * (1.0 + margin_frac))
    zoom_ymax_estimate = max(max_cy * yscale, 1.0)
    ax.set_xlim(0, zoom_xmax_estimate)
    ax.set_ylim(0, zoom_ymax_estimate)

    # Approximate text height in data coords for the crossover-label fontsize.
    fig_h_inches = figsize[1]
    text_h_pixels = label_fontsize / 72 * dpi   # fontsize → inches → pixels
    axes_h_pixels = fig_h_inches * dpi * 0.8    # axes box ≈ 80 % of figure
    text_h_data = (text_h_pixels / axes_h_pixels) * zoom_ymax_estimate

    # Gap between the marker and the top of the first label (clears the marker
    # glyph itself). Gap between successive labels = one text height + padding.
    gap_first = text_h_data * 0.65
    gap_stagger = text_h_data * 0.95

    # Bottom safety margin so the label stays well above the x-axis.
    bottom_margin = text_h_data * 0.15

    # Force a first draw so the renderer is available for window-extent
    # measurements below. Without this, get_window_extent() would fall back
    # to a default renderer with the wrong dpi, defeating the whole point.
    fig.canvas.draw()
    flip_offset = zoom_xmax_estimate * 0.025
    edge_pad_data = zoom_xmax_estimate * 0.015

    series_arrays = [
        (np.asarray(s['sizes']), np.asarray(s['costs']))
        for s in series
    ]

    def _bbox_data(text_artist):
        bbox_disp = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_data = bbox_disp.transformed(ax.transData.inverted())
        x0, x1 = sorted([float(bbox_data.x0), float(bbox_data.x1)])
        y0, y1 = sorted([float(bbox_data.y0), float(bbox_data.y1)])
        return x0, x1, y0, y1

    def _is_bad(text_artist):
        """True iff the rendered label overflows the axes or crosses a curve."""
        x0, x1, y0, y1 = _bbox_data(text_artist)
        if (x0 < edge_pad_data or
                x1 > zoom_xmax_estimate - edge_pad_data or
                y0 < 0 or
                y1 > zoom_ymax_estimate):
            return True
        # Curve-intersection check: sample each curve in the label's x range
        # and see if any sampled y lands inside the bbox.
        for sx, sy in series_arrays:
            if sx.size < 2 or sx.max() < x0 or sx.min() > x1:
                continue
            xs = np.linspace(max(x0, float(sx.min())),
                             min(x1, float(sx.max())), 24)
            ys = np.interp(xs, sx, sy)
            if np.any((ys >= y0) & (ys <= y1)):
                return True
        return False

    # Track how many labels fall back to the bottom-of-inset slot so we can
    # stack them rather than overlap.
    bottom_fallback_idx = 0

    for k, f in enumerate(found):
        ax.scatter([f['cx']], [f['cy']], s=120,
                   color=f['marker_color'], zorder=3, edgecolors='none')

        # Tentative placement: right of the marker, below it (va='top').
        desired_top = f['cy'] - gap_first - k * gap_stagger
        min_top = bottom_margin + text_h_data
        label_y = max(min_top, desired_top)

        original_text = f['label']
        wrapped_text = original_text.replace(' ', '\n', 1) if ' ' in original_text else original_text

        text_artist = ax.text(
            f['cx'] + flip_offset, label_y, original_text,
            fontsize=label_fontsize, ha='left', va='top',
            color=f['marker_color'], clip_on=True,
        )

        def _try_placements():
            # Returns True if a valid (no-overlap, in-bounds) placement was found.
            for variant in (original_text, wrapped_text):
                text_artist.set_text(variant)
                # Try right of marker
                text_artist.set_x(f['cx'] + flip_offset)
                text_artist.set_ha('left')
                if not _is_bad(text_artist):
                    return True
                # Try left of marker
                text_artist.set_x(f['cx'] - flip_offset)
                text_artist.set_ha('right')
                if not _is_bad(text_artist):
                    return True
                if variant is wrapped_text:
                    return False  # both variants × both sides exhausted
            return False

        if not _try_placements():
            # Bottom-of-inset fallback: center horizontally, stack vertically.
            text_artist.set_text(original_text)
            text_artist.set_x(zoom_xmax_estimate * 0.5)
            text_artist.set_ha('center')
            line_h = text_h_data / 2.0
            text_artist.set_y(
                bottom_margin + line_h * 1.2 + bottom_fallback_idx * line_h * 1.4
            )
            text_artist.set_va('bottom')
            bottom_fallback_idx += 1

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, _: f"${int(y/1000)}k" if y >= 1000 else f"${int(y)}"))
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_facecolor('none')

    # Faint white bg behind tick labels so they remain legible if a curve or
    # marker happens to render directly behind them.
    tick_bbox = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.6)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_bbox(tick_bbox)

    os.makedirs(plot_export_dir, exist_ok=True)
    out_path = os.path.join(plot_export_dir, filename)
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    return out_path
