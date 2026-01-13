import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_total_cost_singleFrag_varyLib(frag_len,
                                 commercial_costs_df,
                                 usortm_costs_df,
                                 plot_export_dir,
                                 fold_savings_lib_size=1000
                                 ):
    """Plot total cost comparison between commercial and uSort-M methods.

    Creates two plots: a full-range comparison and a zoomed view near the crossover point.

    Args:
        frag_len: Fragment/sequence length (bp) to plot.
        commercial_costs_df: DataFrame from generate_commercial_cost_dict().
        usortm_costs_df: DataFrame from get_usortm_costs().
        plot_export_dir: Directory path to save output plots.
        fold_savings_lib_size: Library size at which to annotate fold savings (default: 1000).

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
                                              fold_savings_lib_size=1000
                                              ):
    """Plot cost per variant comparison between commercial and uSort-M methods.

    Creates two plots: a full-range comparison and a zoomed view near the crossover point.

    Args:
        frag_len: Fragment/sequence length (bp) to plot.
        commercial_costs_df: DataFrame from generate_commercial_cost_dict().
        usortm_costs_df: DataFrame from get_usortm_costs().
        plot_export_dir: Directory path to save output plots.
        fold_savings_lib_size: Library size at which to annotate fold savings (default: 1000).

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