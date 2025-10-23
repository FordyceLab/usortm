import os
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_cost_singleFrag_varyLib(frag_len, 
                                 cost_stats, 
                                 usortm_cost_dict, 
                                 plot_export_dir
                                 ):

    # =======================
    # Format costs
    # =======================

    # --- Select a fragment size to plot ---
    frag_stats = cost_stats[frag_len]

    # --- Extract parsed synthesis costs ---
    sizes = np.array(sorted(frag_stats.keys()))
    mins  = np.array([frag_stats[s]['min'] for s in sizes])
    means = np.array([frag_stats[s]['mean'] for s in sizes])
    maxs  = np.array([frag_stats[s]['max'] for s in sizes])

    # --- Get uSort-M costs ---
    usort_sizes = np.array(sorted(usortm_cost_dict[frag_len].keys()))
    usort_costs = np.array([usortm_cost_dict[frag_len][s] for s in usort_sizes])

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
    DPI = 300

    # =======================
    # Panel 1: Full range
    # =======================
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.fill_between(sizes, mins, maxs, color='grey', alpha=0.3, zorder=0, edgecolor='none')
    ax.plot(sizes, means, color='grey', zorder=1, linewidth=2, label="Commercial\nGene Fragments")
    ax.plot(usort_sizes, usort_costs, color='#4ba5e2', zorder=1, linewidth=2, label="uSort-M")

    ax.set_xlim(0, 2200)
    # ax.set_ylim(-1000, 120000)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels([f"{x:,}" if x != 0 else f"{int(x)}" for x in ax.get_xticks()])
    ax.set_yticklabels([f"${int(x/1000)}k" if x != 0 else f"${int(x)}" for x in ax.get_yticks()])
    ax.set_xlabel(f"Library Size", fontsize=12)
    ax.set_ylabel("Total Projected Cost (USD)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(f"{frag_len:,} bp fragments")

    # Annotate crossover point
    if crossover_x is not None:
        ax.scatter([crossover_x], [crossover_y], s=20, color='black', zorder=3)

    # --- Add dashed line + savings annotation at 1000 library size ---
    lib_target = 1000
    if lib_target in sizes and lib_target in usort_sizes:
        grey_y = np.interp(lib_target, sizes, means)
        blue_y = np.interp(lib_target, usort_sizes, usort_costs)
        fold_savings = grey_y / blue_y

        # Dashed connector line
        ax.plot([lib_target, lib_target], [blue_y, grey_y],
                color='black', linestyle=(0, (3, 3)), linewidth=1, zorder=2)

        # Arrows and label
        ax.text(lib_target * 1.05, (grey_y + blue_y) / 2,
                f"{fold_savings:.1f}-fold savings\n@{lib_target:,}",
                va='center', ha='left', fontsize=8)

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