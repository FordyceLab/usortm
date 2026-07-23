#!/usr/bin/env python3
"""Pre-compute coverage curves for the sorting.html interactive."""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from usortm.simulate.sortm import simulate_coverage_curve

LIB_SIZES = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
SKEWS = [2, 3, 4, 5, 6, 8]
FOLD_SAMPLINGS = np.linspace(0, 12, 49)  # 0 to 12x in 0.25 steps


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'sort_data')
    os.makedirs(output_dir, exist_ok=True)

    total = len(LIB_SIZES) * len(SKEWS)
    done = 0
    all_curves = {}
    for lib_size in LIB_SIZES:
        all_curves[str(lib_size)] = {}
        for skew in SKEWS:
            done += 1
            print(f"[{done}/{total}] lib_size={lib_size}, skew={skew}x ...", end=' ', flush=True)
            df = simulate_coverage_curve(
                fold_samplings=FOLD_SAMPLINGS,
                lib_size=lib_size,
                skew=skew,
                n_sims=25,
                pbar=False,
            )
            # df has a multi-index; reset to access fold-sampling column
            df = df.reset_index(drop=True)
            grouped = df.groupby('fold-sampling')['coverage'].agg(
                mean='mean',
                p10=lambda x: np.percentile(x, 10),
                p90=lambda x: np.percentile(x, 90),
            ).reset_index()
            all_curves[str(lib_size)][str(skew)] = [
                {
                    'fold': round(float(row['fold-sampling']), 4),
                    'mean': round(float(row['mean']), 4),
                    'p10': round(float(row['p10']), 4),
                    'p90': round(float(row['p90']), 4),
                }
                for _, row in grouped.iterrows()
            ]
            print(f"done ({len(grouped)} points)")

    with open(os.path.join(output_dir, 'coverage_curves.json'), 'w') as f:
        json.dump(all_curves, f)
    print("✓ Generated sort_data/coverage_curves.json")


if __name__ == '__main__':
    main()
