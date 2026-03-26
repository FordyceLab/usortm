#!/usr/bin/env python3
"""
Generate cost data JSON for the interactive plot from Python cost functions.
This ensures the web plot uses the exact same calculations as the Python package.
"""

import json
import sys
import os

# Add src to path to import usortm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from usortm.costs.cost_functions import (
    usortm_synthesis_cost,
    usortm_cloning_cost,
    usortm_sorting_cost,
    usortm_barcoding_cost,
    usortm_sequencing_cost,
    usortm_hitpicking_cost,
    parsed_genefragments_synthesis_cost,
    parsed_genefragments_assembly_cost,
    parsed_genefragments_barcoding_cost,
    parsed_genefragments_sequencing_cost,
    sdm_total_cost,
    sdm_primer_cost,
    sdm_kit_cost,
    sdm_transformation_cost,
    sdm_consumables_cost,
)
from usortm.costs.time_functions import calculate_total_timeline


def calculate_usortm_cost(lib_size, seq_length):
    """Calculate total uSort-M cost with 4x fold sampling."""
    foldSampling = 4
    wells = lib_size * foldSampling

    cost = usortm_synthesis_cost(lib_size, seq_length)
    cost += usortm_cloning_cost(lib_size)
    cost += usortm_sorting_cost(lib_size, fold_sampling=foldSampling)
    cost += usortm_barcoding_cost(n_wells=wells)
    cost += usortm_sequencing_cost(n_wells=wells, seq_length=seq_length)
    cost += usortm_hitpicking_cost(lib_size, seq_length)
    return cost


def calculate_traditional_cost(lib_size, seq_length):
    """Calculate traditional (parsed gene fragments) cost."""
    # Use eBlocks as baseline (most cost-effective)
    synth_cost = parsed_genefragments_synthesis_cost(seq_length, lib_size, 'idt_eblocks')
    assembly_cost = parsed_genefragments_assembly_cost(lib_size, 'hifi')
    barcode_cost = parsed_genefragments_barcoding_cost(lib_size)
    seq_cost = parsed_genefragments_sequencing_cost(seq_length, lib_size)

    return synth_cost + assembly_cost + barcode_cost + seq_cost


def calculate_traditional_range(lib_size, seq_length):
    """Calculate min/max range for traditional methods."""
    # Min: eBlocks
    min_cost = parsed_genefragments_synthesis_cost(seq_length, lib_size, 'idt_eblocks')
    # Max: gBlocks
    max_cost = parsed_genefragments_synthesis_cost(seq_length, lib_size, 'idt_gblocks')

    # Add common costs
    common = (parsed_genefragments_assembly_cost(lib_size, 'hifi') +
              parsed_genefragments_barcoding_cost(lib_size) +
              parsed_genefragments_sequencing_cost(seq_length, lib_size))

    return min_cost + common, max_cost + common


def generate_cost_curves(seq_length, max_lib_size=5000, step=25):
    """Generate cost curves for plotting."""
    data = []

    for lib_size in range(50, max_lib_size + 1, step):
        usortm_cost = calculate_usortm_cost(lib_size, seq_length)
        trad_cost = calculate_traditional_cost(lib_size, seq_length)
        trad_min, trad_max = calculate_traditional_range(lib_size, seq_length)

        sdm_cost = sdm_total_cost(lib_size, seq_length, include_hifi=False)
        sdm_cost_max = sdm_total_cost(lib_size, seq_length, include_hifi=True)

        data.append({
            'library_size': lib_size,
            'usortm_cost': round(usortm_cost, 2),
            'traditional_cost': round(trad_cost, 2),
            'traditional_min': round(trad_min, 2),
            'traditional_max': round(trad_max, 2),
            'sdm_cost': round(sdm_cost, 2),
            'sdm_cost_max': round(sdm_cost_max, 2),
        })

    return data


def generate_detailed_costs(lib_size, seq_length):
    """Generate detailed breakdown of costs for a specific configuration."""
    # Calculate derived values first (needed for cost functions)
    foldSampling = 4
    wells = lib_size * foldSampling
    plates = max(1, -(-wells // 384))  # Ceiling division

    # uSort-M breakdown
    usortm_breakdown = {
        'synthesis': usortm_synthesis_cost(lib_size, seq_length),
        'cloning': usortm_cloning_cost(lib_size),
        'sorting': usortm_sorting_cost(lib_size, fold_sampling=foldSampling),
        'barcoding': usortm_barcoding_cost(n_wells=wells),
        'sequencing': usortm_sequencing_cost(n_wells=wells, seq_length=seq_length),
        'hitpicking': usortm_hitpicking_cost(lib_size, seq_length),
    }
    usortm_breakdown['total'] = sum(usortm_breakdown.values())

    # Traditional breakdown
    trad_breakdown = {
        'synthesis': parsed_genefragments_synthesis_cost(seq_length, lib_size, 'idt_eblocks'),
        'assembly': parsed_genefragments_assembly_cost(lib_size, 'hifi'),
        'barcoding': parsed_genefragments_barcoding_cost(lib_size),
        'sequencing': parsed_genefragments_sequencing_cost(seq_length, lib_size),
    }
    trad_breakdown['total'] = sum(trad_breakdown.values())

    # Calculate timeline
    timeline = calculate_total_timeline(lib_size, seq_length, fold_sampling=foldSampling)

    # SDM breakdown
    sdm_breakdown = {
        'primers': sdm_primer_cost(lib_size),
        'q5_sdm_kit': sdm_kit_cost(lib_size, include_hifi=False),
        'transformation': sdm_transformation_cost(lib_size),
        'consumables': sdm_consumables_cost(lib_size),
        'sequencing': parsed_genefragments_sequencing_cost(seq_length, lib_size),
    }
    sdm_breakdown['total'] = sum(sdm_breakdown.values())

    return {
        'usortm': {k: round(v, 2) for k, v in usortm_breakdown.items()},
        'traditional': {k: round(v, 2) for k, v in trad_breakdown.items()},
        'sdm': {k: round(v, 2) for k, v in sdm_breakdown.items()},
        'savings': round(trad_breakdown['total'] / usortm_breakdown['total'], 2),
        'per_variant_usortm': round(usortm_breakdown['total'] / lib_size, 2),
        'per_variant_traditional': round(trad_breakdown['total'] / lib_size, 2),
        'per_variant_sdm': round(sdm_breakdown['total'] / lib_size, 2),
        'wells': wells,
        'plates': plates,
        'timeline': timeline,
    }


def main():
    """Generate JSON files for common configurations."""
    output_dir = os.path.join(os.path.dirname(__file__), 'cost_data')
    os.makedirs(output_dir, exist_ok=True)

    # Generate curves for all sequence lengths from 100-1500 bp in 50 bp steps
    seq_lengths = list(range(100, 1501, 50))

    for seq_len in seq_lengths:
        curve_data = generate_cost_curves(seq_len)
        filename = f'cost_curve_{seq_len}bp.json'

        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(curve_data, f, indent=2)

        print(f"✓ Generated {filename}")

    # Generate detailed costs for default configuration
    default_config = generate_detailed_costs(500, 300)

    with open(os.path.join(output_dir, 'default_costs.json'), 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"✓ Generated default_costs.json")
    print(f"\n✓ All cost data generated in {output_dir}/")


if __name__ == '__main__':
    main()
