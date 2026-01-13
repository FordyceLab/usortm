import os
import math
import numpy as np
import pandas as pd

def parsed_genefragments_synthesis_cost(length, fragment_number, method):
    
    if method == 'idt_eblocks':
        if (length > 300) and (length <= 500):
            return fragment_number * 35
        else:
            return length * fragment_number * 0.07
    
    elif method == 'idt_gblocks':
        return length * fragment_number * 0.09
    
    elif method == 'twist_genefragments':
        if length < 300:
            return np.nan  # Return NaN instead of None
        elif 300 <= length <= 500:
            return fragment_number * 35
        else:
            return length * fragment_number * 0.07
    
    return np.nan  # Default return for unknown methods
    
def parsed_genefragments_assembly_cost(library_size, assembly_method):
    cost = 0

    # --- Assembly ---
    if assembly_method == 'hifi':
        # $2,680 for 250 reactions
        per_rxn = 2680 / 250
        cost += library_size*per_rxn

    elif assembly_method == 'goldengate':
        # $474.00 for 100 reactions
        per_rxn = 474 / 100
        cost += library_size*per_rxn

    else:
        raise ValueError(f"Unknown assembly method: {assembly_method}")
    
    # --- Transformation ---
    # Cost of NEB 5-alpha:
    # $165 for 6x 200 µL tubes
    neb_total = 165
    per_uL_cells = neb_total/(6*200)
    cost += per_uL_cells * 10 * library_size # assuming 10 µL transformation volume

    return cost

def parsed_genefragments_barcoding_cost(library_size):
    # Assume 8x sorting
    n_plates = library_size/384 # Get number of 384-well plates
    return n_plates*97.73 # From cost sheet

def parsed_genefragments_sequencing_cost(fragment_length, library_size):
    # Base cost of Plasmidsaurus Custom Sequencing
    cost = 500

    # 100 minimum reads per well
    total_reads = library_size*100

    # ASSUMING READ LENGTH IS CDS + 100 BASES FOR BARCODES
    total_bp = total_reads*(fragment_length+100)
    target_Gb = total_bp/1000000000

    if target_Gb > 1:
        cost+=50

    return cost

def generate_commercial_costs(fragment_sizes, library_sizes, assembly_method, steps=None):
    """Tabulate all commercial gene fragment synthesis costs.

    Calculates costs for commercial gene fragment synthesis from vendors (Twist, IDT)
    including synthesis, assembly, barcoding, and sequencing steps.

    Args:
        fragment_sizes: List of fragment sizes (bp) to evaluate.
        library_sizes: List of library sizes (number of variants) to evaluate.
        assembly_method: Assembly method to use ('hifi' or 'goldengate').
        steps: List of cost steps to include. Options: 'synthesis', 'assembly',
               'barcoding', 'sequencing'. If None, includes all steps.

    Returns:
        pandas DataFrame with columns:
            - Length: Fragment length (bp)
            - Library Size: Number of variants
            - Vendor: Vendor name (Twist, IDT)
            - Product: Product name (Gene Fragments, eBlocks, gBlocks)
            - Step: Cost step name (Synthesis, Assembly, Barcoding, Sequencing, Total)
            - Cost: Cost in USD
            - CPV: Cost per variant in USD
    """
    if steps is None:
        steps = ['synthesis', 'assembly', 'barcoding', 'sequencing']

    cost_funcs = {
        'synthesis': lambda frag_len, n, vendor: parsed_genefragments_synthesis_cost(frag_len, n, vendor),
        'assembly': lambda frag_len, n, vendor: parsed_genefragments_assembly_cost(n, assembly_method),
        'barcoding': lambda frag_len, n, vendor: parsed_genefragments_barcoding_cost(n),
        'sequencing': lambda frag_len, n, vendor: parsed_genefragments_sequencing_cost(frag_len, n)
    }

    step_display_names = {
        'synthesis': 'Synthesis',
        'assembly': 'Assembly',
        'barcoding': 'Barcoding',
        'sequencing': 'Sequencing'
    }

    records = []
    vendors = [
        ('Twist', 'Gene Fragments', 'twist_genefragments'),
        ('IDT', 'eBlocks', 'idt_eblocks'),
        ('IDT', 'gBlocks', 'idt_gblocks')
    ]

    for frag_len in fragment_sizes:
        for n in library_sizes:
            for vendor, product, vendor_key in vendors:
                step_costs = []
                for step in steps:
                    if step in cost_funcs:
                        cost = cost_funcs[step](frag_len, n, vendor_key)
                        # Handle NaN values
                        if cost is None or (isinstance(cost, float) and np.isnan(cost)):
                            continue  # Skip this record if cost is NaN
                        cost_int = int(cost)
                        cpv = cost_int / n if n > 0 else 0
                        records.append({
                            "Length": int(frag_len),
                            "Library Size": int(n),
                            "Vendor": vendor,
                            "Product": product,
                            "Step": step_display_names[step],
                            "Cost": cost_int,
                            "CPV": cpv
                        })
                        step_costs.append(cost_int)

                # Add total row if we had any valid costs
                if step_costs:
                    total_cost = sum(step_costs)
                    total_cpv = total_cost / n if n > 0 else 0
                    records.append({
                        "Length": int(frag_len),
                        "Library Size": int(n),
                        "Vendor": vendor,
                        "Product": product,
                        "Step": "Total",
                        "Cost": total_cost,
                        "CPV": total_cpv
                    })
    return pd.DataFrame(records)

def generate_commercial_cost_stats_dict(commercial_cost_comparison_dict, library_sizes):
    # --- Cost statistics ---
    
    # Handle DataFrame input
    if isinstance(commercial_cost_comparison_dict, pd.DataFrame):
        df = commercial_cost_comparison_dict
        cost_stats = {}
        
        for frag_len in df['Length'].unique():
            stats_for_frag = {}
            frag_df = df[df['Length'] == frag_len]
            
            for n in library_sizes:
                lib_df = frag_df[frag_df['Library Size'] == n]
                # Sum costs by vendor/product combination
                costs = lib_df.groupby(['Vendor', 'Product'])['Cost'].sum().values
                
                # Filter out any NaN or zero values
                costs = [c for c in costs if not np.isnan(c) and c > 0]
                
                if len(costs) > 0:
                    stats_for_frag[n] = {
                        'min': min(costs),
                        'mean': sum(costs) / len(costs),
                        'max': max(costs),
                        'count': len(costs),
                    }
                else:
                    # No valid costs for this combination
                    stats_for_frag[n] = {
                        'min': np.nan,
                        'mean': np.nan,
                        'max': np.nan,
                        'count': 0,
                    }
            cost_stats[frag_len] = stats_for_frag
        
        return cost_stats
    
    # Handle dictionary input (original logic)
    cost_stats = {}
    for frag_len, provider_dict in commercial_cost_comparison_dict.items():
        stats_for_frag = {}
        for n in library_sizes:
            # collect costs across providers & fragment types
            costs = [
                cost_dict[n]
                for provider_data in provider_dict.values()
                for cost_dict in provider_data.values()
            ]
            # Filter out NaN values
            costs = [c for c in costs if not np.isnan(c)]
            
            if len(costs) > 0:
                stats_for_frag[n] = {
                    'min': min(costs),
                    'mean': sum(costs) / len(costs),
                    'max': max(costs),
                    'count': len(costs),
                }
            else:
                stats_for_frag[n] = {
                    'min': np.nan,
                    'mean': np.nan,
                    'max': np.nan,
                    'count': 0,
                }
        cost_stats[frag_len] = stats_for_frag

    return cost_stats

# --- Twist Oligo Pool Pricing Table ---
twist_library_costs = {
    (2, 100):   {120: 400.00, 150: 466.00, 200: 520.00, 250: 689.00, 300: 1030.00},
    (101, 500): {120: 800.00, 150: 933.00, 200: 1040.00, 250: 1379.00, 300: 2060.00},
    (501, 1000): {120: 1200.00, 150: 1400.00, 200: 1560.00, 250: 2068.00, 300: 3090.00},
    (1001, 2000): {120: 1600.00, 150: 1867.00, 200: 2080.00, 250: 2757.00, 300: 4121.00},
    (2001, 6000): {120: 2400.00, 150: 2800.00, 200: 3120.00, 250: 4136.00, 300: 6181.00},
    (6001, 12000): {120: 3120.00, 150: 3744.00, 200: 4056.00, 250: 5148.00, 300: 7694.00},
    (12001, 18000): {120: 4056.00, 150: 4867.00, 200: 5273.00, 250: 6694.00, 300: 10004.00},
}

def usortm_synthesis_cost(n_seqs, 
                          seq_length, 
                          library_costs=twist_library_costs,
                          commercial_discount=True,
                          ):
    """
    Compute synthesis cost (USD) for a pooled oligo library sequences.

    """
    # Check length: Twist for smaller than 300
    if seq_length <= 350:
        # Select appropriate tier
        for (low, high), length_dict in library_costs.items():
            if low <= n_seqs <= high:
                # Pick nearest length bracket
                valid_lengths = sorted(length_dict.keys())
                nearest_len = min(valid_lengths, key=lambda x: abs(x - seq_length))
                if commercial_discount:
                    return length_dict[nearest_len] * (2/3)
                else:
                    return length_dict[nearest_len]
    
    # If larger than 300, use Instance pricing scheme
    else:
        total_bp = seq_length*n_seqs
        if n_seqs < 3000:
            return 0.015*total_bp
        elif (n_seqs > 3000) and (n_seqs < 30000):
            if seq_length < 301:
                return 0.001*total_bp
            elif (seq_length >= 301) and (seq_length <= 549):
                return 0.002*total_bp
            elif (seq_length >= 550) and (seq_length <= 2050):
                return 0.004*total_bp
            else:
                return 0
            

        
    return 0  # outside defined tiers

def usortm_cloning_cost(library_size):
    """
    uSort-M Cloning costs
    """
    cost = 0

    ### --- Assembly ---
    # Cost of HiFi assembly reagents:
    # $2,680.00 for 250 reactions of 2X MM at 10 µL per reaction
    per_rxn = 2680/250
    cost += per_rxn*5 # Assuming one 100 µL reaction with 50 µL 2X MM

    ### --- Transformation ---
    # Cost of NEB 5-alpha:
    # $165 for 6x 200 µL tubes
    neb_total = 165
    per_uL_cells = neb_total/(6*200)
    cost += per_uL_cells * 50

    # TODO: add actual transformation scale calculation
    if library_size > 1000:
        cost += 50

    return per_rxn*5

def usortm_sorting_cost(library_size):
    '''Calculate the sorting cost for a given library size.'''
    cost = 0

    # Assume 8x sorting
    total_wells = library_size*8

    # Get number of 384-well plates
    n_plates = int(total_wells/384)

    # Get total sort time in minutes, assuming 6 minutes per plate
    sort_minutes = n_plates*6

    # Add one hour for setup and cleaning
    sort_minutes += 60

    # Convert to cost
    # Rates defined at: https://facs.stanford.edu/facility-info/policies/proposed-rates-2024-2025
    # Sony SH800Z and BD Aria are both $70/hr
    machine_hourly_rate = 70
    operator_hourly_rate = 65

    # Compute total cost
    total_cost = (sort_minutes/60)*(machine_hourly_rate+operator_hourly_rate)

    return total_cost

def usortm_barcoding_cost(n_wells):
    # Assume 8x sorting
    # total_wells = library_size*8
    total_wells = n_wells
    n_plates = int(n_wells/384) # Get number of 384-well plates
    return n_plates*97.73 # From cost sheet

def usortm_sequencing_cost(n_wells, seq_length):
    # Base cost of Plasmidsaurus Custom Sequencing
    cost = 500

    # 100 minimum reads per well
    total_reads = n_wells*100

    # ASSUMING READ LENGTH IS CDS + 100 BASES FOR BARCODES
    total_bp = total_reads*(seq_length+100)
    target_Gb = total_bp/1000000000

    if target_Gb > 1:
        cost+=50

    return cost

# --- 5) Hitpicking Cost --- 
def usortm_hitpicking_cost(library_size, seq_length):
    cost = 0

    # Cost per tip of Integra GripTip
    cost_per_tip = 0.128
    cost += library_size*cost_per_tip

    # Cost per plate for cherrypicking
    cost_per_plate = 7.84
    plates = math.ceil(library_size/384)
    cost += plates*cost_per_plate

    return cost

def get_usortm_costs(library_sizes, seq_lengths, steps=None):
    """Compute total uSort-M costs for given library sizes and sequence lengths.

    Calculates costs for the uSort-M workflow including oligo synthesis, cloning,
    sorting, barcoding, sequencing, and hit-picking steps.

    Args:
        library_sizes: List of library sizes (number of variants) to evaluate.
        seq_lengths: List of sequence lengths (bp) to evaluate.
        steps: List of cost steps to include. Options: 'synthesis', 'cloning',
               'sorting', 'barcoding', 'sequencing', 'hitpicking'. If None, includes all steps.

    Returns:
        pandas DataFrame with columns:
            - Length: Sequence length (bp)
            - Library Size: Number of variants
            - Step: Cost step name (Synthesis, Cloning, Sorting, Barcoding, Sequencing, Hitpicking, Total)
            - Cost: Cost in USD
            - CPV: Cost per variant in USD
    """
    if steps is None:
        steps = ["synthesis", "cloning", "sorting", "barcoding", "sequencing", "hitpicking"]

    cost_funcs = {
        "synthesis": lambda lib_size, seq_length: usortm_synthesis_cost(lib_size, seq_length),
        "cloning": lambda lib_size, seq_length: usortm_cloning_cost(lib_size),
        "sorting": lambda lib_size, seq_length: usortm_sorting_cost(lib_size),
        "barcoding": lambda lib_size, seq_length: usortm_barcoding_cost(n_wells=lib_size*4),
        "sequencing": lambda lib_size, seq_length: usortm_sequencing_cost(n_wells=lib_size*4, seq_length=seq_length),
        "hitpicking": lambda lib_size, seq_length: usortm_hitpicking_cost(lib_size, seq_length),
    }

    step_display_names = {
        "synthesis": "Synthesis",
        "cloning": "Cloning",
        "sorting": "Sorting",
        "barcoding": "Barcoding",
        "sequencing": "Sequencing",
        "hitpicking": "Hitpicking"
    }

    records = []
    for seq_length in seq_lengths:
        for lib_size in library_sizes:
            step_costs = []
            for step in steps:
                if step in cost_funcs:
                    cost = cost_funcs[step](lib_size, seq_length)
                    cost_int = int(cost)
                    cpv = cost_int / lib_size if lib_size > 0 else 0
                    records.append({
                        "Length": int(seq_length),
                        "Library Size": int(lib_size),
                        "Step": step_display_names[step],
                        "Cost": cost_int,
                        "CPV": cpv,
                    })
                    step_costs.append(cost_int)

            # Add total row
            if step_costs:
                total_cost = sum(step_costs)
                total_cpv = total_cost / lib_size if lib_size > 0 else 0
                records.append({
                    "Length": int(seq_length),
                    "Library Size": int(lib_size),
                    "Step": "Total",
                    "Cost": total_cost,
                    "CPV": total_cpv,
                })
    return pd.DataFrame(records)