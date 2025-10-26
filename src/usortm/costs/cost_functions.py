import os
import math
import numpy as np

def idt_eblocks_synthesis_cost(length, fragment_number):
    if (length > 300) and (length <= 500):
        return fragment_number * 35
    else:
        return length * fragment_number * 0.07

def idt_gblocks_synthesis_cost(length, fragment_number):
    return length * fragment_number * 0.09

def twist_genefragments_synthesis_cost(length, fragment_number):
    if length < 300:
        return None
    elif 300 <= length <= 500:
        return fragment_number * 35
    else:
        return length * fragment_number * 0.07
    
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

def generate_commercial_cost_dict(fragment_sizes, library_sizes, assembly_method):
    """Tabulate all commercial costs and store in dictionary
    """
    # --- Compute Costs ---
    commercial_cost_comparison_dict = {}

    for frag_len in fragment_sizes:
        commercial_cost_comparison_dict[frag_len] = {
            'Twist': {
                'Gene Fragments': {
                    int(n): int(twist_genefragments_synthesis_cost(frag_len, n)) + 
                            int(parsed_genefragments_assembly_cost(n, assembly_method)) +
                            int(parsed_genefragments_barcoding_cost(n)) +
                            int(parsed_genefragments_sequencing_cost(frag_len, n))
                    for n in library_sizes
                }
            },
            'IDT': {
                'eBlocks': {
                    int(n): int(idt_eblocks_synthesis_cost(frag_len, n)) +
                            int(parsed_genefragments_assembly_cost(n, assembly_method)) +
                            int(parsed_genefragments_barcoding_cost(n)) +
                            int(parsed_genefragments_sequencing_cost(frag_len, n))
                    for n in library_sizes
                },
                'gBlocks': {
                    int(n): int(idt_gblocks_synthesis_cost(frag_len, n)) +
                            int(parsed_genefragments_assembly_cost(n, assembly_method)) +
                            int(parsed_genefragments_barcoding_cost(n)) +
                            int(parsed_genefragments_sequencing_cost(frag_len, n))
                    for n in library_sizes
                },
            }
        }
    
    return commercial_cost_comparison_dict

def generate_commercial_cost_stats_dict(commercial_cost_comparison_dict, library_sizes):
    # --- Cost statistics ---
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
            print(costs)
            stats_for_frag[n] = {
                'min': min(costs),
                'mean': sum(costs) / len(costs),
                'max': max(costs),
                'count': len(costs),
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
        print("Twist synthesis")
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
        print("Instance synthesis")
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
    cost = 0

    # Assume 8x sorting
    total_wells = library_size*8

    # Get number of 384-well plates
    n_plates = int(total_wells/384)

    # Get sort minutes, assuming 30 minutes per plate
    sort_minutes = n_plates*30

    # Add one hour for setup and cleaning
    sort_minutes += 60

    # Convert to cost
    # Rates defined at: https://facs.stanford.edu/facility-info/policies/proposed-rates-2024-2025
    # Sony SH800Z and BD Aria are both $70/hr
    machine_hourly_rate = 70
    operator_hourly_rate = 65

    total_cost = (sort_minutes/60)*(machine_hourly_rate+operator_hourly_rate)

    return total_cost

def usortm_barcoding_cost(library_size):
    # Assume 8x sorting
    total_wells = library_size*8
    n_plates = int(total_wells/384) # Get number of 384-well plates
    return n_plates*97.73 # From cost sheet

def usortm_sequencing_cost(library_size, seq_length):
    # Base cost of Plasmidsaurus Custom Sequencing
    cost = 500

    # Assume 8x sorting
    total_wells = library_size*8

    # 100 minimum reads per well
    total_reads = total_wells*100

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

# --- uSort-M cost function ---
def usortm_total_cost(library_sizes, seq_lengths):
    
    cost_dict = {}

    for seq_length in seq_lengths:
        single_len_dict = {}

        for lib_size in library_sizes:
            cost = usortm_synthesis_cost(lib_size, seq_length)
            cost += usortm_cloning_cost(lib_size)
            cost += usortm_sorting_cost(lib_size)
            cost += usortm_barcoding_cost(lib_size)
            cost += usortm_sequencing_cost(lib_size, seq_length)
            cost += usortm_hitpicking_cost(lib_size, seq_length)

            single_len_dict[int(lib_size)] = int(cost)
        
        cost_dict[int(seq_length)] = single_len_dict
        
    return cost_dict