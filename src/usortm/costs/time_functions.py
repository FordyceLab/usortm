"""Time estimation functions for uSort-M workflow steps."""

import math


def calculate_sorting_time(n_plates, min_per_plate=8, setup_min=30):
    """Calculate FACS sorting time in minutes.

    Args:
        n_plates: Number of 384-well plates to sort
        min_per_plate: Minutes per plate (default: 8, per sorting protocol)
        setup_min: Setup/calibration time in minutes (default: 30)

    Returns:
        Total sorting time in minutes
    """
    return n_plates * min_per_plate + setup_min


def calculate_barcoding_time(n_plates, min_per_plate=50):
    """Calculate PCR barcoding time in minutes.

    Args:
        n_plates: Number of 384-well plates to barcode
        min_per_plate: Minutes per plate including setup and PCR (default: 50)

    Returns:
        Total barcoding time in minutes
    """
    # TODO: Refine this estimate based on actual lab workflow
    # 50 min per plate is probably too much at the moment
    return n_plates * min_per_plate


def calculate_sequencing_time(n_wells, seq_length, platform='plasmidsaurus'):
    """Calculate sequencing turnaround time in days.

    Args:
        n_wells: Number of wells sequenced
        seq_length: Sequence length in bp
        platform: Sequencing platform (default: 'plasmidsaurus')

    Returns:
        Sequencing turnaround time in days
    """
    if platform == 'plasmidsaurus':
        return 3  # 3-5 business days typical turnaround
    elif platform == 'ont':
        # For ONT in-house: prep (1 day) + run (1 day) + basecalling (1 day)
        return 3
    elif platform == 'illumina':
        # MiSeq: prep (1 day) + run (1-2 days) + processing (1 day)
        return 4
    else:
        return 3  # Default


def calculate_total_timeline(library_size, seq_length, fold_sampling=4):
    """Calculate complete uSort-M timeline breakdown.

    Args:
        library_size: Number of unique variants in library
        seq_length: Sequence length in bp
        fold_sampling: Fold oversampling (wells sorted / library size)

    Returns:
        Dictionary with timeline breakdown:
            - assembly_days: Days for library assembly
            - sort_days: Days for FACS sorting
            - barcode_days: Days for PCR barcoding
            - seq_days: Days for sequencing
            - demux_days: Days for demultiplexing/analysis
            - total_days: Total workflow duration
            - timeline: List of workflow steps with start/end days
    """
    # Calculate derived parameters
    n_wells = int(library_size * fold_sampling)
    n_plates = max(1, -(-n_wells // 384))  # Ceiling division

    # Calculate time for each step
    sort_min = calculate_sorting_time(n_plates)
    barcode_min = calculate_barcoding_time(n_plates)
    seq_days = calculate_sequencing_time(n_wells, seq_length)

    # Convert to days (assuming 8-hour workday)
    sort_hours = sort_min / 60
    barcode_hours = barcode_min / 60

    # Account for setup/cleanup time
    sort_days = max(1, math.ceil((sort_hours + 2) / 8))  # +2 hrs for setup
    barcode_days = max(1, math.ceil(barcode_hours / 8))

    # Fixed durations for other steps
    assembly_days = 2  # Cloning + transformation + overnight growth
    demux_days = 1     # Demultiplexing and analysis

    # Calculate timeline
    current_day = 0

    assembly_start = current_day
    assembly_end = assembly_start + assembly_days
    current_day = assembly_end

    sort_start = current_day
    sort_end = sort_start + sort_days
    current_day = sort_end

    barcode_start = current_day
    barcode_end = barcode_start + barcode_days
    current_day = barcode_end

    seq_start = current_day
    seq_end = seq_start + seq_days
    current_day = seq_end

    demux_start = current_day
    demux_end = demux_start + demux_days
    current_day = demux_end

    total_days = current_day

    return {
        'assembly_days': assembly_days,
        'sort_days': sort_days,
        'barcode_days': barcode_days,
        'seq_days': seq_days,
        'demux_days': demux_days,
        'total_days': total_days,
        'timeline': [
            {'step': 'Library Assembly', 'start': assembly_start, 'end': assembly_end, 'days': assembly_days},
            {'step': 'FACS Sorting', 'start': sort_start, 'end': sort_end, 'days': sort_days},
            {'step': 'PCR Barcoding', 'start': barcode_start, 'end': barcode_end, 'days': barcode_days},
            {'step': 'Sequencing', 'start': seq_start, 'end': seq_end, 'days': seq_days},
            {'step': 'Demux & Analysis', 'start': demux_start, 'end': demux_end, 'days': demux_days},
        ]
    }
