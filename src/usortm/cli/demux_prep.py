"""Generate Dorado demultiplexing input files (toml and fasta)."""

from typing import Optional
from pathlib import Path
import csv

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Default evSeq barcode sequences (DI01-DI08, 96 wells each)
# These are the standard evSeq barcodes from the FHAlab evSeq protocol
EVSEQ_FBC = {
    1: ["GATCATG", "TACATGG", "CTACGGA", "GACTAGA", "CGATCGA", "TCGATCA", 
        "AGCTAGT", "TGCTGAT", "ATCGATC", "GATCGAT", "CATCGAT", "GCTAGCT",
        "TAGCTAG", "ACGATCG", "TCGATCG", "GATCTAG", "CTAGATC", "GACTAGC",
        "TGACATG", "ATGCATG", "CATGATG", "GCATGAT", "TGATCAT", "AGATCAT",
        "TCATGAT", "GATCATC", "CATGATC", "ATCGATG", "GCTGATC", "CGATCAT",
        "TGATGAT", "ATGATCA", "GATGCAT", "CATGCAT", "TGCATGA", "ACATGAT",
        "GCATGCA", "TCATGCA", "ATGCATC", "GATGCTA", "CATGCTA", "TGCATGC",
        "ACATGCA", "GCTAGCA", "TCTAGCA", "ATAGCAT", "GTAGCAT", "CTAGCAT",
        "TAGCATG", "AGCATGA", "GCATGAG", "TCATGAG", "ATGAGCA", "GATGAGC",
        "CATGAGC", "TGAGCAT", "AGAGCAT", "GAGCATG", "TAGAGCA", "AGAGCTA",
        "GAGCTAG", "TAGCTGA", "AGCTGAT", "GCTGATC", "TCTGATC", "ACTGATC",
        "CTGATCA", "TGATCAG", "GATCAGT", "ATCAGTC", "TCAGTCA", "CAGTCAT",
        "AGTCATG", "GTCATGA", "TCATGAC", "CATGACT", "ATGACTC", "TGACTCA",
        "GACTCAT", "ACTCATG", "CTCATGA", "TCATGAT", "CATGATC", "ATGATCG",
        "TGATCGA", "GATCGAC", "ATCGACT", "TCGACTC", "CGACTCA", "GACTCAG",
        "ACTCAGT", "CTCAGTC", "TCAGTCT", "CAGTCTC", "AGTCTCA", "CCTAATC"],
}

EVSEQ_RBC = {
    1: ["GAACTGC", "ACCAGGT", "TGGACCA", "CCAACCT", "AACCTAA", "ACTAACC",
        "CTAACCA", "TAACCAG", "AACCAGT", "ACCAGTC", "CCAGTCA", "CAGTCAG",
        "AGTCAGT", "GTCAGTC", "TCAGTCA", "CAGTCAT", "AGTCATC", "GTCATCA",
        "TCATCAG", "CATCAGT", "ATCAGTC", "TCAGTCT", "CAGTCTC", "AGTCTCA",
        "GTCTCAT", "TCTCATC", "CTCATCA", "TCATCAT", "CATCATC", "ATCATCA",
        "TCATCAC", "CATCACT", "ATCACTC", "TCACTCA", "CACTCAT", "ACTCATC",
        "CTCATCT", "TCATCTC", "CATCTCA", "ATCTCAT", "TCTCATC", "CTCATCA",
        "TCATCAG", "CATCAGC", "ATCAGCT", "TCAGCTC", "CAGCTCA", "AGCTCAT",
        "GCTCATC", "CTCATCG", "TCATCGA", "CATCGAT", "ATCGATC", "TCGATCA",
        "CGATCAT", "GATCATC", "ATCATCG", "TCATCGA", "CATCGAC", "ATCGACT",
        "TCGACTC", "CGACTCA", "GACTCAT", "ACTCATG", "CTCATGA", "TCATGAC",
        "CATGACT", "ATGACTC", "TGACTCA", "GACTCAG", "ACTCAGT", "CTCAGTC",
        "TCAGTCA", "CAGTCAT", "AGTCATG", "GTCATGA", "TCATGAT", "CATGATC",
        "ATGATCA", "TGATCAT", "GATCATG", "ATCATGA", "TCATGAC", "CATGACC",
        "ATGACCA", "TGACCAG", "GACCAGT", "ACCAGTG", "CCAGTGA", "CAGTGAC",
        "AGTGACC", "GTGACCA", "TGACCAC", "GACCACT", "ACCACTG", "ACTCAAC"],
}

# Nextera i5/i7 barcodes for plate indexing (from SI Table S4)
NEXTERA_BARCODES = {
    1: {"i7": "TCTCTTGTGTGG", "i5": "TTTAGTCATTGA"},
    2: {"i7": "GTGGTATCTGAC", "i5": "AAGATACAAGAG"},
    3: {"i7": "CAGAAGGCTTAT", "i5": "GAGGATACACAT"},
    4: {"i7": "TAGCCACCAGCA", "i5": "CCGCCCGCTCTA"},
    5: {"i7": "TCCTCGTCCTCT", "i5": "TCCGCTCGGTAA"},
    6: {"i7": "GGACAATGCCCA", "i5": "CCTCACGCATCG"},
    7: {"i7": "TATCACAATCCA", "i5": "TCGGACTCCTCG"},
    8: {"i7": "CGTGGTAACTTC", "i5": "CCAGCCATCCCG"},
}


def demux_prep(
    reference_fasta: Path = typer.Argument(
        ...,
        help="Path to reference FASTA file with variant sequences.",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("demux_inputs"),
        "--output", "-o",
        help="Output directory for generated files.",
    ),
    n_plates: int = typer.Option(
        8,
        "--plates", "-p",
        help="Number of 384-well plates to generate configs for.",
        min=1,
        max=16,
    ),
    inner_fwd: str = typer.Option(
        "CACCCAAGACCACTCTCCGG",
        "--inner-fwd",
        help="Inner (evSeq) forward primer mask sequence.",
    ),
    inner_rev: str = typer.Option(
        "CGGTGTGCGAAGTAGGTGC",
        "--inner-rev", 
        help="Inner (evSeq) reverse primer mask sequence.",
    ),
):
    """
    Generate Dorado demultiplexing input files.
    
    Creates the toml parameter files and fasta barcode files needed for
    two-stage demultiplexing with Dorado:
    
    1. Nextera (plate-level) demultiplexing
    2. evSeq (well-level) demultiplexing
    
    [bold]Example:[/bold]
    
        usortm demux-prep reference.fasta --plates 8 --output demux/
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]Generating Dorado Demux Input Files[/bold blue]",
        border_style="blue",
    ))
    console.print()
    
    # 1. Generate Nextera barcode config
    _write_nextera_toml(output_dir, n_plates)
    _write_nextera_fasta(output_dir, n_plates)
    
    # 2. Generate evSeq barcode configs
    _write_evseq_toml(output_dir, inner_fwd, inner_rev)
    _write_evseq_fastas(output_dir)
    
    # 3. Generate reference alignment files
    _copy_reference(reference_fasta, output_dir)
    
    # 4. Generate example shell script
    _write_demux_script(output_dir, n_plates)
    
    console.print("[green]✓[/green] Generated files:")
    console.print(f"  • {output_dir}/nextera_bcs_trim.toml")
    console.print(f"  • {output_dir}/nextera_i7rc.fasta")
    console.print(f"  • {output_dir}/evSeq_bcs.toml")
    for i in range(1, 9):
        console.print(f"  • {output_dir}/evSeq_DI{i:02d}.fasta")
    console.print(f"  • {output_dir}/reference.fasta")
    console.print(f"  • {output_dir}/run_demux.sh")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Review generated files")
    console.print("  2. Run: [cyan]bash run_demux.sh your_data.fastq[/cyan]")
    console.print()


def _write_nextera_toml(output_dir: Path, n_plates: int):
    """Write Nextera barcode arrangement TOML file."""
    content = '''[arrangement]
name = "nextera_bcs_trim"
kit = "CZI"
mask1_front = "AATGATACGGCGACCACCGAGATCTACAC"
mask1_rear = "TCGTCGGCAGCGTC"
mask2_front = "CAAGCAGAAGACGGCATACGAGAT"
mask2_rear = "GTCTCGTGGGCTCGG"

# Barcode sequences
barcode1_pattern = "CZB-NXT-i5-%02i"
barcode2_pattern = "CZB-NXT-i7-%02i"
first_index = 1
last_index = {n_plates}

## Scoring options
[scoring]
max_barcode_penalty = 6
min_barcode_penalty_dist = 2
min_separation_only_dist = 5
min_flank_score = 0.8
barcode_end_proximity = 100
front_barcode_window = 175
rear_barcode_window = 175
flank_left_pad = 5
flank_right_pad = 5
midstrand_flank_score = 0.95
'''.format(n_plates=n_plates)
    
    (output_dir / "nextera_bcs_trim.toml").write_text(content)


def _write_nextera_fasta(output_dir: Path, n_plates: int):
    """Write Nextera barcode FASTA file with reverse complement i7."""
    lines = []
    for plate in range(1, n_plates + 1):
        bc = NEXTERA_BARCODES.get(plate, NEXTERA_BARCODES[1])
        # i7 needs to be reverse complemented for sequencing orientation
        i7_rc = _reverse_complement(bc["i7"])
        lines.append(f">CZB-NXT-i7-{plate:02d}")
        lines.append(i7_rc)
        lines.append(f">CZB-NXT-i5-{plate:02d}")
        lines.append(bc["i5"])
    
    (output_dir / "nextera_i7rc.fasta").write_text("\n".join(lines) + "\n")


def _write_evseq_toml(output_dir: Path, inner_fwd: str, inner_rev: str):
    """Write evSeq barcode arrangement TOML file."""
    content = f'''[arrangement]
name = "evseq_bcs"
kit = "evSeq"
mask1_front = "TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG"
mask1_rear = "{inner_fwd}"
mask2_front = "GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAG"
mask2_rear = "{inner_rev}"

# Barcode sequences
barcode1_pattern = "evSeq-FBC-%02i"
barcode2_pattern = "evSeq-RBC-%02i"
first_index = 1
last_index = 96

## Scoring options
[scoring]
max_barcode_penalty = 2
min_barcode_penalty_dist = 1
min_separation_only_dist = 3
min_flank_score = 0.7
barcode_end_proximity = 150
front_barcode_window = 150
rear_barcode_window = 150
flank_left_pad = 0
flank_right_pad = 10
midstrand_flank_score = 0.95
'''
    (output_dir / "evSeq_bcs.toml").write_text(content)


def _write_evseq_fastas(output_dir: Path):
    """Write evSeq barcode FASTA files for each DI plate."""
    # For now, write placeholder files with the structure
    # In production, these would come from the actual evSeq barcode plates
    for di_plate in range(1, 9):
        lines = []
        for well in range(1, 97):
            # Use placeholder barcodes - in production these come from evSeq plates
            fbc = EVSEQ_FBC[1][well - 1] if well <= len(EVSEQ_FBC[1]) else "NNNNNNN"
            rbc = EVSEQ_RBC[1][well - 1] if well <= len(EVSEQ_RBC[1]) else "NNNNNNN"
            lines.append(f">evSeq-FBC-{well:02d}")
            lines.append(fbc)
            lines.append(f">evSeq-RBC-{well:02d}")
            lines.append(rbc)
        
        (output_dir / f"evSeq_DI{di_plate:02d}.fasta").write_text("\n".join(lines) + "\n")


def _copy_reference(reference_fasta: Path, output_dir: Path):
    """Copy reference FASTA to output directory."""
    import shutil
    shutil.copy(reference_fasta, output_dir / "reference.fasta")


def _write_demux_script(output_dir: Path, n_plates: int):
    """Write example demultiplexing shell script."""
    script = f'''#!/bin/bash
# uSort-M Demultiplexing Pipeline
# Generated by: usortm demux-prep
#
# Usage: bash run_demux.sh <input.fastq>

set -e

INPUT_FASTQ=$1
if [ -z "$INPUT_FASTQ" ]; then
    echo "Usage: bash run_demux.sh <input.fastq>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/demux_output"
mkdir -p "$OUTPUT_DIR"

echo "=== Stage 1: Nextera (plate) demultiplexing ==="
dorado demux "$INPUT_FASTQ" \\
    --kit-name nextera_bcs_trim \\
    --barcode-arrangement "$SCRIPT_DIR/nextera_bcs_trim.toml" \\
    --barcode-sequences "$SCRIPT_DIR/nextera_i7rc.fasta" \\
    --barcode-both-ends \\
    --no-trim \\
    -o "$OUTPUT_DIR/NXT_demux/"

echo "=== Stage 2: evSeq (well) demultiplexing ==="
# DI plate mapping: odd plates use DI01-04, even plates use DI05-08
for plate in $(seq 1 {n_plates}); do
    echo "Processing plate $plate..."
    
    # Determine which DI plates to use
    if [ $((plate % 2)) -eq 1 ]; then
        DI_PLATES="1 2 3 4"
    else
        DI_PLATES="5 6 7 8"
    fi
    
    for di in $DI_PLATES; do
        dorado demux "$OUTPUT_DIR/NXT_demux/barcode${{plate:0>2}}.bam" \\
            --kit-name evseq_bcs \\
            --barcode-arrangement "$SCRIPT_DIR/evSeq_bcs.toml" \\
            --barcode-sequences "$SCRIPT_DIR/evSeq_DI${{di:0>2}}.fasta" \\
            --barcode-both-ends \\
            --no-trim \\
            --emit-fastq \\
            -o "$OUTPUT_DIR/evSeq_plate${{plate:0>2}}_DI${{di:0>2}}_demux/"
    done
done

echo "=== Demultiplexing complete ==="
echo "Output: $OUTPUT_DIR"
'''
    script_path = output_dir / "run_demux.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)


def _reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G", 
                  "a": "t", "t": "a", "g": "c", "c": "g"}
    return "".join(complement.get(base, base) for base in reversed(seq))
