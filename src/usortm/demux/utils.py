import os, glob, re, subprocess, pysam
import gzip

import numpy as np
import pandas as pd

import bionumpy as bnp
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import time
import string
import statistics
from tqdm import tqdm

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


export_dir = "demux_results"

def get_fastqs(root_dir):
    """Get all nested fastqs within a directory
    """
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('fastq'):
                full_file_path = os.path.join(dirpath, filename)
                file_paths.append(full_file_path)
    return file_paths

def count_fastq_reads(filepath):
    """Count reads in fastq
    """
    count = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i % 4 == 0:  # Every 4th line (0-indexed) is a new read header
                count += 1
    return count

def count_all_fastqs(root_dir):
    """Count all nested fastqs in a directory
    """
    fastqs = get_fastqs(root_dir)
    reads = 0
    for fastq in fastqs:
        reads += count_fastq_reads(fastq)
    return reads

def extract_first_n_reads(input_fastq_path, output_fastq_path, num_reads=1000):
    """
    Extracts the first 'num_reads' from a FASTQ file and writes them to a new file.

    Args:
        input_fastq_path (str): Path to the input FASTQ file.
        output_fastq_path (str): Path to the output FASTQ file.
        num_reads (int): The number of reads to extract.
    """
    reads_written = 0
    with open(input_fastq_path, 'r') as infile, open(output_fastq_path, 'w') as outfile:
        while reads_written < num_reads:
            # Read the four lines of a FASTQ record
            id_line = infile.readline()
            if not id_line:  # End of file reached before getting enough reads
                break
            seq_line = infile.readline()
            plus_line = infile.readline()
            qual_line = infile.readline()

            # Write the four lines to the output file
            outfile.write(id_line)
            outfile.write(seq_line)
            outfile.write(plus_line)
            outfile.write(qual_line)

            reads_written += 1

def compute_mean_qualities(reads):
    return np.mean(reads.quality, axis=1)

def read_in_barcodes(fbc_path, rbc_path):
    """Read in forward and reverse barcode CSV files and generate DataFrames."""
    # Read in barcodes
    fbc_df = pd.read_csv(fbc_path)
    fbc_df['barcode'] = fbc_df['refseq']
    fbc_df.drop(columns=['refseq'], inplace=True)
    print("FBC DataFrame:")
    display(fbc_df.head(10))

    # Read in barcodes
    rbc_df = pd.read_csv(rbc_path)
    rbc_df['barcode'] = rbc_df['refseq']
    rbc_df.drop(columns=['refseq'], inplace=True)
    print("RBC DataFrame:")
    display(rbc_df.head(10))

    return fbc_df, rbc_df

def write_barcode_fastas(fbc_df, rbc_df):
    """Write all barcodes to fasta files.
    """
    # First write out fbcs from barcode_df to fasta
    with open(f"dorado_fbcs.fasta", 'w') as f:
        for index, row in fbc_df.iterrows():
            f.write(f">LevSeq-fbc-{1+index:02}\n{row['barcode']}\n")

    # First write out fbcs from barcode_df to fasta
    with open(f"dorado_rbcs.fasta", 'w') as f:
        for index, row in rbc_df.iterrows():
            f.write(f">LevSeq-rbc-{1+index:02}\n{row['barcode']}\n")

def demux(
    data,
    output,
    toml,
    barcodes,
    kit_name="levSeq_bcs_map",
    output_fastq=True,
    emit_summary=True,
    bc_both_ends=False,
    no_trim=False,
):
    """
    Run Dorado demux with a custom barcode arrangement and sequences.
    """
    command = [
        "/Users/micaholivas/.dorado/bin/dorado", "demux",
        data,
        "--kit-name", kit_name,
        "--barcode-arrangement", toml,
        "--barcode-sequences", barcodes,
        "-o", output,
    ]

    if output_fastq:
        command.append("--emit-fastq")
    if emit_summary:
        command.append("--emit-summary")
    if bc_both_ends:
        command.append("--barcode-both-ends")
    if no_trim:
        command.append("--no-trim")

    return subprocess.run(command, check=True)

def batch_demux(fastq_dir, 
                output_root, 
                toml, 
                barcodes, 
                kit_name="levSeq_bcs_map"
                ):
    """
    Recursively find all FASTQs under fastq_dir and demux them.
    Each FASTQ gets its own subdirectory in output_root.
    """
    fastqs = glob.glob(os.path.join(fastq_dir, "**", "*.fastq*"), recursive=True)
    print(f"Found {len(fastqs)} FASTQ files")

    for fq in fastqs:
        # make output subdir based on input file name (without extension)
        fq_base = os.path.splitext(os.path.basename(fq))[0]
        fq_out = os.path.join(output_root, fq_base)
        os.makedirs(fq_out, exist_ok=True)

        print(f"→ Demuxing {fq} into {fq_out}")
        demux(
            data=fq,
            output=fq_out,
            toml=toml,
            barcodes=barcodes,
            kit_name=kit_name,
            output_fastq=True,
        )

def make_index(fasta):
    """Create a minimap2 index for a multisequence FASTA file"""
    mmi = fasta + ".mmi"
    if not os.path.exists(mmi):
        subprocess.run(["minimap2", "-d", mmi, fasta], check=True)
    return mmi

def align_and_split(fasta, 
                    fastq, 
                    out_root, 
                    preset="map-ont"
                    ):
    """
    Align one FASTQ → SAM, then split aligned reads into per-ref FASTQs.
    Results are nested under out_root/runid_sample/
    """
    os.makedirs(out_root, exist_ok=True)
    mmi = make_index(fasta)

    parent = os.path.basename(os.path.dirname(fastq))   # run folder
    stem   = os.path.splitext(os.path.basename(fastq))[0]
    sample = f"{parent}_{stem}"                         # include run name

    sample_dir = os.path.join(out_root, sample)
    os.makedirs(sample_dir, exist_ok=True)

    sam_path = os.path.join(sample_dir, f"{sample}.sam")

    if not os.path.exists(sam_path):
        print(f"Aligning {fastq} -> {sam_path}")
        with open(sam_path, "w") as out_sam:
            subprocess.run(
                ["minimap2", "-ax", preset, mmi, fastq],
                stdout=out_sam,
                check=True
            )

    # Split into per-ref FASTQs
    bam_path = sam_path.replace(".sam", ".bam")
    subprocess.run(["samtools", "view", "-bS", sam_path, "-o", bam_path], check=True)

    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_to_records = {}
    for read in bam:
        if read.is_unmapped or read.query_sequence is None:
            continue
        ref = bam.get_reference_name(read.reference_id)
        rec = SeqRecord(Seq(read.query_sequence), id=read.query_name, description="")
        rec.letter_annotations["phred_quality"] = read.query_qualities
        ref_to_records.setdefault(ref, []).append(rec)
    bam.close()

    for ref, recs in ref_to_records.items():
        fq_path = os.path.join(sample_dir, f"{ref}.fastq")
        SeqIO.write(recs, fq_path, "fastq")
        print(f"{sample}: {ref} → {len(recs)} reads")

def batch_align_and_split(fasta, 
                          fastq_dir, 
                          out_root="aln_results"
                          ):
    """
    Find all FASTQs under fastq_dir, align, and split into per-ref FASTQs.
    """
    fastqs = glob.glob(os.path.join(fastq_dir, "**", "*.fastq*"), recursive=True)
    print(f"Found {len(fastqs)} FASTQs")
    for fq in fastqs:
        align_and_split(fasta, fq, out_root)

def create_read_df(base_dir, debug_read=None):
    """
    Create a DataFrame with one row per read, containing:
    - read_name
    - fbc (forward barcode index)
    - rbc (reverse barcode index)
    - ref_name (name of reference sequence aligned to)
    - read_seq (nucleotide sequence of the read)
    """

    # Collect in dicts for fast lookups
    fbc_map, rbc_map, ref_map, seq_map = {}, {}, {}, {}

    # Clean read ID function
    def clean_id(rec_id):
        return rec_id.split()[0]  # drop anything after whitespace

    # --- load FBC demux ---
    print("Loading FBC demux...")
    for fq in tqdm(glob.glob(os.path.join(base_dir, "fbc", "**", "*.fastq*"), recursive=True)):
        fname = os.path.basename(fq)
        if "unclassified" in fname:
            continue
        m = re.search(r"barcode(\d+)", fname)
        if not m:
            continue
        fbc = int(m.group(1)) - 1
        for rec in SeqIO.parse(fq, "fastq"):
            if debug_read and rec.id == debug_read:
                print(f"Found read {rec.id} in FBC demux with fbc={fbc}")
            fbc_map[rec.id] = fbc

    # --- load RBC demux ---
    print("Loading RBC demux...")
    for fq in tqdm(glob.glob(os.path.join(base_dir, "rbc", "**", "*.fastq*"), recursive=True)):
        fname = os.path.basename(fq)
        if "unclassified" in fname:
            continue
        m = re.search(r"barcode(\d+)", fname)
        if not m:
            continue
        rbc = int(m.group(1)) - 1
        for rec in SeqIO.parse(fq, "fastq"):
            if debug_read and rec.id == debug_read:
                print(f"Found read {rec.id} in RBC demux with rbc={rbc}")
            rbc_map[rec.id] = rbc

    # --- load per-ref FASTQs ---
    print("Loading per-ref FASTQs...")
    for fq in tqdm(glob.glob(os.path.join(base_dir, "refs", "**", "*.fastq"), recursive=True)):
        ref_name = os.path.splitext(os.path.basename(fq))[0]
        for rec in SeqIO.parse(fq, "fastq"):
            if debug_read and rec.id == debug_read:
                print(f"Found read {rec.id} in refs with ref_name={ref_name}")
            rid = clean_id(rec.id)
            ref_map[rid] = ref_name
            seq_map[rid] = str(rec.seq)

    # --- build final dataframe ---
    print("Building DataFrame...")
    all_reads = set(fbc_map) | set(rbc_map) | set(ref_map)

    records = []
    for rid in all_reads:
        records.append({
            "read_name": rid,
            "fbc": fbc_map.get(rid),
            "rbc": rbc_map.get(rid),
            "ref_name": ref_map.get(rid),
            "read_seq": seq_map.get(rid),
        })

    df = pd.DataFrame.from_records(records)
    print("DataFrame complete.")

    print(f"Total reads: {len(df):,}")

    return df

def barcode_to_well(fbc_name, rbc_name):
    """
    Map FBxx + RBxx to 384-well plate coordinate like '1A1'.
    FB01–FB96 give position inside 96-well quadrant.
    RB01–RB32 give plate number (1–8) and quadrant.
    """
    if pd.isna(fbc_name) or pd.isna(rbc_name):
        return None

    # Parse integers from names
    fb = int(fbc_name.replace("FB", "")) - 1  # 0-based
    rb = int(rbc_name.replace("RB", "")) - 1  # 0-based

    # Plate number (1–8)
    plate_num = (rb // 4) + 1
    quadrant = rb % 4  # 0=TL, 1=TR, 2=BL, 3=BR

    # Position inside 96 quadrant
    row96 = fb // 12     # 0–7
    col96 = fb % 12      # 0–11

    # Map quadrant to 384 offsets
    if quadrant == 0:    # top-left
        row384 = row96
        col384 = col96
    elif quadrant == 1:  # top-right
        row384 = row96
        col384 = col96 + 12
    elif quadrant == 2:  # bottom-left
        row384 = row96 + 8
        col384 = col96
    elif quadrant == 3:  # bottom-right
        row384 = row96 + 8
        col384 = col96 + 12

    # Convert to well notation
    row_letter = string.ascii_uppercase[row384]   # A–P
    col_num = col384 + 1                         # 1–24
    return f"{plate_num}{row_letter}{col_num}"

def format_df(df, fbc_df=None, rbc_df=None):

    # --- map to names via index ---
    if fbc_df is not None and "fbc" in df.columns:
        df["fbc_name"] = df["fbc"].map(fbc_df["name"])

    if rbc_df is not None and "rbc" in df.columns:
        df["rbc_name"] = df["rbc"].map(rbc_df["name"])


    df = df[[
            'read_name', 
            'fbc_name', 
            'rbc_name', 
            'ref_name', 
            'read_seq'
            ]]

    # Drop reads without barcodes 
    df.dropna(subset=['fbc_name', 'rbc_name', 'ref_name'], inplace=True)

    # Add well position based on barcodes
    df["well_pos"] = df.apply(lambda r: barcode_to_well(r["fbc_name"], r["rbc_name"]), axis=1)

    # Reorder columns
    df = df[[
            'read_name', 
            'fbc_name', 
            'rbc_name', 
            'well_pos',
            'ref_name', 
            'read_seq'
            ]]
    
    display(df.head())
    
    return df