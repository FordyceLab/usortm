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

def make_index(fasta):
    """Create a minimap2 index for a multisequence FASTA file"""
    mmi = fasta + ".mmi"
    if not os.path.exists(os.path.join('demux_results', mmi)):
        subprocess.run(["minimap2", "-d", mmi, fasta], check=True)
    return mmi

def bam_to_fastq_with_ref(bam_path, fastq_out):
    """
    Convert BAM → FASTQ, appending aligned reference name to read ID.
    Handles missing qualities and skips unmapped reads.
    """
    with pysam.AlignmentFile(bam_path, "rb") as bam, open(fastq_out, "w") as fq:
        for read in bam:
            if read.is_unmapped:
                continue
            ref_name = bam.get_reference_name(read.reference_id)
            seq = read.query_sequence or ""
            quals = read.query_qualities
            qual_str = "".join(chr(q + 33) for q in quals) if quals else "I" * len(seq)
            fq.write(f"@{read.query_name}|ref={ref_name}\n{seq}\n+\n{qual_str}\n")

def align_multi_ref(multi_ref_fasta, fastq, out_root, preset="map-ont", direction=None):
    """
    Align one FASTQ to a multi-entry reference and export a single ref-tagged FASTQ.
    Handles disk and SAM parsing errors gracefully.
    """
    os.makedirs(out_root, exist_ok=True)
    mmi = make_index(multi_ref_fasta)

    parent = os.path.basename(os.path.dirname(fastq))
    stem = os.path.splitext(os.path.basename(fastq))[0]
    sample = f"{parent}_{stem}"
    sample_dir = os.path.join(out_root, sample)
    os.makedirs(sample_dir, exist_ok=True)

    sam_path = os.path.join(sample_dir, f"{sample}.sam")
    bam_path = sam_path.replace(".sam", ".bam")
    fq_out = os.path.join(sample_dir, f"{sample}.fastq")

    # --- run minimap2 ---
    if not os.path.exists(sam_path):
        cmd_list = ["minimap2", "-ax", preset, mmi, fastq]
        if direction == "forward":
            cmd_list.append("--for-only")
        elif direction == "reverse":
            cmd_list.append("--rev-only")

        print(f"[INFO] Running: {' '.join(cmd_list)}")
        try:
            with open(sam_path, "w") as out_sam:
                subprocess.run(cmd_list, stdout=out_sam, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"minimap2 failed for {fastq}: {e.stderr.decode(errors='ignore')[:500]}")
            return
        except OSError as e:
            print(f"OS error for {fastq}: {e}")
            return

    # --- convert SAM → BAM ---
    try:
        subprocess.run(["samtools", "view", "-bS", sam_path, "-o", bam_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"samtools view failed for {sam_path}: {e.stderr.decode(errors='ignore')[:500]}")
        return
    except OSError as e:
        print(f"OS error during samtools view for {sam_path}: {e}")
        return

    # --- export ref-tagged FASTQ ---
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam, open(fq_out, "w") as fq:
            for read in bam:
                if read.is_unmapped or not read.query_sequence:
                    continue
                seq = read.query_sequence
                quals = read.query_qualities
                if direction == "reverse":
                    seq = str(Seq(seq).reverse_complement())
                    if quals:
                        quals = quals[::-1]
                ref = bam.get_reference_name(read.reference_id)
                qual_str = "".join(chr(q + 33) for q in (quals or []))
                if not qual_str:
                    qual_str = "I" * len(seq)
                fq.write(f"@{read.query_name}|ref={ref}\n{seq}\n+\n{qual_str}\n")
        print(f"[✓] Wrote combined FASTQ → {fq_out}")
    except Exception as e:
        print(f"Error while writing FASTQ for {fastq}: {e}")

def batch_align(fasta, fastq_dir, out_root, direction=None):
    """
    Recursively align all FASTQs under fastq_dir and export one combined FASTQ per sample.
    """
    fastqs = glob.glob(os.path.join(fastq_dir, "**", "*.fastq*"), recursive=True)
    print(f"Found {len(fastqs)} FASTQs")
    print(fastqs)
    for fq in fastqs:
        try:
            align_multi_ref(fasta, fq, out_root, direction=direction)
        except Exception as e:
            print(f"Skipped {fq}: {e}")

def get_read_names(file):
    """Get read names in current fastq
    """
    names, bad = set(), 0
    open_fn = gzip.open if file.endswith('.gz') else open
    with open_fn(file, 'rt', errors='ignore') as h:
        for rec in SeqIO.parse(h, 'fastq'):
            if rec.id.strip():
                names.add(rec.id)
            else:
                bad += 1
    return names, bad

def get_all_read_names(root_dir):
    """Get all read names in current directory
    """
    names, malformed = set(), 0
    fastqs = get_fastqs(root_dir)
    for f in fastqs:
        try:
            n, b = get_read_names(f)
            names |= n
            malformed += b
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return names, malformed

def ref_alignment_stats(fastq_dir, out_root):

    total_count = count_all_fastqs(fastq_dir)
    fwd_count = count_all_fastqs(os.path.join(out_root, "refs/fwd/"))
    rev_count = count_all_fastqs(os.path.join(out_root, "refs/rev/"))
    total_mapped_count = fwd_count + rev_count

    print("--- Counts ---")
    print(f"Total Read Count: {total_count:,}")
    print(f"Count (Fwd): {fwd_count:,} ({round(100*fwd_count/total_count, 1)})")
    print(f"Count (RevComp): {rev_count:,} ({round(100*rev_count/total_count, 1)})")
    print(f"Total Mapped Count: {total_mapped_count:,} ({round(100*total_mapped_count/total_count, 1)}% of total)")
    print()

    print("--- Intersection ---")
    fwd_names, _ = get_all_read_names(os.path.join(out_root, "refs/fwd/"))
    rev_names, _ = get_all_read_names(os.path.join(out_root, "refs/rev/"))
    intersection = len(fwd_names & rev_names)
    print(f"Overlapping Reads: {len(fwd_names & rev_names):,} ({round(100 * intersection / total_count, 1)}% of total)")

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

def write_barcode_fastas(fbc_df, 
                         rbc_df,
                         export_dir="demux_results"
                         ):
    """Write all barcodes to fasta files.
    """
    # If output directory doesn't exist, create it
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    # First write out fbcs from barcode_df to fasta
    with open(os.path.join(export_dir, "dorado_fbcs.fasta"), 'w') as f:
        for index, row in fbc_df.iterrows():
            f.write(f">LevSeq-fbc-{1+index:02}\n{row['barcode']}\n")
    print("Wrote forward barcodes to:\tdorado_fbcs.fasta")

    # First write out fbcs from barcode_df to fasta
    with open(os.path.join(export_dir, "dorado_rbcs.fasta"), 'w') as f:
        for index, row in rbc_df.iterrows():
            f.write(f">LevSeq-rbc-{1+index:02}\n{row['barcode']}\n")
    print("Wrote forward barcodes to:\tdorado_rbcs.fasta")
    
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
    max_reads=None,
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

    if max_reads is not None:
        command.append("--max-reads")
        command.append(str(max_reads))
    if output_fastq:
        command.append("--emit-fastq")
    if emit_summary:
        command.append("--emit-summary")
    if bc_both_ends:
        command.append("--barcode-both-ends")
    if no_trim:
        command.append("--no-trim")

    return subprocess.run(command, check=True)

def human_format(num):
    """Convert large numbers to human-readable form (e.g. 12.3k)."""
    for unit in ["", "k", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def batch_demux(fastq, 
                output_root, 
                toml, 
                barcodes, 
                kit_name="levSeq_bcs_map",
                max_reads=None,
                ):
    """
    Recursively find all FASTQs under fastq_dir and demux them.
    Each FASTQ gets its own subdirectory in output_root.
    Uses tqdm for prettier, compact progress output.
    """
    if fastq.endswith('.fastq'):
        fastqs = [fastq]
        print("Single fastq")
    else:
        fastqs = glob.glob(os.path.join(fastq, "**", "*.fastq*"), recursive=True)
        print(f"Found {len(fastqs)} FASTQ file(s)\n")
        
    for i, fq in enumerate(fastqs):
        # Get file size
        fq_size = int(os.path.getsize(fq))
        print(f"[{i+1}/{len(fastqs)}]\tDemuxing {os.path.basename(fq)}")
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

def create_read_df(base_dir):
    import os, re, glob, pandas as pd
    from tqdm import tqdm
    from Bio import SeqIO

    fbc_map, rbc_map, ref_map, seq_map, qual_map, avgq_map = {}, {}, {}, {}, {}, {}
    malformed_counts = {"fbc": 0, "rbc": 0, "ref": 0}

    def normalize_id(rid):
        if not rid: return None
        rid = rid.split()[0]
        return re.sub(r"\|ref=.*|/[12]$|_pool_plates.*", "", rid)

    print("Collecting FBC demux...")
    for fq in tqdm(glob.glob(f"{base_dir}/fbc/**/*.fastq*", recursive=True)):
        if "unclassified" in fq: continue
        m = re.search(r"barcode(\d+)", fq)
        if not m: continue
        fbc = int(m.group(1)) - 1
        try:
            for rec in SeqIO.parse(fq, "fastq"):
                rid = normalize_id(rec.id)
                if rid: fbc_map[rid] = fbc
        except: malformed_counts["fbc"] += 1

    print("Collecting RBC demux...")
    for fq in tqdm(glob.glob(f"{base_dir}/rbc/**/*.fastq*", recursive=True)):
        if "unclassified" in fq: continue
        m = re.search(r"barcode(\d+)", fq)
        if not m: continue
        rbc = int(m.group(1)) - 1
        try:
            for rec in SeqIO.parse(fq, "fastq"):
                rid = normalize_id(rec.id)
                if rid: rbc_map[rid] = rbc
        except: malformed_counts["rbc"] += 1

    print("Collecting reference reads...")
    for direction in ["fwd", "rev"]:
        for fq in tqdm(glob.glob(f"{base_dir}/refs/{direction}/**/*.fastq*", recursive=True)):
            try:
                for rec in SeqIO.parse(fq, "fastq"):
                    rid = normalize_id(rec.id)
                    if not rid: continue
                    m = re.search(r"\|ref=([^\s|]+)", rec.id)
                    ref_name = m.group(1) if m else None
                    if ref_name:
                        quals = rec.letter_annotations["phred_quality"]
                        ref_map[rid] = f"{direction}:{ref_name}"
                        seq_map[rid] = str(rec.seq)
                        qual_map[rid] = "".join(chr(q + 33) for q in quals)
                        avgq_map[rid] = sum(quals) / len(quals)
            except: malformed_counts["ref"] += 1

    print("Building DataFrame...")
    all_reads = set(fbc_map) | set(rbc_map) | set(ref_map)
    df = pd.DataFrame([{
        "read_name": rid,
        "fbc": fbc_map.get(rid),
        "rbc": rbc_map.get(rid),
        "ref_name": ref_map.get(rid),
        "read_seq": seq_map.get(rid),
        "read_qual": qual_map.get(rid),
        "avg_qual": avgq_map.get(rid)
    } for rid in all_reads])

    print(f"Total reads: {len(df):,}")
    print(f"Malformed counts: {malformed_counts}")
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