"""Demultiplexing utilities for Dorado, minimap2, and consensus calling.

Provides functions for barcode demultiplexing via Dorado, reference alignment
via minimap2, per-well consensus generation, and variant calling from CIGAR
strings. All external tool paths are auto-detected from PATH by default and
can be overridden via function parameters.
"""

import csv as csv_mod
import os
import glob
import gzip
import logging
import re
import string
import subprocess
import threading

import numpy as np
import pandas as pd
import pysam
from Bio.Seq import Seq
from Bio import SeqIO
from tqdm import tqdm

from usortm.demux.deps import find_dorado, find_minimap2, find_samtools

logger = logging.getLogger(__name__)

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

def make_index(fasta, minimap2_path=None):
    """Create a minimap2 index for a multisequence FASTA file.

    Args:
        fasta: Path to the FASTA file.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.

    Returns:
        Path to the generated .mmi index file.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    mmi = fasta + ".mmi"
    if not os.path.exists(mmi):
        subprocess.run([minimap2_path, "-d", mmi, fasta], check=True, stderr=subprocess.DEVNULL)
    return mmi


def align_and_split_by_strand(
    multi_ref_fasta,
    fastq,
    output_dir,
    minimap2_path=None,
    samtools_path=None,
    threads=4,
    progress_callback=None,
    total_reads=None,
):
    """Align raw reads to a multi-ref library and split by strand.

    Runs a single minimap2 pass (no direction filter), then splits reads
    into forward- and reverse-strand FASTQs.  Reverse reads are
    reverse-complemented back to the forward orientation so that
    downstream Dorado barcode demux sees consistent barcode positions.

    Each read in the output FASTQs is tagged with the reference it
    aligned to (``@readname|ref=REFNAME|dir=fwd``).

    Args:
        multi_ref_fasta: Path to the multi-entry reference FASTA.
        fastq: Path to raw input FASTQ.
        output_dir: Directory for output files.
        minimap2_path: Optional path to minimap2 binary.
        samtools_path: Optional path to samtools binary.
        threads: Number of minimap2/samtools threads.

    Returns:
        Tuple of (oriented_fastq, ref_map, align_stats) where:
        - oriented_fastq: Path to a single FASTQ with all reads in the
          forward orientation, tagged with ref and direction info.
        - ref_map: dict mapping read_name -> {ref, direction} for every
          mapped read.
        - align_stats: dict with keys ``fwd``, ``rev``, ``mapped``,
          ``unmapped``.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    os.makedirs(output_dir, exist_ok=True)
    mmi = make_index(multi_ref_fasta, minimap2_path=minimap2_path)

    bam_path = os.path.join(output_dir, "ref_alignment.bam")
    oriented_fq = os.path.join(output_dir, "oriented_reads.fastq")

    # --- minimap2 | samtools sort → BAM (no intermediate SAM on disk) ---
    if os.path.exists(bam_path):
        logger.info("Using cached alignment BAM: %s", bam_path)
        if progress_callback is not None:
            progress_callback(None, None)  # signal: cached
    else:
        logger.info("Running minimap2 multi-ref alignment...")
        mm2_cmd = [
            minimap2_path, "-ax", "map-ont",
            "--secondary=no",   # skip secondary alignments (we discard them anyway)
            "-t", str(threads),
            mmi, fastq,
        ]
        sort_cmd = [
            samtools_path, "sort",
            "-@", str(threads),
            "-o", bam_path,
        ]
        stderr_target = subprocess.PIPE if progress_callback is not None else subprocess.DEVNULL
        mm2_proc = subprocess.Popen(
            mm2_cmd, stdout=subprocess.PIPE, stderr=stderr_target,
        )

        # Always connect mm2 stdout directly to samtools stdin (no Python relay).
        sort_proc = subprocess.Popen(
            sort_cmd, stdin=mm2_proc.stdout, stderr=subprocess.DEVNULL,
        )
        mm2_proc.stdout.close()  # let sort_proc own the read end

        if progress_callback is not None:
            # Parse minimap2 stderr for "mapped N sequences" progress lines.
            # This avoids routing SAM data through Python, keeping the
            # mm2→samtools pipe at full kernel speed.
            _mm2_stderr_re = re.compile(rb"mapped (\d+) sequences")

            def _parse_stderr():
                count = 0
                for line in mm2_proc.stderr:
                    m = _mm2_stderr_re.search(line)
                    if m:
                        count = int(m.group(1))
                        progress_callback(count, total_reads)
                # Final update with whatever count we saw
                progress_callback(count, total_reads)

            stderr_thread = threading.Thread(target=_parse_stderr, daemon=True)
            stderr_thread.start()
            sort_proc.wait()
            stderr_thread.join()
        else:
            sort_proc.wait()

        if sort_proc.returncode != 0:
            raise subprocess.CalledProcessError(sort_proc.returncode, sort_cmd)
        mm2_proc.wait()

        # Index the BAM for random access
        subprocess.run(
            [samtools_path, "index", bam_path],
            check=True, stderr=subprocess.DEVNULL,
        )

    # --- Split by strand and write oriented FASTQ ---
    logger.info("Splitting reads by strand...")
    ref_map = {}  # read_name -> {"ref": ..., "direction": ...}
    n_fwd = n_rev = n_unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam, open(oriented_fq, "w") as fq_out:
        for read in bam:
            if read.is_unmapped or not read.query_sequence:
                n_unmapped += 1
                continue
            if read.is_secondary or read.is_supplementary:
                continue

            ref_name = bam.get_reference_name(read.reference_id)
            seq = read.query_sequence
            quals = read.query_qualities

            if read.is_reverse:
                # RC back to forward orientation
                seq = str(Seq(seq).reverse_complement())
                if quals is not None:
                    quals = quals[::-1]
                direction = "rev"
                n_rev += 1
            else:
                direction = "fwd"
                n_fwd += 1

            qual_str = "".join(chr(q + 33) for q in quals) if quals else "I" * len(seq)
            read_name = read.query_name

            ref_map[read_name] = {"ref": ref_name, "direction": direction}
            fq_out.write(
                f"@{read_name}|ref={ref_name}|dir={direction}\n"
                f"{seq}\n+\n{qual_str}\n"
            )

    align_stats = {
        "fwd": n_fwd,
        "rev": n_rev,
        "mapped": n_fwd + n_rev,
        "unmapped": n_unmapped,
    }
    logger.info(
        "Strand split complete: %d forward, %d reverse, %d unmapped",
        n_fwd, n_rev, n_unmapped,
    )
    return oriented_fq, ref_map, align_stats


def csv_to_reference_fasta(csv_path, fasta_path, strip_flanking=True):
    """Convert a Name,Sequence CSV to a multi-entry reference FASTA.

    Args:
        csv_path: Path to CSV with 'Name' and 'Sequence' columns.
        fasta_path: Path to output FASTA file.
        strip_flanking: If True, keep only uppercase characters (strips
            lowercase flanking regions).

    Returns:
        Path to the generated FASTA file.
    """
    fasta_path = str(fasta_path)
    os.makedirs(os.path.dirname(fasta_path) or ".", exist_ok=True)

    n_entries = 0
    with open(csv_path) as f_in, open(fasta_path, "w") as f_out:
        reader = csv_mod.DictReader(f_in)
        # Normalise headers: strip surrounding whitespace so "Sequence " == "Sequence"
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            name = row["Name"]
            seq = row["Sequence"]
            if strip_flanking:
                seq = "".join(c for c in seq if c.isupper())
            f_out.write(f">{name}\n{seq}\n")
            n_entries += 1

    logger.info("Wrote %d entries to %s", n_entries, fasta_path)
    return fasta_path


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

def align_multi_ref(
    multi_ref_fasta,
    fastq,
    out_root,
    preset="map-ont",
    direction=None,
    minimap2_path=None,
    samtools_path=None,
):
    """Align one FASTQ to a multi-entry reference and export a ref-tagged FASTQ.

    Handles disk and SAM parsing errors gracefully.

    Args:
        multi_ref_fasta: Path to multi-entry reference FASTA.
        fastq: Path to input FASTQ file.
        out_root: Output directory root.
        preset: minimap2 preset (default "map-ont" for ONT reads).
        direction: "forward", "reverse", or None.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()
    os.makedirs(out_root, exist_ok=True)
    mmi = make_index(multi_ref_fasta, minimap2_path=minimap2_path)

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
        cmd_list = [minimap2_path, "-ax", preset, mmi, fastq]
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
        subprocess.run([samtools_path, "view", "-bS", sam_path, "-o", bam_path],
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

def batch_align(
    fasta,
    fastq_dir,
    out_root,
    direction=None,
    minimap2_path=None,
    samtools_path=None,
):
    """Recursively align all FASTQs under fastq_dir to a reference.

    Args:
        fasta: Path to reference FASTA.
        fastq_dir: Directory to search for FASTQ files.
        out_root: Output directory for aligned results.
        direction: "forward", "reverse", or None.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
    """
    fastqs = glob.glob(os.path.join(fastq_dir, "**", "*.fastq*"), recursive=True)
    print(f"Found {len(fastqs)} FASTQs")
    for fq in fastqs:
        try:
            align_multi_ref(
                fasta, fq, out_root,
                direction=direction,
                minimap2_path=minimap2_path,
                samtools_path=samtools_path,
            )
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
    """Read forward and reverse barcode CSV files and generate DataFrames.

    Expects CSV files with at least a 'refseq' column containing the barcode
    DNA sequence. The column is renamed to 'barcode' in the output.

    Args:
        fbc_path: Path to forward barcode CSV.
        rbc_path: Path to reverse barcode CSV.

    Returns:
        Tuple of (fbc_df, rbc_df) DataFrames.
    """
    fbc_df = pd.read_csv(fbc_path)
    fbc_df['barcode'] = fbc_df['refseq']
    fbc_df.drop(columns=['refseq'], inplace=True)
    print(f"FBC DataFrame: {len(fbc_df)} barcodes loaded")

    rbc_df = pd.read_csv(rbc_path)
    rbc_df['barcode'] = rbc_df['refseq']
    rbc_df.drop(columns=['refseq'], inplace=True)
    print(f"RBC DataFrame: {len(rbc_df)} barcodes loaded")

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
    dorado_path=None,
    output_fastq=True,
    emit_summary=True,
    bc_both_ends=False,
    no_trim=False,
    max_reads=None,
):
    """Run Dorado demux with a custom barcode arrangement and sequences.

    Args:
        data: Path to input FASTQ or BAM file.
        output: Output directory for demultiplexed files.
        toml: Path to barcode arrangement TOML file.
        barcodes: Path to barcode sequences FASTA file.
        kit_name: Kit name identifier for Dorado.
        dorado_path: Path to dorado binary. Auto-detected if None.
        output_fastq: Emit FASTQ output (default True).
        emit_summary: Emit demux summary (default True).
        bc_both_ends: Require barcode on both ends of read.
        no_trim: Do not trim barcodes from reads.
        max_reads: Maximum number of reads to process (None = all).

    Returns:
        CompletedProcess from subprocess.run.
    """
    if dorado_path is None:
        dorado_path = find_dorado()

    command = [
        dorado_path, "demux",
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

    logger.info("Running dorado demux: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Dorado demux failed (exit %d): %s",
            result.returncode,
            result.stderr.strip(),
        )
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )
    if result.stderr:
        logger.debug("Dorado stderr: %s", result.stderr.strip())
    return result

def human_format(num):
    """Convert large numbers to human-readable form (e.g. 12.3k)."""
    for unit in ["", "k", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def batch_demux(
    fastq,
    output_root,
    toml,
    barcodes,
    kit_name="levSeq_bcs_map",
    dorado_path=None,
    max_reads=None,
):
    """Recursively find all FASTQs under a directory and demux them.

    Each FASTQ gets its own subdirectory in output_root.

    Args:
        fastq: Path to a single FASTQ file or directory of FASTQs.
        output_root: Root output directory for demultiplexed files.
        toml: Path to barcode arrangement TOML file.
        barcodes: Path to barcode sequences FASTA file.
        kit_name: Kit name identifier for Dorado.
        dorado_path: Path to dorado binary. Auto-detected if None.
        max_reads: Maximum number of reads to process per file.
    """
    if fastq.endswith('.fastq'):
        fastqs = [fastq]
        print("Single fastq")
    else:
        fastqs = glob.glob(os.path.join(fastq, "**", "*.fastq*"), recursive=True)
        print(f"Found {len(fastqs)} FASTQ file(s)\n")

    for i, fq in enumerate(fastqs):
        print(f"[{i + 1}/{len(fastqs)}]\tDemuxing {os.path.basename(fq)}")
        fq_base = os.path.splitext(os.path.basename(fq))[0]
        fq_out = os.path.join(output_root, fq_base)
        os.makedirs(fq_out, exist_ok=True)

        demux(
            data=fq,
            output=fq_out,
            toml=toml,
            barcodes=barcodes,
            kit_name=kit_name,
            dorado_path=dorado_path,
            output_fastq=True,
            max_reads=max_reads,
        )

def create_read_df(base_dir, ref_map=None, oriented_fastq=None):
    """Build a per-read DataFrame merging barcode demux and reference data.

    Collects FBC assignments from ``base_dir/fbc/``, RBC assignments from
    ``base_dir/rbc/``, and reference/direction info from either a
    pre-computed *ref_map* dict (new pipeline) or by scanning
    ``base_dir/refs/fwd/`` and ``base_dir/refs/rev/`` directories
    (legacy pipeline).

    Args:
        base_dir: Root output directory containing ``fbc/`` and ``rbc/``
            subdirectories from Dorado demux.
        ref_map: Optional dict ``{read_name: {"ref": ..., "direction": ...}}``
            returned by :func:`align_and_split_by_strand`.  When provided,
            the legacy ``refs/`` directory scan is skipped.
        oriented_fastq: Path to the oriented FASTQ produced by
            :func:`align_and_split_by_strand`.  Used to collect read
            sequences and quality scores when *ref_map* is provided.

    Returns:
        DataFrame with columns: ``read_name``, ``fbc``, ``rbc``,
        ``ref_name``, ``read_seq``, ``read_qual``, ``avg_qual``.
    """
    fbc_map, rbc_map = {}, {}
    _ref_map, seq_map, qual_map, avgq_map = {}, {}, {}, {}
    malformed_counts = {"fbc": 0, "rbc": 0, "ref": 0}

    def normalize_id(rid):
        if not rid: return None
        rid = rid.split()[0]
        return re.sub(r"\|ref=.*|\|dir=.*|/[12]$|_pool_plates.*", "", rid)

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

    # --- Collect reference + sequence data ---
    if ref_map is not None and oriented_fastq is not None:
        # New pipeline: ref info from align_and_split_by_strand(),
        # sequences from the oriented FASTQ.
        print("Loading reference assignments from alignment...")
        for read_name, info in ref_map.items():
            direction = info["direction"]
            ref_name = info["ref"]
            _ref_map[normalize_id(read_name)] = f"{direction}:{ref_name}"

        print("Collecting read sequences from oriented FASTQ...")
        open_fn = gzip.open if oriented_fastq.endswith('.gz') else open
        with open_fn(oriented_fastq, 'rt') as fh:
            for rec in tqdm(SeqIO.parse(fh, "fastq")):
                rid = normalize_id(rec.id)
                if not rid:
                    continue
                quals = rec.letter_annotations["phred_quality"]
                seq_map[rid] = str(rec.seq)
                qual_map[rid] = "".join(chr(q + 33) for q in quals)
                avgq_map[rid] = sum(quals) / len(quals)
    else:
        # Legacy pipeline: scan refs/fwd/ and refs/rev/ directories.
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
                            _ref_map[rid] = f"{direction}:{ref_name}"
                            seq_map[rid] = str(rec.seq)
                            qual_map[rid] = "".join(chr(q + 33) for q in quals)
                            avgq_map[rid] = sum(quals) / len(quals)
                except: malformed_counts["ref"] += 1

    print("Building DataFrame...")
    all_reads = set(fbc_map) | set(rbc_map) | set(_ref_map)
    df = pd.DataFrame([{
        "read_name": rid,
        "fbc": fbc_map.get(rid),
        "rbc": rbc_map.get(rid),
        "ref_name": _ref_map.get(rid),
        "read_seq": seq_map.get(rid),
        "read_qual": qual_map.get(rid),
        "avg_qual": avgq_map.get(rid)
    } for rid in all_reads])

    df.attrs["fbc_classified"] = len(fbc_map)
    df.attrs["rbc_classified"] = len(rbc_map)
    df.attrs["ref_assigned"] = len(_ref_map)

    print(f"Total reads: {len(df):,}")
    print(f"  FBC classified: {len(fbc_map):,}")
    print(f"  RBC classified: {len(rbc_map):,}")
    print(f"  Ref assigned: {len(_ref_map):,}")
    print(f"Malformed counts: {malformed_counts}")
    return df

def barcode_to_well(fbc_name, rbc_name):
    """
    Map FBxx + RBxx to interleaved 384-well coordinate like '1A3'.
    Interleaving (by quadrant):
      TL(q=0): odd rows,  odd cols
      TR(q=1): odd rows,  even cols
      BL(q=2): even rows, odd cols
      BR(q=3): even rows, even cols
    RB01–RB32 -> plate 1–8 and quadrant order TL, TR, BL, BR.
    FB01–FB96 index within the 96 grid (A–H x 1–12).
    """
    if pd.isna(fbc_name) or pd.isna(rbc_name):
        return None

    # Parse integers (0-based)
    fb = int(str(fbc_name).replace("FB", "")) - 1  # 0..95
    rb = int(str(rbc_name).replace("RB", "")) - 1  # 0..31
    if not (0 <= fb < 96 and 0 <= rb < 32):
        return None

    # Plate number (1..8) and quadrant (0..3)
    plate_num = (rb // 4) + 1
    quadrant = rb % 4  # 0=TL,1=TR,2=BL,3=BR

    # 96-well row/col (0-based)
    row96 = fb // 12       # 0..7 (A..H)
    col96 = fb % 12        # 0..11 (1..12)

    # Interleaved offsets (1-based parity)
    # TL: (row+1, col+1) = (odd, odd)
    # TR: (odd, even), BL: (even, odd), BR: (even, even)
    row_off = 1 if quadrant in (0, 1) else 2
    col_off = 1 if quadrant in (0, 2) else 2

    # 384 coordinates (1-based)
    row384 = row96 * 2 + row_off          # 1..16
    col384 = col96 * 2 + col_off          # 1..24

    row_letter = string.ascii_uppercase[row384 - 1]  # A..P
    return f"{plate_num}{row_letter}{col384}"

def _parse_well(w):
    if type(w) == str:
        m = re.match(r"(\d+)([A-P]+)(\d+)", str(w))
        return (int(m.group(1)), m.group(2), int(m.group(3))) if m else (None, None, None)
    else:
        return None

def format_df(df, fbc_df=None, rbc_df=None, ref_fasta=None):
    """
    Format merged demux/reference DataFrame.
    Adds readable barcode names, well positions, reference sequences, and lengths.
    """
    # --- map barcode numeric IDs to names ---
    if fbc_df is not None and "fbc" in df.columns:
        df["fbc_name"] = df["fbc"].map(fbc_df["name"])
    if rbc_df is not None and "rbc" in df.columns:
        df["rbc_name"] = df["rbc"].map(rbc_df["name"])

    # Ensure required columns exist even when no reads were classified
    # (e.g., tiny subsample or no-reference run). This keeps downstream
    # filtering deterministic and avoids KeyError on empty demux outputs.
    for col in ("fbc_name", "rbc_name", "ref_name"):
        if col not in df.columns:
            df[col] = pd.NA

    # --- drop reads missing required info ---
    pre_filter = len(df)
    df = df.dropna(subset=["fbc_name", "rbc_name", "ref_name"]).copy()
    logger.info("format_df: %d -> %d reads after dropping incomplete assignments",
                pre_filter, len(df))

    # --- add well position ---
    df["well_pos"] = df.apply(
        lambda r: barcode_to_well(r["fbc_name"], r["rbc_name"]), axis=1
    )

    # --- reorder / include quality columns ---
    cols = [
        "read_name", "fbc_name", "rbc_name", "well_pos",
        "ref_name", "read_seq", "read_qual", "avg_qual"
    ]
    df = df[[c for c in cols if c in df.columns]]

    # --- add reference sequences and lengths ---
    if ref_fasta is not None:
        from Bio import SeqIO
        ref_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(ref_fasta, "fasta")}

        def get_ref_id(ref_name):
            if pd.isna(ref_name):
                return None
            return ref_name.split(":", 1)[-1] if ":" in ref_name else ref_name

        df["ref_id"] = df["ref_name"].apply(get_ref_id)
        df["ref_seq"] = df["ref_id"].map(ref_seqs)
        df["ref_len"] = df["ref_seq"].str.len()

    print(df["well_pos"].unique())
    df = df.sort_values(by="well_pos", key=lambda s: s.map(_parse_well))
    return df

def generate_well_df(read_df):
    output_cols = [
        "plate", "well", "global_well", "depth",
        "major_ref", "major_freq", "ref_len", "ref_seq",
    ]

    # Copy df
    temp_df = read_df.copy()
    if "well_pos" not in temp_df.columns:
        return pd.DataFrame(columns=output_cols)

    temp_df = temp_df.dropna(subset=['well_pos'])
    all_wells = temp_df.well_pos.unique()
    if len(all_wells) == 0:
        return pd.DataFrame(columns=output_cols)

    # Generate well_df
    well_df = pd.DataFrame(
        columns=[
            "plate", "well", "global_well", "depth", "well_row",
            "well_col", "major_ref", "major_freq", "ref_len", "ref_seq",
        ]
    )

    for index, well in tqdm(enumerate(all_wells), total=len(all_wells)):
        curr = read_df[read_df['well_pos'] == well]
        depth = len(curr)
        if depth == 0:
            continue
        # Strip fwd:/rev: strand prefixes so reads for the same variant
        # are counted together (otherwise a well with 100% one variant
        # but split across strands would show ~50% major_freq).
        refs = [r.split(":", 1)[-1] if r.startswith(("fwd:", "rev:")) else r
                for r in curr['ref_name'].to_list()]
        if not refs:
            continue
        major_ref = max(set(refs), key=refs.count)
        major_freq = refs.count(major_ref)/len(curr)
        # Look up ref_seq/ref_len from original column (may still have prefix)
        ref_match = curr[curr['ref_name'].str.endswith(major_ref, na=False)]
        ref_seq = None
        ref_len = None
        if not ref_match.empty:
            if "ref_seq" in ref_match.columns:
                ref_seq = ref_match['ref_seq'].iloc[0]
            if "ref_len" in ref_match.columns and pd.notna(ref_match['ref_len'].iloc[0]):
                ref_len = int(ref_match['ref_len'].iloc[0])

        parsed_well = _parse_well(well)
        if parsed_well is None:
            continue
        plate_num = parsed_well[0]
        well_row = parsed_well[1]
        well_col = parsed_well[2]
        if plate_num is None or well_row is None or well_col is None:
            continue
        plate_well = well_row + str(well_col)

        well_df.at[index, 'plate'] = plate_num
        well_df.at[index, 'well'] = plate_well
        well_df.at[index, 'global_well'] = well
        well_df.at[index, 'depth'] = depth
        well_df.at[index, 'well_row'] = well_row
        well_df.at[index, 'well_col'] = well_col

        well_df.at[index, 'major_ref'] = major_ref
        well_df.at[index, 'major_freq'] = major_freq
        well_df.at[index, 'ref_len'] = ref_len
        well_df.at[index, 'ref_seq'] = ref_seq

    if well_df.empty:
        return pd.DataFrame(columns=output_cols)

    well_df.dropna(subset=['plate', 'well_row', 'well_col'], inplace=True)
    if well_df.empty:
        return pd.DataFrame(columns=output_cols)
    well_df.sort_values(by=['plate', 'well_row', 'well_col'], inplace=True)

    # Drop well_row and well_col
    well_df.drop(columns=['well_row', 'well_col'], inplace=True)

    return well_df

def _process_single_well(well, paths, minimap2_path, samtools_path):
    """Run alignment → consensus → re-alignment → CIGAR extraction for one well.

    Args:
        well: Well identifier string (e.g. "1A1").
        paths: Dict with keys: ref_fa, fq, bam, cons_fa, cons_bam.
        minimap2_path: Path to minimap2 binary.
        samtools_path: Path to samtools binary.

    Returns:
        Tuple of (well, cigar_str, cons_seq) — values may be None on failure.
    """
    ref_fa = paths["ref_fa"]
    fq = paths["fq"]
    bam = paths["bam"]
    cons_fa = paths["cons_fa"]
    cons_bam = paths["cons_bam"]

    # 1) Align reads to reference, pipe through samtools sort
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", ref_fa, fq],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", bam],
            stdin=mm2.stdout,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        mm2.wait()
    except Exception as e:
        print(f"Alignment failed for {well}: {e}")
        return well, None, None

    # 2) Generate consensus
    try:
        with open(cons_fa, "w") as cons_out:
            subprocess.run(
                [samtools_path, "consensus", "-f", "fasta", bam],
                stdout=cons_out,
                check=False,
            )
    except Exception as e:
        print(f"Consensus failed for {well}: {e}")
        return well, None, None

    # 3) Align consensus back to reference
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", ref_fa, cons_fa],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", cons_bam],
            stdin=mm2.stdout,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        mm2.wait()
    except Exception as e:
        print(f"Consensus alignment failed for {well}: {e}")
        return well, None, None

    # 4) Extract CIGAR + consensus sequence
    cigar_str, cons_seq = None, None
    try:
        with pysam.AlignmentFile(cons_bam, "rb") as bamfile:
            for read in bamfile:
                if not read.is_unmapped:
                    cigar_str = read.cigarstring
                    break

        if os.path.exists(cons_fa):
            with open(cons_fa) as f:
                lines = f.read().splitlines()
                cons_seq = "".join(l for l in lines if not l.startswith(">"))
    except Exception as e:
        print(f"Error processing {well}: {e}")

    return well, cigar_str, cons_seq


def generate_per_well_consensus(
    well_df,
    read_df,
    out_root,
    reference_dir,
    minimap2_path=None,
    samtools_path=None,
    workers: int = 4,
):
    """Generate per-well consensus sequences and add alignment info to well_df.

    For each well: writes per-well FASTQ, aligns to reference with minimap2,
    generates consensus with samtools, and extracts CIGAR strings.

    Args:
        well_df: DataFrame with per-well summary (from generate_well_df).
        read_df: DataFrame with per-read data (from format_df).
        out_root: Root output directory.
        reference_dir: Directory containing single_ref_fastas/ subdirectory.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
        workers: Number of parallel threads for consensus alignment.

    Returns:
        Updated well_df with CIGAR and cons_seq columns.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    # Set up output directories
    wells_dir = os.path.join(out_root, "wells")
    well_fastqs_dir = os.path.join(wells_dir, "fastqs")
    os.makedirs(wells_dir, exist_ok=True)
    os.makedirs(well_fastqs_dir, exist_ok=True)

    # 1) Write per-well FASTQs (sequential — fast I/O, sets up paths for parallel step)
    all_wells = read_df['well_pos'].unique()

    print("Writing per-well fastqs...")
    for well in tqdm(all_wells):
        current_per_well_df = read_df[read_df["well_pos"] == well]
        out_path = os.path.join(well_fastqs_dir, f"{well}.fastq")

        with open(out_path, "w") as f:
            for _, row in current_per_well_df.iterrows():
                f.write(
                    f"@{row['read_name']}\n"
                    f"{row['read_seq']}\n+\n"
                    f"{row['read_qual']}\n"
                )

    # Set up consensus output
    single_fasta_reference_dir = os.path.join(reference_dir, "single_ref_fastas")
    well_consensus_dir = os.path.join(wells_dir, "consensus")
    os.makedirs(well_consensus_dir, exist_ok=True)

    # Add columns if missing
    if "CIGAR" not in well_df.columns:
        well_df["CIGAR"] = None
    if "cons_seq" not in well_df.columns:
        well_df["cons_seq"] = None

    # Pre-compute paths for all wells that have summary data
    well_set = set(well_df["global_well"].values)
    well_paths = {}
    for well in all_wells:
        if well not in well_set:
            continue
        major_ref = well_df.loc[well_df["global_well"] == well, "major_ref"].iloc[0]
        if ":" in major_ref:
            major_ref = major_ref.split(":")[-1]
        well_paths[well] = {
            "ref_fa": os.path.join(single_fasta_reference_dir, f"{major_ref}.fasta"),
            "fq": os.path.join(well_fastqs_dir, f"{well}.fastq"),
            "bam": os.path.join(well_consensus_dir, f"{well}.bam"),
            "cons_fa": os.path.join(well_consensus_dir, f"{well}_consensus.fasta"),
            "cons_bam": os.path.join(well_consensus_dir, f"{well}_consensus_align.bam"),
        }

    # 2) Parallel consensus alignment
    print(f"Generating consensus alignments ({workers} workers)...")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_single_well, well, well_paths[well],
                minimap2_path, samtools_path
            ): well
            for well in well_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            well, cigar, cons = future.result()
            results[well] = (cigar, cons)

    # 3) Apply results back to well_df (sequential, avoids DataFrame race conditions)
    for well, (cigar, cons) in results.items():
        well_df.loc[well_df["global_well"] == well, ["CIGAR", "cons_seq"]] = [cigar, cons]

    return well_df

def extract_matches(well_df):
    """Extract reference matches using consensus CIGAR string
    """

    for index, row in well_df.iterrows():
        # Get CIGAR string, reference length
        well = row['global_well']
        ref_len = row['ref_len']
        ref_seq = row['ref_seq']
        cigar = row['CIGAR']
        cons_seq = row['cons_seq']

        status = ""
        
        if cigar == None:
            status = "Error"
        
        else:
            # 1) Check for perfect matches
            if ''.join(x for x in cigar if x.isalpha()).lower() == 'm':
                if int(cigar[:-1]) == int(ref_len):
                    status = "Perfect Match"
                else:
                    status = "Partial Match"
            else:
                status = "Other Error"        
                
                # 2) Check for silent mutations
                # Translate each sequence
                if len(cons_seq) == ref_len:
                    if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                        status = "Silent Mutation"
                else:
                    status = "Error"

        well_df.at[index, "cons_check"] = status

    return well_df

def export_reference_map(df, 
                         filename):

    mapping_dict = {}

    # get list of refs from reference fasta
    all_refs = []
    f = "reference_fasta/multi_entry.fasta"
    with open(f, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            all_refs.append(record.id)

    for ref in all_refs:
        # filter df for each ref
        sub = df[df['ref_name'] == ref]

        # get pure well with the most reads for that ref
        counts = sub['well_pos'].value_counts()
        if len(counts) == 0:
            mapping_dict[ref] = ["Missed", True, "Missed"]
            continue

        top_well = counts.idxmax()
        top_count = counts.max()
        total_count = counts.sum()
        top_frac = top_count / total_count
        if top_count >= 50 and top_frac >= 0.9:
            mapping_dict[ref] = [top_well, False, None]

        # If less than 50 reads but 100% in one well, take it too
        elif top_count < 50 and top_frac == 1.0:
            mapping_dict[ref] = [top_well, True, f"Low reads ({top_count}), but homogeneous"]
        
        # If more than 50 reads but less than 90% in one well, flag it
        elif top_count >= 50 and top_frac < 0.9:
            mapping_dict[ref] = [top_well, True, f"Mixed reads: aligned to {top_frac:.1%} of reads"]

    # Generate a DataFrame from the mapping_dict
    mapping_df = pd.DataFrame.from_dict(mapping_dict, orient='index', columns=['Well', 'Flag', 'Note'])
    mapping_df.index.name = 'Reference'
    mapping_df = mapping_df.reset_index()

    # Save to CSV
    mapping_df.to_csv(os.path.join(export_dir, filename), index=False)

def export_well_map(well_df, filename):
    """Export a mapping of wells to references.
    """
    export_df = well_df.copy()

    row_string = 'ABCDEFGHIJKLMNOP'

    for index, row in well_df.iterrows():
        well_df.at[index, 'Row'] = row_string.index(row['well'][0])
        well_df.at[index, 'Col'] = row['well'][1:]

    # Sort by plate and well
    export_df = export_df.sort_values(['plate', 'Row', 'Col'])
    export_df = export_df.drop(columns=['Row', 'Col'])
    export_df = export_df.reset_index(drop=True)

    # Save to CSV
    export_df.to_csv(os.path.join(export_dir, filename), index=False)
