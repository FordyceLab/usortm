"""Demultiplexing utilities for Dorado, minimap2, and consensus calling.

Provides functions for barcode demultiplexing via Dorado, reference alignment
via minimap2, per-well consensus generation, and variant calling from CIGAR
strings. All external tool paths are auto-detected from PATH by default and
can be overridden via function parameters.
"""

import csv as csv_mod
import hashlib
import os
import glob
import gzip
import logging
import re
import string
import subprocess
import json


def _open_fastq(path: str):
    """Return gzip.open or open based on magic bytes, not file extension."""
    with open(path, "rb") as f:
        magic = f.read(2)
    return gzip.open if magic == b'\x1f\x8b' else open

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


def _rebuild_ref_map_from_fastq(path):
    """Reconstruct ref_map from tagged FASTQ headers.

    Parses lines like ``@readname|ref=X|dir=fwd`` and returns the same
    (ref_map, stats) structure produced by :func:`align_and_split_by_strand`.
    """
    ref_map = {}
    n_fwd = n_rev = 0
    tag_re = re.compile(r"^@([^|]+)\|ref=([^|]+)\|dir=(fwd|rev)")
    with open(path) as fh:
        for line in fh:
            m = tag_re.match(line)
            if m:
                read_name, ref_name, direction = m.group(1), m.group(2), m.group(3)
                ref_map[read_name] = {"ref": ref_name, "direction": direction}
                if direction == "fwd":
                    n_fwd += 1
                else:
                    n_rev += 1
    stats = {"fwd": n_fwd, "rev": n_rev, "mapped": n_fwd + n_rev}
    return ref_map, stats


def _input_fingerprint(fastq_path):
    """Identify an input FASTQ cheaply, for cache invalidation.

    Hashing the reads themselves is not affordable — a production input runs
    to several gigabytes — so the input is identified by absolute path, byte
    size and modification time instead.  The bias is deliberate: a false
    mismatch costs a re-alignment, while a false match would silently process
    the wrong reads.

    Args:
        fastq_path: Path to the input FASTQ.

    Returns:
        Dict describing the file, or ``None`` if it cannot be stat'ed.
    """
    try:
        st = os.stat(fastq_path)
    except OSError:
        return None
    return {
        "path": os.path.abspath(fastq_path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


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

    Streams minimap2 SAM output directly (no samtools sort, no BAM, no
    index) in a single pass.  Reverse-mapped reads are
    reverse-complemented back to forward orientation so that downstream
    Dorado barcode demux sees consistent barcode positions.

    Each read in the output FASTQ is tagged with the reference it
    aligned to (``@readname|ref=REFNAME|dir=fwd``).

    Args:
        multi_ref_fasta: Path to the multi-entry reference FASTA.
        fastq: Path to raw input FASTQ.
        output_dir: Directory for output files.
        minimap2_path: Optional path to minimap2 binary.
        samtools_path: Optional path to samtools binary (unused, kept
            for backward compatibility).
        threads: Number of minimap2 threads.
        progress_callback: Optional ``(n_done, total)`` callback.
        total_reads: Total reads for progress denominator.

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

    os.makedirs(output_dir, exist_ok=True)
    mmi = make_index(multi_ref_fasta, minimap2_path=minimap2_path)

    oriented_fq = os.path.join(output_dir, "oriented_reads.fastq")
    stats_path = os.path.join(output_dir, "align_stats.json")

    # --- Cache invalidation ---
    # The cache is keyed on BOTH the reference and the input FASTQ.  Keying on
    # the reference alone silently reuses a previous run's reads whenever the
    # input changes but the reference does not — e.g. a --subsample pass
    # followed by the full run in the same project directory, which would
    # process only the subsampled reads while reporting success.
    ref_hash = hashlib.md5(open(multi_ref_fasta, "rb").read()).hexdigest()
    input_fp = _input_fingerprint(fastq)

    # --- Cache check: if oriented FASTQ already exists, rebuild from it ---
    if os.path.exists(oriented_fq):
        stale_reason = "no saved alignment stats"
        saved = None
        if os.path.exists(stats_path):
            with open(stats_path) as fh:
                saved = json.load(fh)
            if saved.get("ref_hash") != ref_hash:
                stale_reason = "reference changed"
            elif saved.get("input") != input_fp:
                stale_reason = "input FASTQ changed"
            elif input_fp is None:
                stale_reason = "input FASTQ could not be identified"
            else:
                stale_reason = None

        if stale_reason is None:
            logger.info("Using cached oriented FASTQ: %s", oriented_fq)
            if progress_callback is not None:
                progress_callback(None, None)  # signal: cached
            ref_map, align_stats = _rebuild_ref_map_from_fastq(oriented_fq)
            align_stats["unmapped"] = saved.get("unmapped", 0)
            return oriented_fq, ref_map, align_stats
        else:
            logger.info(
                "Regenerating oriented FASTQ (%s): %s", stale_reason, oriented_fq
            )
            os.remove(oriented_fq)

    # SAM FLAG bits used below
    _FLAG_UNMAPPED = 0x4
    _FLAG_REVERSE = 0x10
    _FLAG_SECONDARY = 0x100
    _FLAG_SUPPLEMENTARY = 0x800

    # --- Stream minimap2 SAM as plain text (no pysam, no samtools) ---
    logger.info("Running minimap2 multi-ref alignment (streaming)...")
    mm2_cmd = [
        minimap2_path, "-ax", "map-ont",
        "--secondary=no",
        "-t", str(threads),
        mmi, fastq,
    ]
    mm2_stderr_path = os.path.join(output_dir, "minimap2.log")
    mm2_stderr_fh = open(mm2_stderr_path, "w")
    logger.info("minimap2 stderr → %s", mm2_stderr_path)
    mm2_proc = subprocess.Popen(
        mm2_cmd, stdout=subprocess.PIPE, stderr=mm2_stderr_fh,
    )

    ref_map = {}
    n_fwd = n_rev = n_unmapped = n_processed = 0

    with open(oriented_fq, "w") as fq_out:
        for raw_line in mm2_proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith("@"):
                continue  # skip SAM header lines

            fields = line.split("\t", 11)  # only need first 11 columns
            if len(fields) < 11:
                continue

            qname = fields[0]
            flag = int(fields[1])
            rname = fields[2]
            seq = fields[9]
            qual = fields[10]

            if flag & (_FLAG_SECONDARY | _FLAG_SUPPLEMENTARY):
                continue

            n_processed += 1
            if progress_callback is not None and n_processed % 5000 == 0:
                progress_callback(n_processed, total_reads)

            if flag & _FLAG_UNMAPPED or seq == "*":
                n_unmapped += 1
                continue

            if flag & _FLAG_REVERSE:
                seq = str(Seq(seq).reverse_complement())
                qual = qual[::-1]
                direction = "rev"
                n_rev += 1
            else:
                direction = "fwd"
                n_fwd += 1

            ref_map[qname] = {"ref": rname, "direction": direction}
            fq_out.write(
                f"@{qname}|ref={rname}|dir={direction}\n"
                f"{seq}\n+\n{qual}\n"
            )

    mm2_proc.wait()
    mm2_stderr_fh.close()
    if mm2_proc.returncode != 0:
        # Clean up partial output so next run doesn't hit cache
        if os.path.exists(oriented_fq):
            os.remove(oriented_fq)
        raise subprocess.CalledProcessError(mm2_proc.returncode, mm2_cmd)

    # Final progress update
    if progress_callback is not None:
        progress_callback(n_processed, total_reads)

    align_stats = {
        "fwd": n_fwd,
        "rev": n_rev,
        "mapped": n_fwd + n_rev,
        "unmapped": n_unmapped,
    }

    # Write sidecar for unmapped count + cache key (reference and input FASTQ)
    sidecar = dict(align_stats)
    sidecar["ref_hash"] = ref_hash
    sidecar["input"] = _input_fingerprint(fastq)
    with open(stats_path, "w") as fh:
        json.dump(sidecar, fh)

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
    open_fn = _open_fastq(file)
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
        open_fn = _open_fastq(oriented_fastq)
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

def format_df(df, fbc_df=None, rbc_df=None, ref_fasta=None, orient_ref_fasta=None):
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
    if len(df) > 0:
        df["well_pos"] = df.apply(
            lambda r: barcode_to_well(r["fbc_name"], r["rbc_name"]), axis=1
        )
    else:
        df["well_pos"] = pd.Series(dtype=object)

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
        if orient_ref_fasta is not None:
            for rec in SeqIO.parse(orient_ref_fasta, "fasta"):
                ref_seqs.setdefault(rec.id, str(rec.seq))

        def get_ref_id(ref_name):
            if pd.isna(ref_name):
                return None
            return ref_name.split(":", 1)[-1] if ":" in ref_name else ref_name

        df["ref_id"] = df["ref_name"].apply(get_ref_id)
        df["ref_seq"] = df["ref_id"].map(ref_seqs)
        df["ref_len"] = df["ref_seq"].str.len()

    if len(df) > 0:
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
    ref_mmi = paths.get("ref_mmi", ref_fa)
    fq = paths["fq"]
    bam = paths["bam"]
    cons_fa = paths["cons_fa"]
    cons_bam = paths["cons_bam"]

    # 1) Align reads to reference, pipe through samtools sort
    #    Use -t 1 since these are small per-well FASTQs run in parallel.
    #    Use pre-built .mmi to avoid concurrent index creation races.
    #    -m 64M limits samtools sort memory to avoid resource exhaustion
    #    with many concurrent workers.
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", "-t", "1", "--secondary=no", ref_mmi, fq],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        sort_result = subprocess.run(
            [samtools_path, "sort", "-m", "64M", "-o", bam],
            stdin=mm2.stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
        mm2.stdout.close()
        mm2.wait()
        if sort_result.returncode != 0:
            print(f"Alignment failed for {well}: samtools sort error: {sort_result.stderr.strip()}")
            return well, None, None
    except Exception as e:
        print(f"Alignment failed for {well}: {e}")
        return well, None, None

    # 2) Generate consensus
    try:
        with open(cons_fa, "w") as cons_out:
            subprocess.run(
                [samtools_path, "consensus", "-f", "fasta", bam],
                stdout=cons_out,
                check=True,
            )
    except Exception as e:
        print(f"Consensus failed for {well}: {e}")
        return well, None, None

    # 3) Align consensus back to reference
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", "-t", "1", "--secondary=no", ref_mmi, cons_fa],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        sort_result2 = subprocess.run(
            [samtools_path, "sort", "-m", "64M", "-o", cons_bam],
            stdin=mm2.stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
        mm2.stdout.close()
        mm2.wait()
        if sort_result2.returncode != 0:
            print(f"Consensus alignment failed for {well}: samtools sort error: {sort_result2.stderr.strip()}")
            return well, None, None

        # Add MD tags (required by _check_flanking_regions)
        calmd_bam = cons_bam + ".calmd.bam"
        with open(calmd_bam, "wb") as _fh:
            subprocess.run(
                [samtools_path, "calmd", "-b", cons_bam, str(ref_fa)],
                stdout=_fh,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        os.replace(calmd_bam, cons_bam)

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


def reassign_refs_from_consensus(well_df, ref_fasta,
                                  flank_5p_len=0, flank_3p_len=0):
    """Reassign major_ref per well by matching consensus to library references.

    When ``--orient-ref`` or ``--vector-fasta`` is used, every well's
    ``major_ref`` points to the orient reference.  This function compares each
    well's consensus sequence against the full variant library and picks the
    best-matching reference, restoring per-well variant identity.

    When *flank_5p_len* / *flank_3p_len* are provided, the flanking regions
    are stripped from the consensus before comparison so that only the
    variable region is matched against the library entries.

    Uses numpy for fast vectorized comparison (~800 refs x ~2500 wells).

    Args:
        well_df: DataFrame from generate_well_df / generate_per_well_consensus
            with ``cons_seq`` column.
        ref_fasta: Path to the full multi-entry reference FASTA.
        flank_5p_len: Length of the 5' flanking region to strip from consensus.
        flank_3p_len: Length of the 3' flanking region to strip from consensus.

    Returns:
        Updated well_df with corrected ``major_ref``, ``ref_seq``, ``ref_len``.
    """
    from Bio import SeqIO

    ref_records = list(SeqIO.parse(ref_fasta, "fasta"))
    if not ref_records:
        return well_df

    ref_ids = [rec.id for rec in ref_records]
    ref_seqs_str = [str(rec.seq).upper() for rec in ref_records]

    # Build padded reference matrix for vectorized comparison
    max_ref_len = max(len(s) for s in ref_seqs_str)
    ref_matrix = np.zeros((len(ref_ids), max_ref_len), dtype=np.uint8)
    for i, seq in enumerate(ref_seqs_str):
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        ref_matrix[i, :len(arr)] = arr

    n_reassigned = 0
    for idx, row in well_df.iterrows():
        cons = row.get("cons_seq")
        if not cons or (isinstance(cons, float) and pd.isna(cons)):
            continue

        cons_upper = cons.upper()

        # Strip flanking regions so we compare only the variable portion
        # against the library entries (which are variable-only).
        if flank_5p_len or flank_3p_len:
            end = len(cons_upper) - flank_3p_len if flank_3p_len else len(cons_upper)
            cons_upper = cons_upper[flank_5p_len:end]

        cons_arr = np.frombuffer(cons_upper.encode(), dtype=np.uint8)

        # Pad or truncate to match ref_matrix width
        if len(cons_arr) < max_ref_len:
            padded = np.zeros(max_ref_len, dtype=np.uint8)
            padded[:len(cons_arr)] = cons_arr
        else:
            padded = cons_arr[:max_ref_len]

        # Vectorized: count matches against all refs at once.
        # Mask out N (ambiguous) positions in the consensus — treat them as
        # wildcards rather than mismatches.  Without this, N at a variant's
        # single-mutation position causes all variants to tie (N != any base),
        # and np.argmax silently picks the first one in FASTA order.
        n_mask = padded != ord('N')  # True where consensus is NOT N
        matches = np.sum((ref_matrix == padded) & n_mask, axis=1)
        best_idx = int(np.argmax(matches))
        best_score = int(matches[best_idx])

        best_ref = ref_ids[best_idx]
        best_seq = ref_seqs_str[best_idx]

        well_df.at[idx, "major_ref"] = best_ref
        well_df.at[idx, "ref_seq"] = best_seq
        well_df.at[idx, "ref_len"] = len(best_seq)
        # Compute match fraction over non-N positions only
        n_comparable = int(np.sum(n_mask))
        well_df.at[idx, "major_freq"] = (
            float(best_score) / n_comparable if n_comparable else 0.0
        )
        n_reassigned += 1

        # Flag ties — when multiple variants score identically, the
        # assignment is unreliable (common in near-identical libraries).
        n_tied = int(np.sum(matches == best_score))
        if n_tied > 1:
            if "assignment_ambiguous" not in well_df.columns:
                well_df["assignment_ambiguous"] = False
            well_df.at[idx, "assignment_ambiguous"] = True

    logger.info("Reassigned %d wells to library variants from consensus", n_reassigned)
    return well_df


def assign_variants_from_reads(
    well_df, read_df, ref_fasta,
    well_fastqs_dir=None,
    minimap2_path=None, workers=4,
    reads_per_well=20,
    full_length_ref_dir=None,
    min_read_len=300,
    min_mapq=0,
):
    """Assign library variants to wells by aligning a sample of reads.

    Samples up to *reads_per_well* reads from each per-well FASTQ and
    aligns them against the multi-variant library FASTA in a single
    minimap2 call. Only a fraction of total reads are aligned, making
    this much faster than aligning everything.

    When *full_length_ref_dir* is provided (directory of per-variant FASTAs
    that include 5' and 3' flanking sequences), a combined full-length
    reference is built and used for alignment instead of *ref_fasta*.  This
    is necessary when the variable insert is very short (e.g. a 6 bp
    negative-control variant) — minimap2 cannot anchor to such short targets
    in long nanopore reads without flanking context.  After assignment the
    flanking sequences are stripped from ``ref_seq`` / ``ref_len`` so
    downstream variant-calling sees only the variable region.

    Args:
        well_df: Per-well summary DataFrame with ``global_well`` column.
        read_df: Per-read DataFrame with ``well_pos`` and ``read_name``.
        ref_fasta: Path to the multi-entry library reference FASTA
            (variable-only sequences).
        well_fastqs_dir: Directory containing per-well FASTQ files.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        workers: Number of minimap2 threads.
        reads_per_well: Max reads to sample from each well for assignment.
        full_length_ref_dir: Directory containing per-variant full-length
            FASTA files (flanks + insert).  When provided, alignment uses
            full-length refs so short inserts are reliably detected.
        min_read_len: Minimum read length to include in variant assignment.
            Reads shorter than this (e.g. concatemer split-reads ~150 bp that
            only cover the 5' flank) are skipped.  Wells whose reads are all
            shorter than this threshold will be marked ``"unassigned"``.
        min_mapq: Minimum mapping quality to accept an alignment.  Reads
            with MAPQ below this threshold are discarded.  minimap2 assigns
            MAPQ=0 when multiple reference targets produce equally-scoring
            alignments, which is common in libraries of near-identical
            variants.  Default is 0 (accept all alignments) since majority
            voting over more reads is more robust than MAPQ filtering for
            near-identical libraries.

    Returns:
        Updated well_df with ``major_ref``, ``ref_seq``, ``ref_len`` columns.
    """
    import tempfile
    import glob as _glob
    from collections import Counter

    if minimap2_path is None:
        minimap2_path = find_minimap2()

    # Load library references for metadata lookup
    ref_records = list(SeqIO.parse(ref_fasta, "fasta"))
    ref_lookup = {rec.id: str(rec.seq) for rec in ref_records}

    # Build a sampled FASTQ: up to reads_per_well reads from each well
    tmp_obj = tempfile.TemporaryDirectory()
    fq_path = os.path.join(tmp_obj.name, "sampled_reads.fastq")
    n_sampled = 0

    if well_fastqs_dir and os.path.isdir(well_fastqs_dir):
        well_fqs = sorted(_glob.glob(os.path.join(well_fastqs_dir, "*.fastq")))
        with open(fq_path, "w") as out:
            for wf in well_fqs:
                count = 0
                with open(wf) as inf:
                    while count < reads_per_well:
                        lines = [inf.readline() for _ in range(4)]
                        if not lines[0]:
                            break
                        if min_read_len and len(lines[1].rstrip()) < min_read_len:
                            continue  # skip concatemer / flank-only reads
                        out.writelines(lines)
                        count += 1
                        n_sampled += 1
    else:
        # Fallback: sample from read_df
        sampled = (read_df[read_df["read_seq"].str.len() >= min_read_len]
                   .groupby("well_pos").head(reads_per_well)
                   if min_read_len else
                   read_df.groupby("well_pos").head(reads_per_well))
        n_sampled = len(sampled)
        with open(fq_path, "w") as fq:
            for _, row in sampled.iterrows():
                fq.write(f"@{row['read_name']}\n{row['read_seq']}\n+\n{row['read_qual']}\n")

    # When full-length refs are available, build a combined FASTA and use it
    # for alignment.  This lets minimap2 anchor on the flanking sequences,
    # which is essential for very short inserts (e.g. a 6 bp GS control)
    # that are too small to align to on their own.
    align_fasta = ref_fasta
    if full_length_ref_dir and os.path.isdir(full_length_ref_dir):
        combined_fl = os.path.join(os.path.dirname(ref_fasta), "full_length_refs.fasta")
        written = 0
        with open(combined_fl, "w") as out_fl:
            for fa_file in sorted(os.listdir(full_length_ref_dir)):
                if not fa_file.endswith(".fasta"):
                    continue
                # Skip stale FASTA files from previous runs whose variant names
                # are no longer in the current library reference.  Without this
                # filter, reads can align to an old name (e.g. "ATF4;25;171")
                # that is absent from ref_lookup, causing the well to be marked
                # "unassigned" even though its reads clearly belong to a known
                # variant ("ATF4.25.171").
                ref_name = fa_file[:-len(".fasta")]
                if ref_name not in ref_lookup:
                    continue
                fa_path = os.path.join(full_length_ref_dir, fa_file)
                with open(fa_path) as inf:
                    out_fl.write(inf.read())
                written += 1
        if written > 0:
            align_fasta = combined_fl

    print(f"Aligning {n_sampled:,} sampled reads ({reads_per_well}/well) "
          f"to {len(ref_records)} library variants ({workers} threads)...")

    mm2_log = os.path.join(os.path.dirname(ref_fasta), "minimap2_variant_assign.log")
    mm2_stderr_fh = open(mm2_log, "w")
    mm2 = subprocess.Popen(
        [minimap2_path, "-x", "map-ont", "--secondary=no",
         "-t", str(workers), align_fasta, fq_path],
        stdout=subprocess.PIPE, stderr=mm2_stderr_fh,
    )

    # Parse PAF output — extract target name and MAPQ for quality filtering.
    # PAF columns: 0=qname 1=qlen 2=qstart 3=qend 4=strand 5=tname
    #              6=tlen 7=tstart 8=tend 9=matches 10=block_len 11=mapq
    # minimap2 assigns MAPQ=0 when the read maps equally well to multiple
    # near-identical references.  Filtering these prevents misattribution
    # in single-substitution libraries.
    read_to_ref = {}
    n_low_mapq = 0
    for raw_line in mm2.stdout:
        parts = raw_line.decode("utf-8", errors="replace").split("\t", 13)
        if len(parts) < 12:
            continue
        try:
            mapq = int(parts[11])
        except (ValueError, IndexError):
            continue
        if mapq < min_mapq:
            n_low_mapq += 1
            continue
        read_name = parts[0].split("|")[0]
        read_to_ref[read_name] = parts[5]  # target name

    mm2.wait()
    mm2_stderr_fh.close()
    if mm2.returncode != 0:
        with open(mm2_log) as f:
            logger.warning("minimap2 variant assignment failed (rc=%d): %s",
                           mm2.returncode, f.read()[:500])
    tmp_obj.cleanup()
    if n_low_mapq:
        print(f"  Filtered {n_low_mapq:,} ambiguous read alignments (MAPQ < {min_mapq})")

    # Build well_pos lookup from read_df
    read_to_well = dict(zip(read_df["read_name"], read_df["well_pos"]))

    # Count per-well variant assignments
    well_counts: dict[str, Counter] = {}
    for read_name, ref_name in read_to_ref.items():
        well = read_to_well.get(read_name)
        if well is None:
            continue
        if well not in well_counts:
            well_counts[well] = Counter()
        well_counts[well][ref_name] += 1

    # Assign majority variant to each well
    if "assignment_confidence" not in well_df.columns:
        well_df["assignment_confidence"] = np.nan
    n_assigned = 0
    n_ambiguous = 0
    for well, counts in well_counts.items():
        best_ref, best_count = counts.most_common(1)[0]
        mask = well_df["global_well"] == well
        if not mask.any():
            continue
        # ref_lookup holds variable-only sequences keyed by variant ID.
        # When full-length refs were used for alignment, best_ref is still
        # the same variant ID (FASTA headers match), so the lookup gives
        # the correct short variable sequence — no stripping needed.
        ref_seq = ref_lookup.get(best_ref, "")
        well_df.loc[mask, "major_ref"] = best_ref
        well_df.loc[mask, "ref_seq"] = ref_seq
        well_df.loc[mask, "ref_len"] = len(ref_seq)
        total = sum(counts.values())
        well_df.loc[mask, "major_freq"] = best_count / total if total else 0

        # Assignment confidence: fraction of aligned reads supporting the
        # majority variant.  Low confidence (e.g. 55/45 split) indicates
        # ambiguous assignment — common in near-identical libraries.
        confidence = best_count / total if total else 0
        well_df.loc[mask, "assignment_confidence"] = confidence
        if len(counts) > 1:
            second_count = counts.most_common(2)[1][1]
            if second_count / total > 0.3:
                n_ambiguous += 1
        n_assigned += 1

    # Wells that got no alignments couldn't be assigned — mark them clearly
    # rather than leaving the internal orient-ref name visible in the output.
    unassigned_mask = ~well_df["major_ref"].isin(ref_lookup)
    if unassigned_mask.any():
        well_df.loc[unassigned_mask, "major_ref"] = "unassigned"
        well_df.loc[unassigned_mask, "ref_seq"] = ""
        well_df.loc[unassigned_mask, "ref_len"] = 0
        well_df.loc[unassigned_mask, "major_freq"] = 0.0
        well_df.loc[unassigned_mask, "assignment_confidence"] = 0.0
        n_unassigned = int(unassigned_mask.sum())
        print(f"  {n_unassigned} wells could not be assigned to any library variant "
              f"(reads too short or no alignment)")

    print(f"  Assigned variants to {n_assigned:,} / {len(well_df):,} wells")
    if n_ambiguous:
        print(f"  {n_ambiguous} wells have ambiguous assignment "
              f"(2nd variant >30% of reads)")
    return well_df


def write_per_well_fastqs(read_df, out_root):
    """Write per-well FASTQ files from the read DataFrame.

    Args:
        read_df: Per-read DataFrame with well_pos, read_name, read_seq, read_qual.
        out_root: Root output directory. FASTQs go to ``out_root/wells/fastqs/``.
    """
    wells_dir = os.path.join(out_root, "wells")
    well_fastqs_dir = os.path.join(wells_dir, "fastqs")
    os.makedirs(well_fastqs_dir, exist_ok=True)

    all_wells = read_df["well_pos"].unique()
    print("Writing per-well fastqs...")
    for well in tqdm(all_wells):
        current = read_df[read_df["well_pos"] == well]
        out_path = os.path.join(well_fastqs_dir, f"{well}.fastq")
        with open(out_path, "w") as f:
            for _, row in current.iterrows():
                f.write(
                    f"@{row['read_name']}\n"
                    f"{row['read_seq']}\n+\n"
                    f"{row['read_qual']}\n"
                )


def _realign_single_consensus(well, cons_seq, ref_fa, ref_mmi, tmp_dir,
                               minimap2_path, samtools_path,
                               consensus_dir=None):
    """Align a single consensus sequence against a reference and return CIGAR.

    Args:
        well: Well identifier string.
        cons_seq: Consensus sequence string.
        ref_fa: Path to reference FASTA.
        ref_mmi: Path to minimap2 index (or ref_fa as fallback).
        tmp_dir: Directory for temporary files.
        minimap2_path: Path to minimap2 binary.
        samtools_path: Path to samtools binary.

    Returns:
        Tuple of (well, cigar_str) — cigar_str may be None on failure.
    """
    cons_fa = os.path.join(tmp_dir, f"{well}_recons.fasta")
    cons_bam = os.path.join(tmp_dir, f"{well}_recons.bam")

    with open(cons_fa, "w") as f:
        f.write(f">{well}_consensus\n{cons_seq}\n")

    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", "-t", "1", ref_mmi, cons_fa],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", cons_bam],
            stdin=mm2.stdout,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        mm2.wait()

        # Add MD tags so get_aligned_pairs(with_seq=True) works in flank checking
        calmd_bam = cons_bam + ".calmd.bam"
        with open(calmd_bam, "wb") as _fh:
            subprocess.run(
                [samtools_path, "calmd", "-b", cons_bam, str(ref_fa)],
                stdout=_fh,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        os.replace(calmd_bam, cons_bam)

    except Exception as e:
        print(f"Re-alignment failed for {well}: {e}")
        return well, None

    cigar_str = None
    try:
        with pysam.AlignmentFile(cons_bam, "rb") as bamfile:
            for read in bamfile:
                if not read.is_unmapped:
                    cigar_str = read.cigarstring
                    break
    except Exception as e:
        print(f"CIGAR extraction failed for {well}: {e}")

    # Overwrite the old orient-ref BAM so _check_flanking_regions
    # reads the correct per-variant alignment
    if consensus_dir is not None and cigar_str is not None:
        import shutil
        dest = os.path.join(consensus_dir, f"{well}_consensus_align.bam")
        shutil.copy2(cons_bam, dest)

    return well, cigar_str


def realign_consensus_to_assigned_refs(
    well_df, reference_dir,
    minimap2_path=None, samtools_path=None, workers=4,
    consensus_dir=None,
):
    """Re-align consensus sequences against their newly assigned references.

    After ``reassign_refs_from_consensus`` swaps ``major_ref`` / ``ref_seq`` /
    ``ref_len`` to the best-matching library variant, the CIGAR string is still
    from the original orient-ref alignment.  This function re-aligns each well's
    consensus to the correct reference so ``extract_matches`` sees a CIGAR that
    matches ``ref_len``.

    Args:
        well_df: DataFrame with ``cons_seq``, ``major_ref``, and ``global_well``.
        reference_dir: Directory containing ``single_ref_fastas/`` subdirectory.
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
        workers: Number of parallel threads.

    Returns:
        Updated well_df with corrected ``CIGAR`` column.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    single_fasta_dir = os.path.join(reference_dir, "single_ref_fastas")
    tmp_dir = os.path.join(reference_dir, "realign_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Collect wells that need re-alignment
    tasks = []
    for _, row in well_df.iterrows():
        cons = row.get("cons_seq")
        if not cons or (isinstance(cons, float) and pd.isna(cons)):
            continue
        well = row["global_well"]
        major_ref = row["major_ref"]
        if ":" in str(major_ref):
            major_ref = major_ref.split(":")[-1]

        ref_fa = os.path.join(single_fasta_dir, f"{major_ref}.fasta")
        if not os.path.exists(ref_fa):
            continue
        ref_mmi = ref_fa + ".mmi"
        if not os.path.exists(ref_mmi):
            ref_mmi = ref_fa
        tasks.append((well, cons, ref_fa, ref_mmi))

    if not tasks:
        return well_df

    # Pre-build any missing minimap2 indexes
    seen_refs = set()
    for _, _, ref_fa, ref_mmi in tasks:
        if ref_mmi == ref_fa and ref_fa not in seen_refs:
            mmi = ref_fa + ".mmi"
            subprocess.run(
                [minimap2_path, "-d", mmi, ref_fa],
                stderr=subprocess.DEVNULL, check=False,
            )
            seen_refs.add(ref_fa)
            # Update tasks to use newly built index
    if seen_refs:
        tasks = [
            (w, c, rf, rf + ".mmi" if rf in seen_refs else rm)
            for w, c, rf, rm in tasks
        ]

    print(f"Re-aligning {len(tasks)} consensus sequences to assigned refs ({workers} workers)...")
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _realign_single_consensus, well, cons, ref_fa, ref_mmi,
                tmp_dir, minimap2_path, samtools_path, consensus_dir,
            ): well
            for well, cons, ref_fa, ref_mmi in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            well, cigar = future.result()
            if cigar is not None:
                results[well] = cigar

    # Apply updated CIGARs
    n_updated = 0
    for well, cigar in results.items():
        well_df.loc[well_df["global_well"] == well, "CIGAR"] = cigar
        n_updated += 1

    logger.info("Re-aligned CIGAR for %d / %d wells", n_updated, len(tasks))
    return well_df


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

    # 1) Write per-well FASTQs if they don't exist yet
    all_wells = read_df['well_pos'].dropna().unique()

    if len(all_wells) == 0:
        logger.warning(
            "generate_per_well_consensus: no wells with classified reads — "
            "skipping per-well consensus generation."
        )
        return well_df

    sample_fq = os.path.join(well_fastqs_dir, f"{all_wells[0]}.fastq")
    if not os.path.exists(sample_fq):
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

    # Pre-index all reference FASTAs so parallel workers don't race on
    # index creation.  Also build minimap2 .mmi indexes for each unique
    # reference to avoid repeated indexing.
    unique_refs = set()
    well_set = set(well_df["global_well"].values)
    for well in all_wells:
        if well not in well_set:
            continue
        mr = well_df.loc[well_df["global_well"] == well, "major_ref"].iloc[0]
        if ":" in mr:
            mr = mr.split(":")[-1]
        unique_refs.add(mr)

    ref_mmi_map = {}
    for ref_name in unique_refs:
        ref_fa = os.path.join(single_fasta_reference_dir, f"{ref_name}.fasta")
        if os.path.exists(ref_fa):
            # samtools faidx (for samtools consensus)
            fai = ref_fa + ".fai"
            if not os.path.exists(fai):
                subprocess.run(
                    [samtools_path, "faidx", ref_fa],
                    stderr=subprocess.DEVNULL, check=False,
                )
            # minimap2 index (avoids per-call re-indexing)
            mmi = ref_fa + ".mmi"
            if not os.path.exists(mmi):
                subprocess.run(
                    [minimap2_path, "-d", mmi, ref_fa],
                    stderr=subprocess.DEVNULL, check=False,
                )
            ref_mmi_map[ref_name] = mmi

    # Pre-compute paths for all wells that have summary data
    well_paths = {}
    n_skipped = 0
    for well in all_wells:
        if well not in well_set:
            continue
        major_ref = well_df.loc[well_df["global_well"] == well, "major_ref"].iloc[0]
        if ":" in major_ref:
            major_ref = major_ref.split(":")[-1]
        ref_fa = os.path.join(single_fasta_reference_dir, f"{major_ref}.fasta")
        if not os.path.exists(ref_fa):
            n_skipped += 1
            continue
        mmi = ref_mmi_map.get(major_ref, ref_fa)
        well_paths[well] = {
            "ref_fa": ref_fa,
            "ref_mmi": mmi,
            "fq": os.path.join(well_fastqs_dir, f"{well}.fastq"),
            "bam": os.path.join(well_consensus_dir, f"{well}.bam"),
            "cons_fa": os.path.join(well_consensus_dir, f"{well}_consensus.fasta"),
            "cons_bam": os.path.join(well_consensus_dir, f"{well}_consensus_align.bam"),
        }

    if n_skipped:
        logger.info("Skipped %d wells with no matching reference FASTA", n_skipped)

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

def parse_vector_fasta(vector_fasta_path: str) -> tuple:
    """Parse a vector FASTA to extract 5' and 3' flanking sequences.

    The vector FASTA should contain a single entry where the variable region
    is replaced with X or N characters (case-insensitive).

    Args:
        vector_fasta_path: Path to vector FASTA file.

    Returns:
        Tuple of (flank_5p, flank_3p) sequences flanking the X/N region.

    Raises:
        ValueError: If zero or multiple variable regions are found, or if the
            FASTA does not contain exactly one entry.
    """
    record = SeqIO.read(vector_fasta_path, "fasta")
    seq = str(record.seq)

    x_regions = list(re.finditer(r"[XxNn]+", seq))
    if len(x_regions) == 0:
        raise ValueError(
            f"No variable region (X or N characters) found in vector FASTA: "
            f"{vector_fasta_path}"
        )
    if len(x_regions) > 1:
        raise ValueError(
            f"Multiple variable regions found in vector FASTA ({len(x_regions)} "
            f"regions). Expected exactly one contiguous run of X or N characters."
        )

    match = x_regions[0]
    flank_5p = seq[: match.start()]
    flank_3p = seq[match.end() :]
    return flank_5p, flank_3p


def _extract_variable_region(cons_seq, ref_seq, flank_5p_len=0, flank_3p_len=0):
    """Extract the variable region from a consensus sequence.

    Tries flank-length trimming first (when available), then falls back to
    string search for the start of the reference sequence.

    Returns the variable portion of cons_seq, or None on failure.
    """
    if not cons_seq or not ref_seq:
        return None
    cons_upper = cons_seq.upper()
    ref_upper = ref_seq.upper()
    ref_len = len(ref_upper)

    # Method 1: trim by known flank lengths
    if flank_5p_len or flank_3p_len:
        end = len(cons_upper) - flank_3p_len if flank_3p_len else len(cons_upper)
        trimmed = cons_upper[flank_5p_len:end]
        if abs(len(trimmed) - ref_len) <= 3:
            return trimmed[:ref_len]

    # Method 2: if consensus is already the right length, use it directly
    if len(cons_upper) == ref_len:
        return cons_upper

    # Method 3: find the variable region by anchoring on the first 20 bp
    start = cons_upper.find(ref_upper[:20])
    if start >= 0 and start + ref_len <= len(cons_upper):
        return cons_upper[start:start + ref_len]

    return None


def _protein_check(ref_seq, cons_var, frame_offset=0):
    """Compare protein translations of reference and consensus variable regions.

    Returns one of: "Match", "Silent", "Missense", "Frameshift", or None if
    comparison is not possible.
    """
    if not ref_seq or not cons_var:
        return None
    ref_upper = ref_seq.upper()
    cons_upper = cons_var.upper()

    # Skip if consensus contains too many N bases (unreliable translation)
    n_count = cons_upper.count('N')
    if n_count > len(cons_upper) * 0.05:
        return None

    # Replace N with the reference base for translation (best-guess)
    cons_for_translation = list(cons_upper)
    for i, c in enumerate(cons_for_translation):
        if c == 'N' and i < len(ref_upper):
            cons_for_translation[i] = ref_upper[i]
    cons_for_translation = ''.join(cons_for_translation)

    # Trim to complete codons from the frame offset
    def _trim_to_codons(seq, offset):
        seq = seq[offset:]
        return seq[:len(seq) - len(seq) % 3]

    try:
        ref_codons = _trim_to_codons(ref_upper, frame_offset)
        cons_codons = _trim_to_codons(cons_for_translation, frame_offset)
        if len(ref_codons) != len(cons_codons):
            return "Frameshift"
        ref_protein = str(Seq(ref_codons).translate())
        cons_protein = str(Seq(cons_codons).translate())
        if ref_protein == cons_protein:
            if ref_codons == cons_codons:
                return "Match"
            return "Silent"
        return "Missense"
    except Exception:
        return None


def _check_column_agreement(
    well: str,
    consensus_dir: str,
    orf_seq: str,
    orf_start: int,
    threshold: float = 0.10,
    min_depth: int = 10,
) -> dict:
    """Check per-column read agreement in the variable region.

    Opens the per-well read BAM and checks each ORF position for reads
    that disagree with the assigned variant's reference.  Positions where
    >*threshold* fraction of reads differ are flagged.

    Args:
        well: Global well identifier (e.g. "1A1").
        consensus_dir: Directory containing per-well BAMs ({well}.bam).
        orf_seq: The assigned variant's ORF sequence (from well_df ref_seq).
        orf_start: 0-based start of the ORF in the BAM reference coordinate
            system (= flank_5p_len).
        threshold: Maximum non-reference fraction allowed (default 0.10).
        min_depth: Minimum read depth at a position to evaluate it.

    Returns:
        Dict with keys: n_flagged_positions, min_agreement, max_mismatch_frac.
    """
    import pysam
    from collections import Counter

    result = {"n_flagged_positions": 0, "min_agreement": 1.0, "max_mismatch_frac": 0.0}
    bam_path = os.path.join(consensus_dir, f"{well}.bam")
    if not os.path.exists(bam_path):
        return result

    orf_len = len(orf_seq)
    try:
        bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
        flagged = 0
        min_agree = 1.0
        max_mismatch = 0.0
        for col in bam.pileup(min_base_quality=0):
            orf_pos = col.reference_pos - orf_start
            if orf_pos < 0 or orf_pos >= orf_len:
                continue
            counts = Counter()
            for read in col.pileups:
                if not read.is_del and not read.is_refskip:
                    base = read.alignment.query_sequence[read.query_position].upper()
                    counts[base] += 1
            total = sum(counts.values())
            if total < min_depth:
                continue
            ref_base = orf_seq[orf_pos].upper()
            ref_count = counts.get(ref_base, 0)
            agreement = ref_count / total
            mismatch_frac = 1.0 - agreement
            if agreement < min_agree:
                min_agree = agreement
            if mismatch_frac > max_mismatch:
                max_mismatch = mismatch_frac
            if mismatch_frac > threshold:
                flagged += 1
        bam.close()
        result["n_flagged_positions"] = flagged
        result["min_agreement"] = round(min_agree, 4)
        result["max_mismatch_frac"] = round(max_mismatch, 4)
    except Exception:
        pass
    return result


def extract_matches(well_df, flank_5p_len: int = 0, flank_3p_len: int = 0,
                    consensus_dir: str = None, frame_offset: int = 0):
    """Extract reference matches using consensus CIGAR string.

    When flank lengths are provided (from --vector-fasta), also checks
    flanking regions for mismatches using the consensus BAM alignment.

    Also performs protein-level comparison of the consensus variable region
    against the assigned variant's reference, stored in a ``protein_check``
    column ("Match", "Silent", "Missense", "Frameshift", or empty).

    Args:
        well_df: Per-well DataFrame with CIGAR, ref_len, ref_seq, cons_seq.
        flank_5p_len: Length of the 5' flanking region (0 = no flank check).
        flank_3p_len: Length of the 3' flanking region (0 = no flank check).
        consensus_dir: Path to consensus BAM directory (required when flanks
            are provided).
        frame_offset: Reading frame offset (0, 1, or 2) for protein translation.

    Returns:
        Updated well_df with cons_check and protein_check columns (and
        flank_check, flank_5p_mismatches, flank_3p_mismatches when flanks
        are provided).
    """
    has_flanks = (flank_5p_len > 0 or flank_3p_len > 0) and consensus_dir

    for index, row in well_df.iterrows():
        # Get CIGAR string, reference length
        well = row['global_well']
        ref_len = row['ref_len']
        ref_seq = row['ref_seq']
        cigar = row['CIGAR']
        cons_seq = row['cons_seq']

        status = ""

        if has_flanks:
            # Use aligned pairs from consensus BAM for per-position analysis
            flank_result = _check_flanking_regions(
                well, ref_len, flank_5p_len, flank_3p_len, consensus_dir,
                frame_offset=frame_offset,
            )
            status = flank_result["cons_check"]
            well_df.at[index, "flank_check"] = flank_result["flank_check"]
            well_df.at[index, "flank_5p_mismatches"] = flank_result["flank_5p_mismatches"]
            well_df.at[index, "flank_3p_mismatches"] = flank_result["flank_3p_mismatches"]
            # Replace the noisy 5-read sampling fraction with the actual
            # per-position variable-region match fraction from the consensus BAM.
            well_df.at[index, "major_freq"] = flank_result["var_match_fraction"]
            well_df.at[index, "var_n_count"] = flank_result["var_n_count"]
        else:
            # Original CIGAR-based logic
            if cigar is None or ref_len is None or pd.isna(ref_len):
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
                        if Seq(ref_seq).translate() == Seq(cons_seq).translate():
                            status = "Silent Mutation"
                    else:
                        status = "Error"

        well_df.at[index, "cons_check"] = status

        # Per-column read agreement check: scan the per-well read BAM for
        # positions where >10% of reads disagree with the reference.
        if has_flanks and status in ("Perfect Match", "Silent Mutation") and ref_seq:
            col_result = _check_column_agreement(
                well, consensus_dir, ref_seq, flank_5p_len,
            )
            well_df.at[index, "n_flagged_positions"] = col_result["n_flagged_positions"]
            well_df.at[index, "max_mismatch_frac"] = col_result["max_mismatch_frac"]

        # Protein-level check: compare translation of the variable region
        # in the consensus against the assigned variant's reference.
        if status == "Perfect Match":
            # BAM-based analysis already confirmed 0 mismatches and 0 indels
            # in the variable region — protein must match.  Skip the less
            # reliable substring extraction which can fail when flank indels
            # shift the consensus coordinates.
            well_df.at[index, "protein_check"] = "Match"
        elif ref_seq and cons_seq and ref_len and not pd.isna(ref_len):
            cons_var = _extract_variable_region(
                cons_seq, ref_seq, flank_5p_len, flank_3p_len,
            )
            pcheck = _protein_check(ref_seq, cons_var, frame_offset)
            well_df.at[index, "protein_check"] = pcheck or ""
        else:
            well_df.at[index, "protein_check"] = ""

    return well_df


def detect_consensus_hotspots(well_df, threshold=0.1, flank_5p_len=0, flank_3p_len=0):
    """Scan wells for systematic mismatch positions in the variable region.

    When a large fraction of wells share the same mismatch at the same
    position, it's usually a library-level issue (synthesis error, PCR
    recombination) rather than random sequencing noise.

    Args:
        well_df: Per-well DataFrame with ref_seq and cons_seq columns.
        threshold: Minimum fraction of wells with a mismatch at a position
            to report it as a hotspot.
        flank_5p_len: Length of the 5' flanking region to strip from consensus.
        flank_3p_len: Length of the 3' flanking region to strip from consensus.

    Returns:
        List of dicts with keys: position, ref_base, alt_base, n_wells,
        fraction, codon_position, aa_position.  Empty list if no hotspots.
    """
    from collections import Counter

    # Collect per-position mismatch counts
    pos_mismatches: dict[int, Counter] = {}
    n_compared = 0

    for _, row in well_df.iterrows():
        ref_seq = row.get("ref_seq")
        cons_seq = row.get("cons_seq")
        if not ref_seq or not cons_seq or (isinstance(ref_seq, float) and pd.isna(ref_seq)):
            continue

        cons_var = _extract_variable_region(
            cons_seq, ref_seq, flank_5p_len, flank_3p_len,
        )
        if cons_var is None:
            continue

        ref_upper = ref_seq.upper()
        n_compared += 1
        for i in range(min(len(ref_upper), len(cons_var))):
            if cons_var[i] != ref_upper[i] and cons_var[i] != 'N':
                if i not in pos_mismatches:
                    pos_mismatches[i] = Counter()
                pos_mismatches[i][cons_var[i]] += 1

    if n_compared == 0:
        return []

    hotspots = []
    for pos, alt_counts in sorted(pos_mismatches.items()):
        for alt_base, count in alt_counts.most_common(1):
            frac = count / n_compared
            if frac >= threshold:
                # Get the reference base from the first available ref_seq
                ref_base = "?"
                for _, row in well_df.iterrows():
                    rs = row.get("ref_seq")
                    if rs and not (isinstance(rs, float) and pd.isna(rs)):
                        if pos < len(rs):
                            ref_base = rs.upper()[pos]
                        break
                hotspots.append({
                    "position": pos,
                    "ref_base": ref_base,
                    "alt_base": alt_base,
                    "n_wells": count,
                    "fraction": round(frac, 3),
                    "codon_position": pos % 3,
                    "aa_position": pos // 3 + 1,
                })

    if hotspots:
        logger.info(
            "Detected %d consensus hotspot(s) affecting >%.0f%% of wells:",
            len(hotspots), threshold * 100,
        )
        for hs in hotspots:
            logger.info(
                "  Position %d (%s→%s): %d/%d wells (%.0f%%), aa %d codon pos %d",
                hs["position"], hs["ref_base"], hs["alt_base"],
                hs["n_wells"], n_compared, hs["fraction"] * 100,
                hs["aa_position"], hs["codon_position"],
            )

    return hotspots


def _check_flanking_regions(
    well: str,
    variable_len,
    flank_5p_len: int,
    flank_3p_len: int,
    consensus_dir: str,
    frame_offset: int = 0,
) -> dict:
    """Analyse flanking and variable regions from a consensus BAM.

    Uses pysam ``get_aligned_pairs(with_seq=True)`` for per-position
    comparison against the full-length reference (flanks + variable).

    Returns:
        Dict with keys: cons_check, flank_check, flank_5p_mismatches,
        flank_3p_mismatches.
    """
    result = {
        "cons_check": "Error",
        "flank_check": "No alignment",
        "flank_5p_mismatches": 0,
        "flank_3p_mismatches": 0,
        "var_match_fraction": 0.0,
        "var_n_count": 0,
    }

    cons_bam = os.path.join(consensus_dir, f"{well}_consensus_align.bam")
    if not os.path.exists(cons_bam):
        return result

    try:
        with pysam.AlignmentFile(cons_bam, "rb", check_sq=False) as bam:
            reads = list(bam.fetch(until_eof=True))
            if not reads:
                return result
            read = reads[0]
    except Exception:
        return result

    if read.is_unmapped:
        return result

    # Get aligned pairs: list of (query_pos, ref_pos, ref_base)
    # Requires MD tag (added by samtools calmd in _realign_single_consensus)
    try:
        pairs = read.get_aligned_pairs(with_seq=True)
    except ValueError:
        return result

    if variable_len is None or pd.isna(variable_len):
        return result

    variable_len = int(variable_len)
    total_ref_len = flank_5p_len + variable_len + flank_3p_len

    # Count mismatches in each region
    flank_5p_mm = 0
    flank_3p_mm = 0
    var_mismatches = 0
    var_matches = 0
    var_indels = 0
    var_n_count = 0

    query_seq = read.query_sequence

    for qpos, rpos, ref_base in pairs:
        if rpos is None:
            # Insertion in query (no ref position)
            if qpos is not None:
                # Determine which region this insertion is associated with
                # (based on surrounding ref positions — skip for simplicity)
                pass
            continue

        if rpos < flank_5p_len:
            # 5' flank region
            if qpos is None:
                # Deletion
                flank_5p_mm += 1
            elif ref_base is not None and ref_base.islower():
                # Lowercase ref_base = mismatch in pysam
                flank_5p_mm += 1
        elif rpos < flank_5p_len + variable_len:
            # Variable region
            if qpos is None:
                var_indels += 1
            elif ref_base is not None and ref_base.islower():
                # Pysam flags this as mismatch. Distinguish N (ambiguous
                # consensus) from a real substitution.
                if query_seq[qpos] in ('N', 'n'):
                    var_n_count += 1
                else:
                    var_mismatches += 1
            else:
                var_matches += 1
        else:
            # 3' flank region
            if qpos is None:
                flank_3p_mm += 1
            elif ref_base is not None and ref_base.islower():
                flank_3p_mm += 1

    # Determine variable region status (same categories as CIGAR logic).
    # N bases in consensus (ambiguous calls) are counted separately from
    # real mismatches — they indicate uncertainty, not a confirmed error.
    if var_mismatches == 0 and var_indels == 0 and (var_matches + var_n_count) == variable_len:
        cons_check = "Perfect Match"
    elif var_indels == 0 and var_matches + var_mismatches + var_n_count == variable_len:
        # Same length — check for silent mutations
        # Extract variable-region subsequences for translation check
        var_query_bases = []
        var_ref_bases = []
        for qpos, rpos, ref_base in pairs:
            if rpos is not None and flank_5p_len <= rpos < flank_5p_len + variable_len:
                if qpos is not None and ref_base is not None:
                    qbase = query_seq[qpos]
                    rbase = ref_base.upper()
                    # Substitute N with reference base for translation
                    # (consistent with _protein_check behavior).
                    if qbase in ('N', 'n'):
                        qbase = rbase
                    var_query_bases.append(qbase)
                    var_ref_bases.append(rbase)
        if len(var_query_bases) == variable_len and len(var_ref_bases) == variable_len:
            try:
                q_protein = Seq("".join(var_query_bases)[frame_offset:]).translate()
                r_protein = Seq("".join(var_ref_bases)[frame_offset:]).translate()
                if q_protein == r_protein:
                    cons_check = "Silent Mutation"
                else:
                    cons_check = "Other Error"
            except Exception:
                cons_check = "Other Error"
        else:
            cons_check = "Other Error"
    else:
        cons_check = "Error"

    # Determine flank status
    has_5p = flank_5p_mm > 0
    has_3p = flank_3p_mm > 0
    if has_5p and has_3p:
        flank_check = "5'+3' mismatch"
    elif has_5p:
        flank_check = "5' mismatch"
    elif has_3p:
        flank_check = "3' mismatch"
    else:
        flank_check = "OK"

    result["cons_check"] = cons_check
    result["flank_check"] = flank_check
    result["flank_5p_mismatches"] = flank_5p_mm
    result["flank_3p_mismatches"] = flank_3p_mm
    result["var_match_fraction"] = (var_matches + var_n_count) / variable_len if variable_len else 0.0
    result["var_n_count"] = var_n_count
    return result

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
