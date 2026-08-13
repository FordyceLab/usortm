"""Count library sequencing reads per variant.

Aligns a shallow sequencing run of the amplified library against the
starting variant list and tallies how many reads each variant claims.

Two details drive the design:

*Coverage is judged against the reference, not the read.*  An amplicon read
carries vector context beyond the variable region, so it legitimately
overhangs its reference.  The filter asks whether the alignment spans most
of the *reference*.

*A read only counts when its assignment is unambiguous.*  minimap2 is asked
to report secondary alignments so the best hit can be compared against the
best hit on a different variant; reads that do not clear the margin are
tallied as ambiguous rather than forced onto their nominal best match.

Needs only the minimap2 binary — no dorado, samtools, or pysam — so this
runs on machines that cannot run the full demux pipeline.
"""
from __future__ import annotations

import gzip
import logging
import os
import re
import subprocess
from collections import OrderedDict

from usortm.demux.deps import find_minimap2
from usortm.qc.resolve import read_variant_sequences
from usortm.qc.skew import VariantCounts

logger = logging.getLogger(__name__)

# SAM FLAG bits
_FLAG_UNMAPPED = 0x4
_FLAG_SUPPLEMENTARY = 0x800

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
# CIGAR operations that advance along the reference.
_REF_CONSUMING = frozenset("MDN=X")

__all__ = ["count_variant_reads", "count_fastq_reads", "write_reference_fasta"]


def count_fastq_reads(fastq) -> int:
    """Count reads in a FASTQ, gzipped or plain.

    Detects gzip by magic bytes rather than extension, matching
    :func:`usortm.demux.utils._open_fastq`.
    """
    with open(fastq, "rb") as probe:
        opener = gzip.open if probe.read(2) == b"\x1f\x8b" else open
    with opener(fastq, "rb") as fh:
        return sum(1 for i, _ in enumerate(fh) if i % 4 == 0)


def write_reference_fasta(variants_csv, fasta_path) -> dict:
    """Write a multi-entry reference FASTA from a variant CSV.

    Returns the name -> sequence mapping that was written.
    """
    sequences = read_variant_sequences(variants_csv)
    os.makedirs(os.path.dirname(str(fasta_path)) or ".", exist_ok=True)
    with open(fasta_path, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n{seq}\n")
    logger.info("Wrote %d references to %s", len(sequences), fasta_path)
    return sequences


def _ref_span(cigar: str) -> int:
    """Reference bases spanned by a CIGAR string."""
    if not cigar or cigar == "*":
        return 0
    return sum(
        int(length)
        for length, op in _CIGAR_RE.findall(cigar)
        if op in _REF_CONSUMING
    )


def _alignment_score(fields) -> int:
    """Extract the minimap2 AS:i: tag, or 0 when absent."""
    for field in fields[11:]:
        if field.startswith("AS:i:"):
            try:
                return int(field[5:])
            except ValueError:
                return 0
    return 0


def _resolve_group(records, ref_lengths, min_ref_cov, margin):
    """Decide what a single read's alignments amount to.

    Args:
        records: List of (rname, cigar, score) for one read; empty when
            the read had no alignment.
        ref_lengths: Reference name -> length.
        min_ref_cov: Minimum fraction of the reference the alignment must span.
        margin: Required relative score lead over the best other reference.

    Returns:
        ("assigned", ref_name) | ("ambiguous", None) | ("low_cov", None)
        | ("unmapped", None)
    """
    if not records:
        return "unmapped", None

    # Best alignment per reference, so multiple hits to one variant do not
    # look like competition between variants.
    best_per_ref = {}
    for rname, cigar, score in records:
        prev = best_per_ref.get(rname)
        if prev is None or score > prev[1]:
            best_per_ref[rname] = (cigar, score)

    best_ref = max(best_per_ref, key=lambda r: best_per_ref[r][1])
    best_cigar, best_score = best_per_ref[best_ref]

    ref_len = ref_lengths.get(best_ref, 0)
    if ref_len > 0 and (_ref_span(best_cigar) / ref_len) < min_ref_cov:
        return "low_cov", None

    runner_up = max(
        (score for ref, (_, score) in best_per_ref.items() if ref != best_ref),
        default=None,
    )
    if runner_up is not None and best_score > 0:
        if (best_score - runner_up) / best_score < margin:
            return "ambiguous", None

    return "assigned", best_ref


def count_variant_reads(
    fastq,
    variants_csv,
    work_dir,
    *,
    min_ref_cov: float = 0.8,
    margin: float = 0.02,
    threads: int = 4,
    minimap2_path=None,
    progress_callback=None,
    total_reads=None,
) -> VariantCounts:
    """Align library reads to the variant list and count per-variant hits.

    Args:
        fastq: Library sequencing reads (plain or gzipped).
        variants_csv: CSV with Name and Sequence columns.
        work_dir: Directory for the reference FASTA and minimap2 log.
        min_ref_cov: Minimum fraction of a reference an alignment must
            span to count.
        margin: Minimum relative alignment-score lead the best variant
            must hold over the best other variant.
        threads: minimap2 threads.
        minimap2_path: Path to minimap2; auto-detected if None.
        progress_callback: Called as ``(n_reads_done, total_reads)``.
        total_reads: Denominator for progress reporting, if known.

    Returns:
        VariantCounts with an entry for every variant, including zeros.

    Raises:
        subprocess.CalledProcessError: If minimap2 exits non-zero.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()

    work_dir = str(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    ref_fasta = os.path.join(work_dir, "library_reference.fasta")
    sequences = write_reference_fasta(variants_csv, ref_fasta)

    # Every variant gets an entry so zero-count variants survive into the
    # statistics, where they carry the dropout signal.
    counts = OrderedDict((name, 0) for name in sequences)
    ref_lengths = {name: len(seq) for name, seq in sequences.items()}

    cmd = [
        minimap2_path, "-ax", "map-ont",
        "--secondary=yes", "-N", "5", "-p", "0.6",
        "-t", str(threads),
        ref_fasta, str(fastq),
    ]
    log_path = os.path.join(work_dir, "minimap2.log")
    logger.info("Counting reads per variant; minimap2 stderr -> %s", log_path)

    tallies = {"ambiguous": 0, "unmapped": 0, "low_cov": 0}
    n_reads = 0
    current_qname = None
    records = []

    def flush():
        """Resolve the alignments accumulated for one read."""
        nonlocal n_reads
        if current_qname is None:
            return
        outcome, ref = _resolve_group(records, ref_lengths, min_ref_cov, margin)
        if outcome == "assigned":
            counts[ref] += 1
        else:
            tallies[outcome] += 1
        n_reads += 1
        if progress_callback is not None and n_reads % 5000 == 0:
            progress_callback(n_reads, total_reads)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fh)
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith("@"):
                continue  # SAM header; reference lengths come from the FASTA

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue

            qname, flag = fields[0], int(fields[1])
            if flag & _FLAG_SUPPLEMENTARY:
                continue  # chimeric fragment, not an independent assignment

            # minimap2 emits a read's alignments consecutively and in input
            # order, so grouping on consecutive names needs no buffering.
            if qname != current_qname:
                flush()
                current_qname = qname
                records = []

            if flag & _FLAG_UNMAPPED:
                continue
            records.append((fields[2], fields[5], _alignment_score(fields)))

        flush()
        proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    if progress_callback is not None:
        progress_callback(n_reads, total_reads or n_reads)

    logger.info(
        "Counted %d reads: %d assigned, %d ambiguous, %d low-coverage, %d unmapped",
        n_reads, sum(counts.values()),
        tallies["ambiguous"], tallies["low_cov"], tallies["unmapped"],
    )

    return VariantCounts(
        counts=dict(counts),
        ambiguous=tallies["ambiguous"],
        unmapped=tallies["unmapped"],
        low_cov=tallies["low_cov"],
        total_reads=n_reads,
    )
