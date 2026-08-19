"""Call a well's variant by translating its consensus and diffing against WT.

The alternative approach aligns a well's reads against every library member and
takes the majority target. In a single-codon scan the members differ by one
base, so the alignment scores are near-identical and the winner is decided by
noise; a well whose sequence is not in the library at all still gets one, since
the aligner has no way to answer "none of these".

This module asks a different question. Reads are aligned to a single wild-type
reference, a per-position consensus is taken over the well's reads, and the
consensus is translated and compared to the wild-type protein. The variant is
read out of the difference rather than chosen from a list, so an unmutated well
comes back as WT and a well carrying something undesigned comes back as itself.

Consensus first, then translate: at nanopore error rates a single read carries
several substitutions across a 294 bp insert, which is far more than the one the
library encodes, so no per-read call is meaningful. Averaging over the well's
depth is what makes the codon legible.
"""

from __future__ import annotations

import collections
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from Bio.Seq import Seq

__all__ = ["WellCall", "call_well", "call_wells", "build_wt_reference",
           "derive_wt_insert"]


def derive_wt_insert(inserts) -> Optional[str]:
    """Recover the unmutated sequence a scan library was built from.

    Every member of a substitution scan differs from the parent at one codon
    and agrees with it everywhere else, so the most common base at each
    position is the parent's -- even though the parent itself is usually not a
    member.

    Args:
        inserts: The library's variable regions, as a sequence or a mapping's
            values.

    Returns:
        The derived sequence, or None if the members are not all one length,
        there are too few to vote, or the result is itself a library member --
        which means this is not a scan and the vote means nothing.
    """
    seqs = list(inserts.values() if hasattr(inserts, "values") else inserts)
    if len(seqs) < 4:
        return None
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        return None

    width = lengths.pop()
    wt = "".join(
        collections.Counter(s[i] for s in seqs).most_common(1)[0][0]
        for i in range(width)
    )

    # In a scan each position is unmutated in almost every member, so the
    # winning base wins overwhelmingly.  A library whose members genuinely
    # differ has no such consensus, and the "wild type" would be a chimera of
    # nothing.
    for i in range(width):
        top = collections.Counter(s[i] for s in seqs).most_common(1)[0][1]
        if top < 0.6 * len(seqs):
            return None
    return wt


@dataclass
class WellCall:
    """What a well's reads say, relative to wild type."""

    well: str
    call: str
    """``"WT"``, a substitution such as ``"Q95*"``, or a description of why
    the insert could not be read (``"deletion(-253)"``, ``"low-coverage"``)."""

    n_reads: int = 0
    n_aligned: int = 0
    median_depth: int = 0
    support: float = 0.0
    """Fraction of reads carrying the consensus codon at the called position;
    1.0 for a WT call means every read agreed at every position."""

    aa_changes: list = field(default_factory=list)
    insert_len: int = 0
    consensus_nt: str = ""

    @property
    def is_clean(self) -> bool:
        """True when a single amino-acid change, or none, explains the well."""
        return self.call == "WT" or len(self.aa_changes) == 1


def build_wt_reference(wt_insert: str, flank_5p: str, flank_3p: str, path):
    """Write the single reference every read is aligned against."""
    with open(path, "w") as fh:
        fh.write(f">wt_construct\n{flank_5p}{wt_insert}{flank_3p}\n")
    return path


def _consensus_codons(bam_path, ref_start, ref_end, min_depth=3):
    """Majority codon at each reference codon in ``[ref_start, ref_end)``.

    Consensus is taken a codon at a time rather than a base at a time, and
    bases the aligner inserted are kept with the reference position they
    follow.  Both matter: where a well's codon differs from the reference at
    two or three of its bases, minimap2 often scores a deletion plus an
    insertion more cheaply than three mismatches, so the read's real bases sit
    in the insertion.  Tallying columns of aligned bases alone discards them
    and the codon comes back a base short.

    Returns ``(codons, depths, agreement)`` per codon, where a codon is ``""``
    if too few reads spanned it.
    """
    import pysam

    n_codons = (ref_end - ref_start) // 3
    tally = [collections.Counter() for _ in range(n_codons)]

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.query_sequence is None:
                continue
            seq = read.query_sequence
            # Bases this read places at each reference position, with any
            # insertion attached to the position it follows.
            at = collections.defaultdict(str)
            last_ref = None
            for qpos, rpos in read.get_aligned_pairs(matches_only=False):
                if rpos is not None:
                    last_ref = rpos
                    if ref_start <= rpos < ref_end and qpos is not None:
                        at[rpos] += seq[qpos]
                elif last_ref is not None and ref_start <= last_ref < ref_end:
                    at[last_ref] += seq[qpos]

            covered = read.get_reference_positions()
            if not covered:
                continue
            lo, hi = covered[0], covered[-1]
            for c in range(n_codons):
                s = ref_start + c * 3
                if s < lo or s + 2 > hi:
                    continue  # read does not span this codon
                tally[c][at[s] + at[s + 1] + at[s + 2]] += 1

    codons, depths, agreement = [], [], []
    for c in range(n_codons):
        counts = tally[c]
        total = sum(counts.values())
        depths.append(total)
        if total < min_depth:
            codons.append("")
            agreement.append(0.0)
            continue
        codon, n = counts.most_common(1)[0]
        codons.append(codon)
        agreement.append(n / total)
    return codons, depths, agreement


def _name_change(consensus_aa: str, wt_aa: str):
    """Describe the protein difference between a consensus and wild type."""
    changes = [(i + 1, wt_aa[i], consensus_aa[i])
               for i in range(min(len(consensus_aa), len(wt_aa)))
               if consensus_aa[i] != wt_aa[i]]
    return changes


def call_well(well, fastq, wt_ref_fasta, insert_start, insert_len, wt_aa,
              minimap2_path=None, samtools_path="samtools", threads=1,
              min_depth=3, tmp_dir=None):
    """Align one well's reads to wild type and read its variant off the consensus."""
    from usortm.demux.utils import find_minimap2

    if minimap2_path is None:
        minimap2_path = find_minimap2()

    n_reads = 0
    with open(fastq) as fh:
        for i, _ in enumerate(fh):
            n_reads = (i + 1) // 4

    if n_reads == 0:
        return WellCall(well=well, call="no-reads")

    cleanup = tmp_dir is None
    tmp = tempfile.mkdtemp() if cleanup else tmp_dir
    bam = os.path.join(tmp, f"{well}.bam")
    try:
        sam = subprocess.run(
            [minimap2_path, "-ax", "map-ont", "--secondary=no",
             "-t", str(threads), wt_ref_fasta, fastq],
            capture_output=True, check=False,
        )
        if sam.returncode != 0:
            return WellCall(well=well, call="align-failed", n_reads=n_reads)
        sort = subprocess.run(
            [samtools_path, "sort", "-o", bam, "-"],
            input=sam.stdout, capture_output=True, check=False,
        )
        if sort.returncode != 0:
            return WellCall(well=well, call="sort-failed", n_reads=n_reads)
        subprocess.run([samtools_path, "index", bam],
                       capture_output=True, check=False)

        codons, depths, agreement = _consensus_codons(
            bam, insert_start, insert_start + insert_len, min_depth)
    finally:
        if cleanup:
            for f in (bam, bam + ".bai"):
                if os.path.exists(f):
                    os.remove(f)
            os.rmdir(tmp) if not os.listdir(tmp) else None

    consensus = "".join(codons)
    covered = [d for d in depths if d > 0]
    median_depth = int(sorted(covered)[len(covered) // 2]) if covered else 0
    n_aligned = max(depths) if depths else 0

    if median_depth < min_depth:
        return WellCall(well=well, call="low-coverage", n_reads=n_reads,
                        n_aligned=n_aligned, median_depth=median_depth,
                        insert_len=len(consensus), consensus_nt=consensus)

    if len(consensus) != insert_len:
        delta = len(consensus) - insert_len
        return WellCall(well=well, call=f"indel({delta:+d})",
                        n_reads=n_reads, n_aligned=n_aligned,
                        median_depth=median_depth, insert_len=len(consensus),
                        consensus_nt=consensus)

    aa = str(Seq(consensus).translate())
    changes = _name_change(aa, wt_aa)

    if not changes:
        call = "WT"
    elif len(changes) == 1:
        pos, ref, alt = changes[0]
        call = f"{ref}{pos}{alt}"
    else:
        call = f"multi({len(changes)})"

    # How cleanly the reads agreed on the codon the call rests on.  For a WT
    # call that is every codon, since any one of them disagreeing would have
    # made it a different call.
    if len(changes) == 1:
        support = agreement[changes[0][0] - 1]
    elif not changes:
        scored = [a for a, d in zip(agreement, depths) if d >= min_depth]
        support = min(scored) if scored else 0.0
    else:
        support = min(agreement[p - 1] for p, _, _ in changes)

    return WellCall(well=well, call=call, n_reads=n_reads,
                    n_aligned=n_aligned, median_depth=median_depth,
                    support=round(support, 3),
                    aa_changes=[f"{r}{p}{a}" for p, r, a in changes],
                    insert_len=len(consensus), consensus_nt=consensus)


def call_wells(well_fastq_dir, wt_insert, flank_5p, flank_3p, out_dir,
               minimap2_path=None, samtools_path="samtools", workers=4,
               min_depth=3, progress_callback=None):
    """Call every well in *well_fastq_dir* against a single wild-type reference."""
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(out_dir, exist_ok=True)
    ref = build_wt_reference(wt_insert, flank_5p, flank_3p,
                             os.path.join(out_dir, "wt_construct.fasta"))
    wt_aa = str(Seq(wt_insert).translate())
    insert_start, insert_len = len(flank_5p), len(wt_insert)

    fastqs = sorted(glob.glob(os.path.join(well_fastq_dir, "*.fastq")))
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(call_well, os.path.basename(f)[:-6], f, ref,
                        insert_start, insert_len, wt_aa, minimap2_path,
                        samtools_path, 1, min_depth): f
            for f in fastqs
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                continue
            done += 1
            if progress_callback and done % 25 == 0:
                progress_callback(done, len(fastqs))
    if progress_callback:
        progress_callback(len(fastqs), len(fastqs))
    return sorted(results, key=lambda r: r.well)
