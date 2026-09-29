"""Work out a read template from a parent vector, rather than drawing one.

A read template (:mod:`usortm.demux.read_template`) is one read with its three
variable spans masked: forward barcode, variable region, reverse barcode.
Drawing it by hand means knowing where the barcoding primers anneal, what
their tails are, and where the variable region starts and stops.  Given the
parent vector, the expected sequences and a sample of the reads, each of those
can be read off instead:

- the variable region is where the expected sequences land in the vector, or,
  when they do not occur in it at all, the stretch of vector the reads skip;
- the amplicon is the stretch of vector the reads align to, since plasmid
  outside the primers is never sequenced;
- the primer tails are what the reads carry beyond each barcode, found by
  locating the LevSeq barcodes themselves in the clipped read ends.

Locating the barcodes also settles which end is which.  The pipeline turns
every read to match the reference and runs Dorado once, so it expects the
forward barcode 5' of the expected sequences.  The mirror-image layout would
demultiplex without complaint and transpose every well, so it is refused
rather than written out.

The result is an ordinary read template, so what was detected can be read,
corrected by hand, and passed back with ``--read-template``.
"""

from __future__ import annotations

import gzip
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from usortm.demux.barcodes import LEVSEQ_FBC, LEVSEQ_RBC

# Bases taken from each end of an expected sequence to find it in the vector.
# Long enough to be unique in a plasmid, short enough that a scan's mutation
# usually falls outside both.
ANCHOR_LENGTH = 20

# Reads sampled to find the amplicon and the primer tails.  A few thousand
# locate both well past the point where adding reads changes the answer.
DEFAULT_SAMPLE_READS = 4000
MIN_SPANNING_READS = 30

# A barcode is placed in a read end when its best match has at most this many
# mismatches and the next-best barcode is clearly worse.
MAX_BARCODE_MISMATCHES = 4
MIN_BARCODE_MARGIN = 3
# How far either side of the expected position a barcode is looked for;
# alignment ends wander a few bases in noisy reads.
SEARCH_SLACK = 12

# Primer-tail consensus: at most this many bases, kept while this share of
# reads agree, and refused if fewer than MIN_OUTER_LENGTH bases survive.
OUTER_LENGTH = 22
MIN_OUTER_LENGTH = 8
CONSENSUS_AGREEMENT = 0.7

# Coverage below this share of the peak is outside the amplicon.
UNCOVERED_FRACTION = 0.05
# When the expected sequences do not occur in the vector, the variable region
# is the vector the reads do not contain -- a stuffer the insert replaced.
# Alignment coverage cannot show it: a read aligned straight through a
# stuffer still matches most of its bases by chance, and one split around it
# often reports only the longer half.  Whether the vector's k-mers occur in
# the reads can: a flank k-mer survives ONT error in most reads and a stuffer
# k-mer in essentially none.  Positions below DIP_FRACTION of the amplicon's
# typical count are the stuffer.
KMER = 15
KMER_READS = 1000
DIP_FRACTION = 0.5
MIN_DIP = 30
# Bases a detected boundary may be moved back by, when the insert happens to
# begin or end with the same bases as the stuffer it replaced.
MAX_JUNCTION_SHIFT = 4

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i


class LayoutError(ValueError):
    """Raised when a read layout cannot be worked out from the vector."""


def _rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _encode(seq: str) -> np.ndarray:
    return _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


@dataclass
class VectorLayout:
    """A read layout detected from a parent vector.

    Coordinates are on :attr:`vector`, which is the input vector turned to
    read forward barcode first and rotated so the amplicon does not cross the
    origin.

    Attributes:
        vector: The oriented, rotated vector.
        amplicon: ``(start, end)`` of the sequenced stretch.
        variable: ``(start, end)`` of the vector the variable region replaces.
        variable_length: Length of the masked variable span in the template,
            the median length of the expected sequences.
        variable_source: ``"expected sequences"`` or ``"reads"``.
        outer_5p: Primer tail before the forward barcode.
        outer_3p: Primer tail after the reverse barcode.
        spacer_5p: Anything between the forward barcode and the vector.
        spacer_3p: Anything between the vector and the reverse barcode.
        barcode_length: Length of the barcode spans.
        reads_sampled: Reads read from the FASTQ.
        reads_spanning: Reads aligned across the whole variable region.
        barcodes_placed: Reads with a barcode found at the 5' and 3' ends.
        inserts_located: ``(located, total)`` expected sequences found in the
            vector.
        reverse_complemented: Whether the vector was turned around.
        rotation: Bases the vector was rotated by.
        orientation_votes: Reads whose barcode shows the forward barcode 5'
            and 3' of the expected sequences.  Only FB01-FB12 and RB01-RB12
            can tell, the rest being shared between the sets, so both are
            zero when a run uses none of them.
    """

    vector: str
    amplicon: tuple
    variable: tuple
    variable_length: int
    variable_source: str
    outer_5p: str
    outer_3p: str
    spacer_5p: str
    spacer_3p: str
    barcode_length: int
    reads_sampled: int
    reads_spanning: int
    barcodes_placed: tuple
    inserts_located: tuple
    reverse_complemented: bool
    rotation: int
    orientation_votes: tuple = (0, 0)

    @property
    def flank_5p(self) -> str:
        return self.vector[self.amplicon[0]:self.variable[0]]

    @property
    def flank_3p(self) -> str:
        return self.vector[self.variable[1]:self.amplicon[1]]

    def template_sequence(self) -> str:
        """The read template, in read order."""
        bc = "N" * self.barcode_length
        return (
            self.outer_5p + bc + self.spacer_5p + self.flank_5p
            + "N" * self.variable_length
            + self.flank_3p + self.spacer_3p + bc + self.outer_3p
        )

    def describe(self) -> list:
        """Lines summarising what was detected, for the console."""
        a0, a1 = self.amplicon
        v0, v1 = self.variable
        located, total = self.inserts_located
        placed_5, placed_3 = self.barcodes_placed
        turned = ", reverse strand" if self.reverse_complemented else ""
        return [
            f"amplicon {a1 - a0:,} bp of a {len(self.vector):,} bp vector{turned}",
            f"variable region {v1 - v0:,} bp of vector at {v0 - a0:,} bp into "
            f"the amplicon, from the {self.variable_source}"
            + (f" ({located}/{total} located)"
               if self.variable_source == "expected sequences" else ""),
            f"5' flank {len(self.flank_5p):,} bp, 3' flank {len(self.flank_3p):,} bp",
            f"primer tails {len(self.outer_5p)} bp and {len(self.outer_3p)} bp, "
            f"from barcodes placed in {placed_5:,} and {placed_3:,} of "
            f"{self.reads_spanning:,} spanning reads",
            (f"forward barcode 5', from {self.orientation_votes[0]:,} reads "
             "with an unshared barcode"
             if self.orientation_votes[0] else
             "barcode orientation not checked: every barcode found is one "
             "the forward and reverse sets share"),
        ]

    def summary(self) -> dict:
        """JSON-serialisable record of the layout, for the run's outputs."""
        return {
            "vector_length": len(self.vector),
            "reverse_complemented": self.reverse_complemented,
            "rotation": self.rotation,
            "amplicon": list(self.amplicon),
            "variable": list(self.variable),
            "variable_length": self.variable_length,
            "variable_source": self.variable_source,
            "flank_5p_length": len(self.flank_5p),
            "flank_3p_length": len(self.flank_3p),
            "outer_5p": self.outer_5p,
            "outer_3p": self.outer_3p,
            "spacer_5p": self.spacer_5p,
            "spacer_3p": self.spacer_3p,
            "barcode_length": self.barcode_length,
            "reads_sampled": self.reads_sampled,
            "reads_spanning": self.reads_spanning,
            "barcodes_placed": list(self.barcodes_placed),
            "inserts_located": list(self.inserts_located),
            "orientation_votes": list(self.orientation_votes),
        }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_vector(path) -> str:
    """Read a parent vector: one FASTA or GenBank record, unmasked.

    Raises:
        LayoutError: If the file does not hold exactly one record, or the
            sequence has anything but A, C, G and T in it.
    """
    from Bio import SeqIO

    path = Path(path)
    fmt = ("genbank" if path.suffix.lower() in {".gb", ".gbk", ".gbff", ".genbank"}
           else "fasta")
    try:
        records = list(SeqIO.parse(str(path), fmt))
    except Exception as exc:
        raise LayoutError(f"{path}: could not be read as {fmt} ({exc}).")
    if len(records) != 1:
        raise LayoutError(
            f"{path}: expected one record, found {len(records)}. A parent "
            "vector is a single sequence."
        )
    seq = str(records[0].seq).upper()
    other = sorted(set(seq) - set("ACGT"))
    if not seq:
        raise LayoutError(f"{path}: the record is empty.")
    if other:
        raise LayoutError(
            f"{path}: the vector contains {''.join(other)}. A parent vector is "
            "the full sequence with nothing masked; a vector with its "
            "variable region marked by X or N is a --vector-fasta, and a read "
            "with all three spans masked is a --read-template."
        )
    return seq


def sample_reads(fastq, n: int = DEFAULT_SAMPLE_READS) -> list:
    """The first *n* reads from a FASTQ, a list of them, or a directory."""
    from usortm.demux.utils import resolve_fastq_inputs

    reads = []
    for path in resolve_fastq_inputs(fastq):
        with open(path, "rb") as fh:
            gz = fh.read(2) == b"\x1f\x8b"
        opener = gzip.open if gz else open
        with opener(path, "rt") as fh:
            while len(reads) < n:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline().strip().upper()
                fh.readline()
                fh.readline()
                if seq:
                    reads.append(seq)
        if len(reads) >= n:
            break
    return reads


# ---------------------------------------------------------------------------
# The variable region, from the expected sequences
# ---------------------------------------------------------------------------

def _unique_position(doubled: str, kmer: str, length: int) -> Optional[int]:
    """Start of *kmer* in a circular sequence, if it occurs exactly once."""
    hits = []
    pos = doubled.find(kmer)
    while pos != -1 and pos < length:
        hits.append(pos)
        if len(hits) > 1:
            return None
        pos = doubled.find(kmer, pos + 1)
    return hits[0] if hits else None


def locate_inserts(vector: str, inserts: Sequence[str],
                   anchor: int = ANCHOR_LENGTH) -> Optional[tuple]:
    """Where the expected sequences sit in the vector.

    Each sequence votes with its first and last *anchor* bases, wherever they
    occur exactly once in the circular vector on either strand.  Anchoring on
    the ends rather than aligning whole sequences is what lets a substitution
    scan be located: every member agrees with the parent at both ends except
    the few mutated there, and those are outvoted.

    Args:
        vector: The vector, as given.
        inserts: The expected sequences' variable regions.
        anchor: Bases taken from each end.

    Returns:
        ``(strand, start, end, located)``, with coordinates on the vector if
        *strand* is ``"+"`` and on its reverse complement if ``"-"``, and
        *end* past *start* even when the span crosses the origin.  None if
        fewer than half the sequences agree on both ends.
    """
    length = len(vector)
    frames = {"+": vector + vector, "-": _rc(vector) + _rc(vector)}
    starts: Counter = Counter()
    ends: Counter = Counter()
    located = 0
    usable = [s.upper() for s in inserts if len(s) >= 2 * anchor]
    for insert in usable:
        hit = False
        for strand, doubled in frames.items():
            s = _unique_position(doubled, insert[:anchor], length)
            if s is not None:
                starts[(strand, s)] += 1
                hit = True
            e = _unique_position(doubled, insert[-anchor:], length)
            if e is not None:
                ends[(strand, (e + anchor) % length)] += 1
                hit = True
        located += hit

    if not usable or not starts or not ends:
        return None
    (strand, start), n_start = starts.most_common(1)[0]
    same_strand = [(k, v) for k, v in ends.items() if k[0] == strand]
    if not same_strand:
        return None
    (_, end), n_end = max(same_strand, key=lambda kv: kv[1])
    if min(n_start, n_end) < 0.5 * len(usable):
        return None
    if end <= start:
        end += length
    return strand, start, end, located


# ---------------------------------------------------------------------------
# The variable region, from the reads, when the vector does not contain it
# ---------------------------------------------------------------------------

def _kmer_codes(seq: str, k: int = KMER) -> np.ndarray:
    """Every k-mer of *seq* as an integer; k-mers holding N are dropped."""
    codes = _encode(seq).astype(np.int64)
    if len(codes) < k:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    keep = (windows < 4).all(axis=1)
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    return (windows[keep] * powers).sum(axis=1)


class _KmerTable:
    """How many times each k-mer occurs across a set of reads, both strands."""

    def __init__(self, reads: Sequence[str], k: int = KMER):
        self.k = k
        parts = [_kmer_codes(r, k) for r in reads]
        parts += [_kmer_codes(_rc(r), k) for r in reads]
        allk = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        self.keys, self.counts = np.unique(allk, return_counts=True)

    def count(self, seq: str) -> np.ndarray:
        """Occurrences of each k-mer of *seq*, by start position."""
        codes = _kmer_codes(seq, self.k)
        if not len(self.keys) or not len(codes):
            return np.zeros(len(codes), dtype=np.int64)
        idx = np.searchsorted(self.keys, codes).clip(0, len(self.keys) - 1)
        return np.where(self.keys[idx] == codes, self.counts[idx], 0)


def _variable_from_reads(vector: str, lo: int, hi: int, reads: Sequence[str],
                         inserts: Sequence[str]) -> Optional[tuple]:
    """``(start, end)`` of the vector the reads never contain, inside the amplicon."""
    table = _KmerTable(reads[:KMER_READS])
    k = table.k
    counts = table.count(vector[lo:hi])
    if not len(counts):
        return None
    typical = np.percentile(counts, 90)
    dip = _longest_run(counts < DIP_FRACTION * typical, circular=False)
    if dip is None:
        return None
    # A k-mer starting up to k-1 bases before the stuffer already holds some
    # of it, so the dip in k-mer starts opens k-1 bases early.
    v0 = lo + dip[0] + k - 1
    v1 = lo + dip[0] + dip[1]
    if v1 - v0 < MIN_DIP:
        return None

    # If an insert begins with the base the stuffer began with, that base
    # reads as flank and the reference would carry it twice.  The reads decide:
    # of the candidate junctions, keep the one whose spanning k-mers they hold.
    half = k // 2

    def _score(junctions):
        return int(sum(table.count(j).sum() for j in junctions if len(j) == k))

    best_5 = max(range(MAX_JUNCTION_SHIFT + 1), key=lambda j: (
        _score([vector[v0 - j - half:v0 - j] + ins[:k - half] for ins in inserts]), -j))
    best_3 = max(range(MAX_JUNCTION_SHIFT + 1), key=lambda j: (
        _score([ins[-(k - half):] + vector[v1 + j:v1 + j + half] for ins in inserts]), -j))
    return v0 - best_5, v1 + best_3


# ---------------------------------------------------------------------------
# Reads against the vector
# ---------------------------------------------------------------------------

@dataclass
class _Hit:
    read: int
    qlen: int
    qstart: int
    qend: int
    strand: str
    tstart: int
    tend: int
    matches: int
    # Vector intervals the read matches base for base.  A read carrying an
    # insert the vector lacks can align straight through the stuffer as one
    # long mismatch, so the alignment's ends alone would hide the gap.
    matched: tuple = ()


def _align(reads: Sequence[str], target: str, minimap2_path: str,
           threads: int, workdir: Path) -> list:
    """Align reads to one target sequence; primary and supplementary hits."""
    workdir.mkdir(parents=True, exist_ok=True)
    target_fa = workdir / "vector.fa"
    reads_fa = workdir / "reads.fa"
    target_fa.write_text(f">vector\n{target}\n")
    with open(reads_fa, "w") as fh:
        for i, seq in enumerate(reads):
            fh.write(f">{i}\n{seq}\n")
    proc = subprocess.run(
        [minimap2_path, "-x", "map-ont", "--secondary=no", "-c", "--eqx",
         "-t", str(threads),
         str(target_fa), str(reads_fa)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise LayoutError(
            f"minimap2 failed aligning reads to the vector: {proc.stderr.strip()}"
        )
    hits = []
    for line in proc.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 12:
            continue
        cigar = next((t[5:] for t in f[12:] if t.startswith("cg:Z:")), "")
        hits.append(_Hit(read=int(f[0]), qlen=int(f[1]), qstart=int(f[2]),
                         qend=int(f[3]), strand=f[4], tstart=int(f[7]),
                         tend=int(f[8]), matches=int(f[9]),
                         matched=_matched_intervals(cigar, int(f[7]))))
    return hits


_CIGAR_OP = re.compile(r"(\d+)([=XIDNSHM])")


def _matched_intervals(cigar: str, tstart: int) -> tuple:
    """Target intervals the CIGAR matches base for base."""
    out, pos = [], tstart
    for n, op in _CIGAR_OP.findall(cigar):
        n = int(n)
        if op == "=":
            out.append((pos, pos + n))
            pos += n
        elif op in "XDNM":
            pos += n
    return tuple(out)


def _coverage(hits: Iterable[_Hit], length: int) -> np.ndarray:
    """Per-base count of reads matching the vector there."""
    diff = np.zeros(length + 1, dtype=np.int64)
    for h in hits:
        for s, e in h.matched or ((h.tstart, h.tend),):
            diff[s] += 1
            diff[e] -= 1
    return np.cumsum(diff[:-1])


def _longest_run(mask: np.ndarray, circular: bool) -> Optional[tuple]:
    """``(start, length)`` of the longest run of True, or None."""
    if not mask.any():
        return None
    n = len(mask)
    if mask.all():
        return 0, n
    seq = np.concatenate([mask, mask]) if circular else mask
    best, best_start, run, run_start = 0, 0, 0, 0
    for i, v in enumerate(seq):
        if v:
            if run == 0:
                run_start = i
            run += 1
            if run > best:
                best, best_start = run, run_start
        else:
            run = 0
    return best_start % n, min(best, n)


def _read_spans(hits: Sequence[_Hit]) -> dict:
    """Per read: strand, and the outermost vector and read coordinates.

    Read coordinates are on the read turned to match the vector, so a read's
    5' clipped end is always ``read[:qstart]``.
    """
    by_read = defaultdict(list)
    for h in hits:
        by_read[h.read].append(h)
    spans = {}
    for read, group in by_read.items():
        weight = Counter()
        for h in group:
            weight[h.strand] += h.matches
        strand = weight.most_common(1)[0][0]
        group = [h for h in group if h.strand == strand]
        qlen = group[0].qlen
        if strand == "+":
            qs = [(h.tstart, h.qstart) for h in group]
            qe = [(h.tend, h.qend) for h in group]
        else:
            qs = [(h.tstart, qlen - h.qend) for h in group]
            qe = [(h.tend, qlen - h.qstart) for h in group]
        tstart, qstart = min(qs)
        tend, qend = max(qe)
        spans[read] = (strand, tstart, tend, qstart, qend)
    return spans


def _mode(values: Sequence[int]) -> int:
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Barcodes and primer tails, from the clipped read ends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Catalog:
    """The barcodes one read end can carry, each distinct sequence once.

    The two LevSeq sets overlap: RB13-RB96 are the same sequences as
    FB13-FB96, and RB01-RB12 are FB01-FB12 reverse-complemented.  Listed per
    set, a shared barcode would be its own runner-up and never place, so each
    sequence appears once with every set it belongs to.
    """

    matrix: np.ndarray
    labels: tuple  # per row, the frozenset of set names holding it


def _catalog(sets: dict) -> _Catalog:
    seqs, labels = [], []
    index = {}
    for name, members in sets.items():
        for seq in members:
            if seq in index:
                labels[index[seq]] = labels[index[seq]] | {name}
            else:
                index[seq] = len(seqs)
                seqs.append(seq)
                labels.append(frozenset({name}))
    return _Catalog(np.stack([_encode(s) for s in seqs]), tuple(labels))


def _place_barcode(window: str, catalog: _Catalog) -> Optional[tuple]:
    """Best barcode in *window*: ``(sets, offset)`` or None.

    Refused unless every other barcode scores at least MIN_BARCODE_MARGIN
    worse.  The winner itself a base or two over is not a competitor -- an
    indel near the barcode's edge produces exactly that.
    """
    width = catalog.matrix.shape[1]
    if len(window) < width:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(_encode(window), width)
    mism = (frames[:, None, :] != catalog.matrix[None, :, :]).sum(axis=2)
    off, idx = np.unravel_index(np.argmin(mism), mism.shape)
    score = int(mism[off, idx])
    if score > MAX_BARCODE_MISMATCHES:
        return None
    others = mism.copy()
    others[max(0, off - 2):off + 3, idx] = width
    if int(others.min()) - score < MIN_BARCODE_MARGIN:
        return None
    return catalog.labels[idx], int(off)


def _consensus(seqs: Sequence[str], anchored_right: bool) -> str:
    """Column consensus, kept outward from the anchor while reads agree."""
    if not seqs:
        return ""
    cols = []
    for i in range(OUTER_LENGTH):
        column = [s[-1 - i] if anchored_right else s[i]
                  for s in seqs if len(s) > i]
        if len(column) < max(5, 0.3 * len(seqs)):
            break
        base, n = Counter(column).most_common(1)[0]
        if n / len(column) < CONSENSUS_AGREEMENT:
            break
        cols.append(base)
    out = "".join(cols)
    return out[::-1] if anchored_right else out


@dataclass
class _PrimerEnds:
    outer_5p: str
    outer_3p: str
    spacer_5p: str
    spacer_3p: str
    placed_5: int
    placed_3: int
    # Reads whose barcode belongs to one set only, counted by the layout it
    # supports: forward barcode 5' (standard) or reverse barcode 5' (flipped).
    standard: int
    flipped: int


def _primer_ends(reads, spans, amplicon, barcode_length) -> _PrimerEnds:
    """Locate barcodes in the clipped ends and read the primer tails.

    With the read turned to match the vector, the 5' end carries a barcode
    as written and the 3' end one reverse-complemented.  Which set each came
    from is only knowable for the barcodes the sets do not share -- FB01-FB12
    and RB01-RB12 -- so only those vote on orientation.
    """
    a0, a1 = amplicon
    cat_5 = _catalog({"fbc": LEVSEQ_FBC, "rbc": LEVSEQ_RBC})
    cat_3 = _catalog({"fbc": [_rc(s) for s in LEVSEQ_FBC],
                      "rbc": [_rc(s) for s in LEVSEQ_RBC]})
    reach = barcode_length + SEARCH_SLACK

    votes = Counter()
    placed_5 = placed_3 = 0
    tails_5, tails_3, gaps_5, gaps_3 = [], [], [], []
    spacer_5, spacer_3 = [], []
    for read, (strand, tstart, tend, qstart, qend) in spans.items():
        seq = reads[read] if strand == "+" else _rc(reads[read])
        # Where the amplicon edge falls in this read, allowing for an
        # alignment that stopped a few bases short of it.
        edge_5 = qstart - (tstart - a0)
        edge_3 = qend + (a1 - tend)
        # The primer tail can be shorter than the search reach, so the
        # window is clamped to the read rather than the read skipped.
        if edge_5 > 0:
            lo = max(0, edge_5 - reach)
            hit = _place_barcode(seq[lo:edge_5 + SEARCH_SLACK], cat_5)
            if hit:
                sets, off = hit
                placed_5 += 1
                if len(sets) == 1:
                    votes["standard" if "fbc" in sets else "flipped"] += 1
                b0 = lo + off
                tails_5.append(seq[max(0, b0 - OUTER_LENGTH):b0])
                gap = edge_5 - (b0 + barcode_length)
                gaps_5.append(gap)
                spacer_5.append(seq[b0 + barcode_length:edge_5] if gap > 0 else "")
        if edge_3 < len(seq):
            lo = max(0, edge_3 - SEARCH_SLACK)
            hit = _place_barcode(seq[lo:edge_3 + reach], cat_3)
            if hit:
                sets, off = hit
                placed_3 += 1
                if len(sets) == 1:
                    votes["standard" if "rbc" in sets else "flipped"] += 1
                b0 = lo + off
                b1 = b0 + barcode_length
                tails_3.append(seq[b1:b1 + OUTER_LENGTH])
                gap = b0 - edge_3
                gaps_3.append(gap)
                spacer_3.append(seq[edge_3:b0] if gap > 0 else "")

    if placed_5 < 10 or placed_3 < 10:
        raise LayoutError(
            f"LevSeq barcodes were found at the amplicon ends of only "
            f"{placed_5} and {placed_3} reads. The reads may not carry LevSeq "
            "barcodes, or the vector may not be the construct that was "
            "sequenced. Pass --read-template to give the layout directly."
        )

    def _spacer(gaps, seqs, anchored_right):
        if not gaps or statistics.median(gaps) < 2:
            return ""
        # Consensus-called like the tails, anchored on the barcode side.
        return _consensus([s for s in seqs if s], anchored_right=anchored_right)

    return _PrimerEnds(
        outer_5p=_consensus(tails_5, anchored_right=True),
        outer_3p=_consensus(tails_3, anchored_right=False),
        spacer_5p=_spacer(gaps_5, spacer_5, anchored_right=False),
        spacer_3p=_spacer(gaps_3, spacer_3, anchored_right=True),
        placed_5=placed_5,
        placed_3=placed_3,
        standard=votes["standard"],
        flipped=votes["flipped"],
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_layout(
    vector_path,
    inserts: Sequence[str],
    fastq,
    minimap2_path: str,
    workdir,
    threads: int = 4,
    n_reads: int = DEFAULT_SAMPLE_READS,
) -> VectorLayout:
    """Work out the read layout from a parent vector.

    Args:
        vector_path: FASTA or GenBank file holding the parent vector.
        inserts: The expected sequences' variable regions.
        fastq: The run's reads: a FASTQ, a list of them, or a directory.
        minimap2_path: minimap2 executable.
        workdir: Scratch directory for the alignments.
        threads: minimap2 threads.
        n_reads: Reads to sample.

    Returns:
        VectorLayout.

    Raises:
        LayoutError: If any part of the layout cannot be found, or the
            barcodes sit the wrong way round for the pipeline.
    """
    workdir = Path(workdir)
    vector = load_vector(vector_path)
    length = len(vector)
    inserts = [s.upper() for s in inserts if s]
    if not inserts:
        raise LayoutError("No expected sequences to locate in the vector.")
    variable_length = int(statistics.median(len(s) for s in inserts))

    reads = sample_reads(fastq, n_reads)
    if not reads:
        raise LayoutError(f"No reads found in {fastq}.")

    located = locate_inserts(vector, inserts)
    reverse = bool(located and located[0] == "-")
    if reverse:
        vector = _rc(vector)

    # Rotate so the amplicon does not cross the origin: the longest stretch
    # the reads never reach is outside it, so start the vector in the middle
    # of that.  A vector covered end to end is taken as linear.
    hits = _align(reads, vector, minimap2_path, threads, workdir / "pass1")
    aligned = len({h.read for h in hits})
    if aligned < MIN_SPANNING_READS:
        raise LayoutError(
            f"Only {aligned} of {len(reads):,} sampled reads align to the "
            "vector. Check that it is the construct that was sequenced."
        )
    cov = _coverage(hits, length)
    gap = _longest_run(cov < UNCOVERED_FRACTION * cov.max(), circular=True)
    rotation = (gap[0] + gap[1] // 2) % length if gap else 0
    vector = vector[rotation:] + vector[:rotation]
    hits = _align(reads, vector, minimap2_path, threads, workdir / "pass2")
    cov = _coverage(hits, length)

    covered = np.flatnonzero(cov >= UNCOVERED_FRACTION * cov.max())
    lo, hi = int(covered[0]), int(covered[-1]) + 1

    if located:
        _, start, end, n_located = located
        v0 = (start - rotation) % length
        v1 = v0 + (end - start)
        source = "expected sequences"
    else:
        n_located = 0
        found = _variable_from_reads(vector, lo, hi, reads, inserts)
        if found is None:
            raise LayoutError(
                "The expected sequences do not occur in the vector, and the "
                "reads contain the whole amplicon, so there is no way to tell "
                "where the variable region is. Pass --read-template to give "
                "the layout directly."
            )
        v0, v1 = found
        source = "reads"
    if not (0 <= v0 < v1 <= length):
        raise LayoutError(
            "The variable region crosses the point the vector was rotated at; "
            "the reads do not cover the vector the way an amplicon would."
        )

    spans = _read_spans(hits)
    spanning = {r: s for r, s in spans.items() if s[1] <= v0 and s[2] >= v1}
    if len(spanning) < MIN_SPANNING_READS:
        raise LayoutError(
            f"Only {len(spanning)} sampled reads span the variable region "
            f"({v0}-{v1} on the vector); at least {MIN_SPANNING_READS} are "
            "needed to find the amplicon ends."
        )
    a0 = _mode([s[1] for s in spanning.values()])
    a1 = _mode([s[2] for s in spanning.values()])
    if not (a0 < v0 and v1 < a1):
        raise LayoutError(
            f"The amplicon ({a0}-{a1}) does not contain the variable region "
            f"({v0}-{v1}); the vector does not match the reads."
        )

    barcode_length = len(LEVSEQ_FBC[0])
    ends = _primer_ends(reads, spanning, (a0, a1), barcode_length)
    if ends.flipped > ends.standard:
        raise LayoutError(
            "The reverse barcode is 5' of the expected sequences and the "
            "forward barcode 3'. The pipeline turns every read to match the "
            "expected sequences before reading barcodes, so this layout would "
            "swap the two and put every read in the wrong well. Give the "
            "expected sequences as the reverse complement."
        )
    if ends.flipped and ends.flipped > 0.2 * ends.standard:
        raise LayoutError(
            "The barcodes found at the amplicon ends disagree about which end "
            f"is which ({ends.standard} reads read forward barcode first, "
            f"{ends.flipped} reverse barcode first). Pass --read-template to "
            "give the layout directly."
        )
    outer_5p, outer_3p = ends.outer_5p, ends.outer_3p
    spacer_5p, spacer_3p = ends.spacer_5p, ends.spacer_3p

    # Alignments tend to stop a few bases short of the primer site, which
    # leaves vector between the barcode and the amplicon edge looking like a
    # spacer.  Whatever of it matches the vector is the amplicon's.
    while spacer_5p and a0 > 0 and vector[a0 - 1] == spacer_5p[-1]:
        spacer_5p, a0 = spacer_5p[:-1], a0 - 1
    while spacer_3p and a1 < length and vector[a1] == spacer_3p[0]:
        spacer_3p, a1 = spacer_3p[1:], a1 + 1

    for label, tail in (("5'", outer_5p), ("3'", outer_3p)):
        if len(tail) < MIN_OUTER_LENGTH:
            raise LayoutError(
                f"The {label} primer tail could not be read from the reads "
                f"(only {len(tail)} bp agree beyond the barcode). Pass "
                "--read-template to give the layout directly."
            )

    return VectorLayout(
        vector=vector,
        amplicon=(a0, a1),
        variable=(v0, v1),
        variable_length=variable_length,
        variable_source=source,
        outer_5p=outer_5p,
        outer_3p=outer_3p,
        spacer_5p=spacer_5p,
        spacer_3p=spacer_3p,
        barcode_length=barcode_length,
        reads_sampled=len(reads),
        reads_spanning=len(spanning),
        barcodes_placed=(ends.placed_5, ends.placed_3),
        orientation_votes=(ends.standard, ends.flipped),
        inserts_located=(n_located, len(inserts)),
        reverse_complemented=reverse,
        rotation=rotation,
    )


def write_read_template(layout: VectorLayout, path, source: str = "") -> Path:
    """Write the detected layout as a read template FASTA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = f" from {source}" if source else ""
    path.write_text(f">derived_read_template{origin}\n{layout.template_sequence()}\n")
    return path
