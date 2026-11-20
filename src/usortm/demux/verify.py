"""Check each well of a demultiplexed plate against what it should hold.

The pipeline names each well's variant by assigning it to the closest library
member, and for a check that is the wrong question: a well holding something
the plate does not contain still gets the nearest name, and a scan library's
translation assignment gives up on a well with more than one change, leaving
no consensus at all.  So each well is instead tested against its own
expectation: its reads are aligned to the construct it should hold, flanks
and all, and the consensus insert compared with the expected one directly.
The pipeline's assignment is used only to name what is there when it is not
what was expected.

Whether a well holds more than one template is judged from how well its
reads agree with each other at each insert position, not with the expected
sequence -- a clean well of the wrong construct disagrees with the expected
sequence everywhere it differs, and is not mixed.
"""

from __future__ import annotations

import csv
import difflib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from usortm.demux.expected_plate import ExpectedPlate, ExpectedWell
from usortm.demux.utils import MIXED_TEMPLATE_THRESHOLD, _process_single_well

# Reads a well needs before it is called at all.  Below it the consensus is
# too thin to say more than what the reads look like.
DEFAULT_MIN_READS = 10
# Insert columns need this depth to count towards the agreement check.
MIN_COLUMN_DEPTH = 10
# A consensus insert at least this identical to the expected one is the
# expected construct carrying changes; below it, something else.
CHANGED_IDENTITY = 0.9
# Consensus inserts with more than this share of N are not read at all.
MAX_N_FRACTION = 0.05
# Bases of each flank used to find where the insert starts and stops.
JUNCTION = 20

VERDICTS = ("match", "mixed", "wrong construct", "changed", "unrecognised",
            "too few reads", "no reads", "unexpected reads")
_CLEAN_CALLS = {"Perfect Match", "Silent Mutation"}


@dataclass
class WellVerdict:
    """One well's result.

    Attributes:
        plate: LevSeq plate.
        well: 384-well position.
        label: The well as the expected-plate CSV named it.
        expected: Expected construct, empty for a well expected empty.
        verdict: One of :data:`VERDICTS`.
        observed: Construct the well appears to hold, if one can be named.
        reads: Reads demultiplexed to the well.
        differences: Changes from the expected insert, e.g. ``A271G``.
        protein_changes: The same, as amino-acid changes, when in frame.
        disagreement: Largest share of reads disagreeing with the majority at
            any insert position; high in a well holding two templates.
        note: Anything else worth saying.
    """

    plate: int
    well: str
    label: str
    expected: str
    verdict: str
    observed: str = ""
    reads: int = 0
    differences: list = field(default_factory=list)
    protein_changes: list = field(default_factory=list)
    disagreement: Optional[float] = None
    note: str = ""

    def row(self) -> dict:
        return {
            "plate": self.plate,
            "well": self.well,
            "label": self.label,
            "expected": self.expected,
            "verdict": self.verdict,
            "observed": self.observed,
            "reads": self.reads,
            "differences": " ".join(self.differences),
            "protein_changes": " ".join(self.protein_changes),
            "disagreement": ("" if self.disagreement is None
                             else round(self.disagreement, 3)),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Comparing sequences
# ---------------------------------------------------------------------------

def _equal_ignoring_n(a: str, b: str) -> bool:
    return len(a) == len(b) and all(x == y or x == "N" for x, y in zip(a, b))


def _identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def describe_differences(expected: str, observed: str) -> list:
    """Changes from *expected* to *observed*, 1-based on the expected insert.

    Substitutions read ``A271G``; a run of them ``271-273 AAG>GCT``;
    deletions ``del271-273``; insertions ``ins271^272 ACG``.  N in the
    observed sequence is an unread base, not a change.
    """
    out = []
    sm = difflib.SequenceMatcher(None, expected, observed, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            # Report substitution runs base by base where they are isolated,
            # skipping positions the consensus left unread.
            k = i1
            while k < i2:
                if observed[j1 + k - i1] in ("N", expected[k]):
                    k += 1
                    continue
                s = k
                while (k < i2 and observed[j1 + k - i1] != "N"
                       and observed[j1 + k - i1] != expected[k]):
                    k += 1
                if k - s == 1:
                    out.append(f"{expected[s]}{s + 1}{observed[j1 + s - i1]}")
                else:
                    out.append(f"{s + 1}-{k} {expected[s:k]}>{observed[j1 + s - i1:j1 + k - i1]}")
        elif op == "delete":
            out.append(f"del{i1 + 1}" if i2 - i1 == 1 else f"del{i1 + 1}-{i2}")
        elif op == "insert":
            out.append(f"ins{i1}^{i1 + 1} {observed[j1:j2]}")
        else:
            out.append(f"{i1 + 1}-{i2} {expected[i1:i2]}>{observed[j1:j2]}")
    return out


def describe_protein_changes(expected: str, observed: str) -> list:
    """Amino-acid changes, when both inserts are in frame and one length."""
    if len(expected) != len(observed) or len(expected) % 3:
        return []
    from Bio.Seq import Seq

    out = []
    for i in range(0, len(expected), 3):
        e, o = expected[i:i + 3], observed[i:i + 3]
        if e == o or "N" in o:
            continue
        ea, oa = str(Seq(e).translate()), str(Seq(o).translate())
        if ea != oa:
            out.append(f"{ea}{i // 3 + 1}{oa}")
    return out


def extract_insert(consensus: str, flank_5p: str, flank_3p: str) -> Optional[str]:
    """The consensus between the flanks, found by their inner ends."""
    if not consensus:
        return None
    consensus = consensus.upper()
    left = flank_5p[-JUNCTION:]
    right = flank_3p[:JUNCTION]
    i = consensus.find(left)
    j = consensus.find(right, i + len(left) if i >= 0 else 0)
    if i < 0 or j < 0:
        return None
    # Ambiguity codes and lowercase low-confidence calls are unread bases.
    return "".join(c if c in "ACGT" else "N" for c in consensus[i + len(left):j])


def _column_disagreement(bam_path: str, start: int, end: int) -> Optional[float]:
    """Largest share of reads not carrying the majority base, over [start, end)."""
    import pysam
    from collections import Counter

    if not os.path.exists(bam_path):
        return None
    worst = None
    try:
        # Indexed every time: the BAM is rebuilt on every run, and an index
        # left from an earlier one would be read against the new data.
        pysam.index(bam_path)
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            if not bam.references:
                return None
            ref = bam.references[0]
            for col in bam.pileup(ref, start, end, truncate=True,
                                  min_base_quality=0, stepper="nofilter"):
                bases = Counter()
                for pr in col.pileups:
                    if pr.is_refskip:
                        continue
                    if pr.is_del:
                        bases["-"] += 1
                    else:
                        bases[pr.alignment.query_sequence[pr.query_position]] += 1
                depth = sum(bases.values())
                if depth < MIN_COLUMN_DEPTH:
                    continue
                frac = 1 - bases.most_common(1)[0][1] / depth
                worst = frac if worst is None else max(worst, frac)
    except (OSError, ValueError, pysam.utils.SamtoolsError):
        return None
    return worst


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _observed_wells(demux_dir: Path) -> dict:
    """``{(plate, well): row}`` from the pipeline's per-well table."""
    path = demux_dir / "well_df.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for row in df.to_dict("records"):
        try:
            key = (int(row["plate"]), str(row["well"]))
        except (TypeError, ValueError):
            continue
        out[key] = row
    return out


def _pipeline_call(row: Optional[dict], constructs: set) -> Optional[str]:
    """The construct the pipeline assigned, if it is on the plate and clean."""
    if not row:
        return None
    ref = str(row.get("major_ref") or "")
    if ":" in ref:
        ref = ref.split(":")[-1]
    if ref in constructs and str(row.get("cons_check")) in _CLEAN_CALLS:
        return ref
    return None


def verify_plate(
    plate: ExpectedPlate,
    demux_dir,
    flank_5p: str,
    flank_3p: str,
    tool_paths: dict,
    out_dir,
    min_reads: int = DEFAULT_MIN_READS,
    workers: int = 4,
) -> list:
    """Give every expected well, and every unexpected well with reads, a verdict.

    Args:
        plate: The expectation.
        demux_dir: The pipeline's output directory for this run.
        flank_5p: Vector sequence between the forward barcode and the insert.
        flank_3p: Vector sequence between the insert and the reverse barcode.
        tool_paths: ``minimap2`` and ``samtools`` executables.
        out_dir: Where the per-well checks are written.
        min_reads: Reads a well needs to be called.
        workers: Wells checked at once.

    Returns:
        WellVerdicts, in plate and well order.
    """
    demux_dir = Path(demux_dir)
    out_dir = Path(out_dir)
    ref_dir = out_dir / "references"
    cons_dir = out_dir / "consensus"
    ref_dir.mkdir(parents=True, exist_ok=True)
    cons_dir.mkdir(parents=True, exist_ok=True)

    constructs = plate.constructs()           # sequence -> name
    by_name = {n: s for s, n in constructs.items()}
    expected_at: dict = {}
    for w in plate.wells.values():
        if not w.empty:
            expected_at.setdefault(plate.reference_name(w), []).append(w)
    observed = _observed_wells(demux_dir)
    fastq_dir = demux_dir / "wells" / "fastqs"

    # One full-length reference per construct, as the pipeline builds them.
    for seq, name in constructs.items():
        (ref_dir / f"{name}.fasta").write_text(f">{name}\n{flank_5p}{seq}{flank_3p}\n")

    def _depth(key) -> int:
        row = observed.get(key)
        try:
            return int(row["depth"]) if row else 0
        except (TypeError, ValueError):
            return 0

    def _check(key, name):
        """Consensus of a well's reads against one construct, full length."""
        gw = f"{key[0]}{key[1]}"
        fq = fastq_dir / f"{gw}.fastq"
        if not fq.exists():
            return key, None, None
        paths = {
            "ref_fa": str(ref_dir / f"{name}.fasta"),
            "fq": str(fq),
            "bam": str(cons_dir / f"{gw}.bam"),
            "cons_fa": str(cons_dir / f"{gw}_consensus.fasta"),
            "cons_bam": str(cons_dir / f"{gw}_consensus_align.bam"),
        }
        _, _, cons = _process_single_well(gw, paths, tool_paths["minimap2"],
                                          tool_paths["samtools"])
        start = len(flank_5p)
        disagreement = _column_disagreement(
            paths["bam"], start, start + len(by_name[name]))
        return key, extract_insert(cons or "", flank_5p, flank_3p), disagreement

    # A well expected empty, or not listed, is checked against whatever the
    # pipeline called it when that is a plate construct, and otherwise the
    # first construct -- which for a scan aligns across the insert as well
    # as any other.
    first = next(iter(by_name), None)
    jobs = []
    for w in plate.wells.values():
        if not w.empty and _depth(w.key) >= min_reads:
            jobs.append((w.key, plate.reference_name(w)))
    unlisted = [k for k in observed if k not in plate.wells]
    for key in [w.key for w in plate.wells.values() if w.empty] + unlisted:
        if _depth(key) >= min_reads and first:
            jobs.append((key, _pipeline_call(observed.get(key), set(by_name)) or first))
    checked = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for key, insert, disagreement in pool.map(lambda j: _check(*j), jobs):
            checked[key] = (insert, disagreement)

    def _identify(key):
        """What a well holds: ``(names, insert, readable, disagreement)``.

        *names* are the plate constructs the consensus insert matches, unread
        bases aside -- more than one when an unread base falls exactly where
        they differ.
        """
        insert, disagreement = checked.get(key, (None, None))
        readable = bool(insert) and insert.count("N") <= MAX_N_FRACTION * len(insert)
        names = [n for seq, n in constructs.items()
                 if readable and _equal_ignoring_n(insert, seq)]
        # Reads that do not align across the insert leave only fragments of
        # alignment over it, whose disagreement says nothing about templates.
        if not readable:
            disagreement = None
        return names, insert, readable, disagreement

    def _describe_contents(key) -> tuple:
        """``(observed, note)`` for a well with no expectation to test."""
        names, insert, readable, disagreement = _identify(key)
        call = _pipeline_call(observed.get(key), set(by_name))
        obs = names[0] if len(names) == 1 else (call or "")
        note = ""
        if disagreement is not None and disagreement > MIXED_TEMPLATE_THRESHOLD:
            note = "more than one template"
        elif not obs and readable:
            nearest = max(by_name, key=lambda n: _identity(by_name[n], insert))
            n_changes = len(describe_differences(by_name[nearest], insert))
            note = f"matches no construct; nearest {nearest}, {n_changes} change(s)"
        return obs, note

    verdicts = []
    for w in sorted(plate.wells.values(), key=lambda w: w.key):
        depth = _depth(w.key)
        row = observed.get(w.key)
        name = plate.reference_name(w) or ""
        v = WellVerdict(plate=w.plate, well=w.well, label=w.label,
                        expected=name, verdict="", reads=depth)

        if w.empty:
            if depth < min_reads:
                continue
            v.verdict = "unexpected reads"
            v.observed, note = _describe_contents(w.key)
            v.note = "; ".join(filter(None, ["expected empty", note]))
            v.disagreement = _identify(w.key)[3]
            verdicts.append(v)
            continue
        if depth == 0:
            v.verdict = "no reads"
            verdicts.append(v)
            continue
        if depth < min_reads:
            v.verdict = "too few reads"
            v.observed = _pipeline_call(row, set(by_name)) or ""
            verdicts.append(v)
            continue

        names, insert, readable, disagreement = _identify(w.key)
        v.disagreement = disagreement
        mixed = disagreement is not None and disagreement > MIXED_TEMPLATE_THRESHOLD
        call = _pipeline_call(row, set(by_name))
        others = [n for n in names if n != name]
        if name in names:
            v.observed = name
            v.verdict = "match"
        elif others or (call and call != name and not readable):
            other = others[0] if len(others) == 1 else (call or others[0])
            v.verdict = "wrong construct"
            v.observed = other
            homes = [x.label for x in expected_at.get(other, [])]
            v.note = f"expected at {', '.join(homes)}" if homes else ""
        elif readable and _identity(w.sequence, insert) >= CHANGED_IDENTITY:
            v.verdict = "changed"
            v.observed = f"{name} (changed)"
            v.differences = describe_differences(w.sequence, insert)
            v.protein_changes = describe_protein_changes(w.sequence, insert)
        else:
            v.verdict = "unrecognised"
            if readable:
                nearest = max(by_name, key=lambda n: _identity(by_name[n], insert))
                v.note = (f"nearest construct {nearest}, "
                          f"{_identity(by_name[nearest], insert):.0%} identical")
            else:
                v.note = "insert not read: the reads do not align across it"
        # With a second template the consensus is a vote between two, so the
        # call above describes only the majority, and the note says so.
        # When the consensus left unread exactly the bases where two plate
        # constructs differ, it fits both, and those are the two templates.
        if mixed:
            if len(names) > 1:
                lead = f"consensus fits {' and '.join(names)}"
            else:
                lead = f"majority reads as {v.observed or 'unidentified'}"
            v.note = "; ".join(filter(None, [lead, v.note]))
            v.verdict = "mixed"
        verdicts.append(v)

    # Reads in wells the CSV does not mention at all.
    for key in sorted(set(observed) - set(plate.wells)):
        depth = _depth(key)
        if depth < min_reads:
            continue
        obs, note = _describe_contents(key)
        verdicts.append(WellVerdict(
            plate=key[0], well=key[1], label=key[1], expected="",
            verdict="unexpected reads", reads=depth, observed=obs,
            disagreement=_identify(key)[3],
            note="; ".join(filter(None, ["not in the expected plate", note])),
        ))

    _mark_swaps(verdicts)
    verdicts.sort(key=lambda v: (v.plate, v.well[0], int(v.well[1:])))
    return verdicts


def _mark_swaps(verdicts: list) -> None:
    """Note pairs of wells holding each other's construct."""
    wrong = {(v.expected, v.observed): v for v in verdicts
             if v.verdict == "wrong construct"}
    for (exp, obs), v in wrong.items():
        partner = wrong.get((obs, exp))
        if partner is not None:
            v.note = f"swapped with {partner.label}"


def stray_read_wells(plate: ExpectedPlate, demux_dir, min_reads: int) -> int:
    """Wells expected empty, or not listed, holding a few reads below the call."""
    observed = _observed_wells(Path(demux_dir))
    n = 0
    for key, row in observed.items():
        w = plate.wells.get(key)
        if w is not None and not w.empty:
            continue
        try:
            depth = int(row.get("depth") or 0)
        except (TypeError, ValueError):
            depth = 0
        if 0 < depth < min_reads:
            n += 1
    return n


def write_verification(verdicts: list, path) -> Path:
    """Write the verdicts as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(WellVerdict(0, "", "", "", "").row())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for v in verdicts:
            writer.writerow(v.row())
    return path


def tally(verdicts: list) -> dict:
    """Verdict counts, in :data:`VERDICTS` order."""
    counts = {k: 0 for k in VERDICTS}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return {k: n for k, n in counts.items() if n}
