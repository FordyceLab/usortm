"""Reads shorter than --min-read-length never reach a well.

A fragment counts towards a well's depth and, forced to a name, is called
for whichever library member its end happens to fit.  A sequenced pick
plate of a 1,942-base construct came back 72% under 300 bases and named
wells for the last codon that way.  The filter sits at the orientation
stage, after the length histogram (which must still show what was dropped)
and before anything is spent on the read.
"""
import json
import random
import shutil

import pytest

from usortm.demux import utils
from usortm.demux.utils import align_and_split_by_strand


def _tool(name):
    finder = getattr(utils, f"find_{name}", None)
    try:
        return finder() if finder else shutil.which(name)
    except Exception:                                  # noqa: BLE001
        return shutil.which(name)


needs_tools = pytest.mark.skipif(not (_tool("minimap2") and _tool("samtools")),
                                 reason="minimap2 and samtools are needed")


def _reads(tmp_path):
    rng = random.Random(5)
    ref = "".join(rng.choice("ACGT") for _ in range(1942))
    (tmp_path / "ref.fasta").write_text(f">V1\n{ref}\n")
    reads = [("full1", ref), ("full2", ref[10:]), ("frag1", ref[:600]),
             ("frag2", ref[900:1400]), ("frag3", ref[1600:])]
    with open(tmp_path / "reads.fastq", "w") as fh:
        for name, seq in reads:
            fh.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")
    return tmp_path / "ref.fasta", tmp_path / "reads.fastq"


def _names(fastq):
    return {line[1:].split("|")[0] for line in open(fastq) if line.startswith("@")}


@needs_tools
def test_short_reads_are_dropped_and_counted(tmp_path):
    ref, fq = _reads(tmp_path)
    out, ref_map, stats = align_and_split_by_strand(
        str(ref), str(fq), str(tmp_path / "align"), threads=1, min_read_length=1900)
    assert _names(out) == {"full1", "full2"}
    assert set(ref_map) == {"full1", "full2"}
    assert stats["short"] == 3 and stats["mapped"] == 2 and stats["min_read_length"] == 1900
    # the histogram still counts every read, dropped or kept
    assert sum(stats["read_len_hist"]["counts"]) == 5


@needs_tools
def test_without_a_filter_every_aligned_read_is_kept(tmp_path):
    ref, fq = _reads(tmp_path)
    out, _, stats = align_and_split_by_strand(
        str(ref), str(fq), str(tmp_path / "align"), threads=1)
    assert stats["short"] == 0 and stats["min_read_length"] == 0
    assert "frag1" in _names(out)


@needs_tools
def test_a_changed_filter_invalidates_the_cached_orientation(tmp_path):
    """The oriented FASTQ holds only what passed; a resumed run with a
    different filter must not read the old file back as current."""
    ref, fq = _reads(tmp_path)
    out, _, _ = align_and_split_by_strand(
        str(ref), str(fq), str(tmp_path / "align"), threads=1, min_read_length=0)
    assert "frag1" in _names(out)
    out2, _, stats2 = align_and_split_by_strand(
        str(ref), str(fq), str(tmp_path / "align"), threads=1, min_read_length=1900)
    assert _names(out2) == {"full1", "full2"} and stats2["short"] == 3
    # and the same filter again is served from the cache, with its counts
    out3, _, stats3 = align_and_split_by_strand(
        str(ref), str(fq), str(tmp_path / "align"), threads=1, min_read_length=1900)
    assert _names(out3) == {"full1", "full2"} and stats3["short"] == 3
    sidecar = json.load(open(tmp_path / "align" / "align_stats.json")) \
        if (tmp_path / "align" / "align_stats.json").exists() else {}
    assert sidecar.get("min_read_length", 1900) == 1900
