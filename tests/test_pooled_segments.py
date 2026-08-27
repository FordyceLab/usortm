"""FASTQs that cover the same sort plate are read together.

A plate sequenced twice arrives as two FASTQs carrying the same barcode plate
for the same sort plate.  Refusing the run turned a real situation into a dead
end; allowing it as two segments would have been worse, since each writes
per-well files named by sort plate and well and the merged view replaces on
collision -- one FASTQ's reads would have gone missing without a word.
"""
from pathlib import Path

import pytest

from usortm.demux.plate_map import PlateMapError, parse_plate_map


def _map(*entries):
    return {"fastq": [
        {"name": name, "path": f"{name}.fastq", "plates": plates}
        for name, plates in entries
    ]}


def test_two_fastqs_on_one_plate_are_pooled():
    segs = parse_plate_map(_map(("run1", {"1": 16}), ("run2", {"1": 16})),
                           base_dir=Path("."))
    assert len(segs) == 1
    assert [p.name for p in segs[0].all_paths] == ["run1.fastq", "run2.fastq"]
    assert segs[0].plates == {1: 16}


def test_the_pooling_is_reported():
    """The run has fewer segments than the file has entries; that is said."""
    segs = parse_plate_map(_map(("run1", {"1": 16}), ("run2", {"1": 16})),
                           base_dir=Path("."))
    assert segs.notes
    assert "16" in segs.notes[0]
    assert "pooled" in segs.notes[0]


def test_untouched_segments_keep_their_own_reads():
    segs = parse_plate_map(
        _map(("a", {"1": 1}), ("b", {"1": 2})), base_dir=Path("."))
    assert len(segs) == 2
    assert all(not s.extra_paths for s in segs)
    assert not segs.notes


def test_pooling_is_transitive():
    """A third FASTQ sharing a plate with either belongs with both."""
    segs = parse_plate_map(
        _map(("a", {"1": 5}), ("b", {"1": 5, "2": 6}), ("c", {"2": 6})),
        base_dir=Path("."))
    assert len(segs) == 1
    assert len(segs[0].all_paths) == 3
    assert segs[0].plates == {1: 5, 2: 6}


def test_a_segment_sharing_nothing_stays_separate():
    segs = parse_plate_map(
        _map(("a", {"1": 16}), ("b", {"1": 16, "2": 17}), ("c", {"1": 18})),
        base_dir=Path("."))
    assert [s.name for s in segs] == ["a", "c"]
    assert len(segs[0].all_paths) == 2
    assert segs[0].plates == {1: 16, 2: 17}
    assert not segs[1].extra_paths


def test_one_barcode_plate_cannot_mean_two_sort_plates():
    """The FASTQs share plate 16, but disagree about where barcode 1 goes."""
    with pytest.raises(PlateMapError, match="maps to sort plate"):
        parse_plate_map(_map(("a", {"1": 16}), ("b", {"1": 17, "2": 16})),
                        base_dir=Path("."))


def test_two_barcode_plates_cannot_feed_one_sort_plate():
    """Which barcode a read carries would decide its well; nothing can pool."""
    with pytest.raises(PlateMapError, match="reached from barcode plates"):
        parse_plate_map(_map(("a", {"1": 16}), ("b", {"2": 16})),
                        base_dir=Path("."))


def test_order_is_kept():
    """Segment order sets the output directories; pooling must not shuffle."""
    segs = parse_plate_map(
        _map(("first", {"1": 1}), ("second", {"1": 2}), ("third", {"1": 3})),
        base_dir=Path("."))
    assert [s.name for s in segs] == ["first", "second", "third"]


def test_the_pipeline_hands_the_aligner_every_pooled_file(tmp_path, monkeypatch):
    """A pooled segment reaches minimap2 as many files, not one long name.

    parse_plate_map pools two FASTQs covering one sort plate into a segment
    carrying both paths, and the tests above stop there.  Between that and the
    aligner the list passed through a ``str()``, which turned it into a single
    filename -- ``[PosixPath('a.fastq'), PosixPath('b.fastq')]`` -- that
    nothing could open, so every pooled run died at the orientation step with
    a minimap2 exit status and nothing saying why.

    Asserted against the pipeline rather than against the aligner: the aligner
    always resolved a list correctly, and a test calling it with one passes
    just as well on the broken code.
    """
    from usortm.demux import pipeline as pl

    monkeypatch.setattr(
        pl, "check_all_dependencies",
        lambda: {"dorado": "dorado", "minimap2": "minimap2",
                 "samtools": "samtools"})

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pl.utils, "align_and_split_by_strand", _capture)

    reads = []
    for name in ("a", "b"):
        p = tmp_path / f"{name}.fastq"
        p.write_text("@r1\nACGT\n+\nIIII\n")
        reads.append(p)
    ref = tmp_path / "ref.fasta"
    ref.write_text(">r\nACGTACGTACGT\n")
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(_Stop):
        pl.run_levseq_pipeline(
            fastq=reads, output_dir=out, reference=ref,
            n_plates=1, threads=1, workers=1)

    # The list itself, not a string standing in for it.
    assert not isinstance(seen["fastq"], (str, bytes))
    assert [str(p) for p in seen["fastq"]] == [str(p) for p in reads]
