"""Tests for reusing a per-well consensus after an interrupted run.

The value is skipping work; the risk is reusing a consensus built from
different reads. This package has had that failure three times in other forms
-- output left in place from an earlier run and taken as current -- so what is
tested here is mostly the refusals.
"""

import os
import time

import pytest

from usortm.demux.utils import _consensus_is_reusable


@pytest.fixture
def well(tmp_path):
    """A well whose outputs are newer than its inputs, as after a good run."""
    paths = {
        "fq": tmp_path / "1A1.fastq",
        "ref_fa": tmp_path / "var.fasta",
        "cons_fa": tmp_path / "1A1_consensus.fasta",
        "cons_bam": tmp_path / "1A1_consensus_align.bam",
    }
    paths["fq"].write_text("@r\nACGT\n+\nIIII\n")
    paths["ref_fa"].write_text(">var\nACGT\n")
    time.sleep(0.01)
    paths["cons_fa"].write_text(">1A1_consensus reads=12\nACGT\n")
    paths["cons_bam"].write_bytes(b"\x1f\x8b\x08\x00fake")
    return {k: str(v) for k, v in paths.items()}


def _touch(path, offset=1):
    """Move a file's mtime *offset* seconds into the future."""
    now = time.time() + offset
    os.utime(path, (now, now))


class TestWhenReuseIsAllowed:

    def test_outputs_newer_than_their_inputs(self, well):
        assert _consensus_is_reusable(well)


class TestWhenItRefuses:

    def test_the_reads_changed_afterwards(self, well):
        """The consensus no longer describes the reads it is filed under."""
        _touch(well["fq"])
        assert not _consensus_is_reusable(well)

    def test_the_reference_changed_afterwards(self, well):
        """A library edit makes every consensus built against it suspect."""
        _touch(well["ref_fa"])
        assert not _consensus_is_reusable(well)

    def test_a_missing_consensus(self, well):
        os.remove(well["cons_fa"])
        assert not _consensus_is_reusable(well)

    def test_a_missing_alignment(self, well):
        os.remove(well["cons_bam"])
        assert not _consensus_is_reusable(well)

    def test_an_empty_consensus(self, well):
        """A run killed mid-write leaves the file created and empty."""
        open(well["cons_fa"], "w").close()
        assert not _consensus_is_reusable(well)

    def test_an_empty_alignment(self, well):
        open(well["cons_bam"], "w").close()
        assert not _consensus_is_reusable(well)

    def test_missing_reads(self, well):
        os.remove(well["fq"])
        assert not _consensus_is_reusable(well)

    def test_the_older_of_the_two_outputs_decides(self, well):
        """Both must post-date the inputs; the newer one cannot vouch for the
        other, since a kill between the two writes leaves exactly that."""
        _touch(well["cons_fa"], offset=10)
        _touch(well["fq"], offset=5)
        assert not _consensus_is_reusable(well)


class TestOffByDefault:

    def test_a_well_is_rebuilt_unless_resume_is_asked_for(self, well,
                                                          monkeypatch):
        """Reuse is opt-in: a clean run must not silently inherit output."""
        from usortm.demux import utils

        called = []
        monkeypatch.setattr(utils, "_consensus_is_reusable",
                            lambda p: called.append(p) or True)
        # resume defaults to False, so the check is never consulted.
        try:
            utils._process_single_well("1A1", well, "minimap2", "samtools")
        except Exception:
            pass
        assert called == []
