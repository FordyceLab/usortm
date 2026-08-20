"""Tests for reusing a Dorado demux after an interrupted run.

Dorado's output is decided by the reads it was given and the barcode
arrangement it was given them with. Recording both is what lets a resumed run
tell whether the calls on disk are the calls that would be made again -- and
refusing when they are not is the whole safety of it, since another run's
barcode calls carried into this one would misplace every read they touch.
"""

import json
import os

import pytest

from usortm.demux.utils import _demux_fingerprint, _demux_is_reusable


@pytest.fixture
def demuxed(tmp_path):
    """A finished demux: reads, an arrangement, a summary and a sidecar."""
    reads = tmp_path / "oriented.fastq"
    reads.write_text("@a\nACGT\n+\nIIII\n")
    toml = tmp_path / "arrangement.toml"
    toml.write_text("kit = 'levSeq'\n")
    barcodes = tmp_path / "barcodes.fasta"
    barcodes.write_text(">bc01\nACGTACGT\n")

    out = tmp_path / "fbc"
    out.mkdir()
    (out / "sequencing_summary.txt").write_text("read_id\tbarcode\n")
    (out / "demux_inputs.json").write_text(
        json.dumps(_demux_fingerprint(str(reads), str(toml), str(barcodes)))
    )
    return {"out": str(out), "reads": str(reads), "toml": str(toml),
            "barcodes": str(barcodes)}


def _check(d):
    return _demux_is_reusable(d["out"], d["reads"], d["toml"], d["barcodes"])


class TestWhenReuseIsAllowed:

    def test_the_same_reads_and_arrangement(self, demuxed):
        assert _check(demuxed)


class TestWhenItRefuses:

    def test_the_reads_changed(self, demuxed):
        """Different reads mean different barcode calls."""
        with open(demuxed["reads"], "a") as fh:
            fh.write("@b\nTTTT\n+\nIIII\n")
        assert not _check(demuxed)

    def test_the_arrangement_changed(self, demuxed):
        """New masks read barcodes differently -- the case that once gave 19
        classified reads of 35,015."""
        with open(demuxed["toml"], "w") as fh:
            fh.write("kit = 'other'\n")
        assert not _check(demuxed)

    def test_the_barcode_sequences_changed(self, demuxed):
        with open(demuxed["barcodes"], "w") as fh:
            fh.write(">bc01\nTTTTTTTT\n")
        assert not _check(demuxed)

    def test_no_summary_on_disk(self, demuxed):
        os.remove(os.path.join(demuxed["out"], "sequencing_summary.txt"))
        assert not _check(demuxed)

    def test_no_sidecar(self, demuxed):
        """Output from before this was recorded cannot vouch for itself."""
        os.remove(os.path.join(demuxed["out"], "demux_inputs.json"))
        assert not _check(demuxed)

    def test_an_unreadable_sidecar(self, demuxed):
        with open(os.path.join(demuxed["out"], "demux_inputs.json"), "w") as fh:
            fh.write("{not json")
        assert not _check(demuxed)

    def test_a_missing_output_directory(self, tmp_path):
        assert not _demux_is_reusable(str(tmp_path / "absent"), "r", "t", "b")


class TestTheFingerprint:

    def test_it_covers_reads_and_configuration(self, demuxed):
        fp = _demux_fingerprint(demuxed["reads"], demuxed["toml"],
                                demuxed["barcodes"])
        assert set(fp) == {"input", "config"}

    def test_unreadable_configuration_yields_nothing(self, tmp_path):
        """An empty fingerprint never equals a recorded one, so a run whose
        configuration cannot be read re-runs rather than trusting disk."""
        assert _demux_fingerprint(str(tmp_path / "a"), str(tmp_path / "b"),
                                  str(tmp_path / "c")) == {}
