"""Tests for caches surviving a project being moved.

Keying a cache on the absolute path meant relocating a project -- off a synced
folder, onto a bigger disk -- silently discarded every cached stage and re-ran
hours of work on reads that had not changed. Name, size and modification time
identify the file without pinning it to a directory.
"""

import os

import pytest

from usortm.demux.utils import _fingerprints_match, _input_fingerprint


@pytest.fixture
def reads(tmp_path):
    old = tmp_path / "old" / "fastqs"
    old.mkdir(parents=True)
    path = old / "sample.fastq"
    path.write_text("@r\nACGT\n+\nIIII\n")
    return path


class TestSurvivingAMove:

    def test_the_same_file_in_a_new_directory_matches(self, reads, tmp_path):
        """What a move looks like: same bytes, same timestamp, new parent."""
        before = _input_fingerprint(str(reads))

        moved = tmp_path / "new" / "fastqs"
        moved.mkdir(parents=True)
        dest = moved / reads.name
        os.replace(reads, dest)

        assert _fingerprints_match(before, _input_fingerprint(str(dest)))

    def test_the_directory_is_not_recorded(self, reads):
        fp = _input_fingerprint(str(reads))
        assert set(fp[0]) == {"name", "size", "mtime_ns"}
        assert "old" not in str(fp)


class TestStillRefusingDifferentReads:

    def test_a_changed_file_does_not_match(self, reads):
        before = _input_fingerprint(str(reads))
        reads.write_text("@r\nACGT\n+\nIIII\n@s\nTTTT\n+\nIIII\n")
        assert not _fingerprints_match(before, _input_fingerprint(str(reads)))

    def test_a_different_name_does_not_match(self, reads):
        before = _input_fingerprint(str(reads))
        other = reads.parent / "other.fastq"
        os.replace(reads, other)
        assert not _fingerprints_match(before, _input_fingerprint(str(other)))

    def test_a_different_number_of_files_does_not_match(self, reads):
        before = _input_fingerprint(str(reads))
        (reads.parent / "second.fastq").write_text("@x\nA\n+\nI\n")
        after = _input_fingerprint(str(reads.parent))
        assert not _fingerprints_match(before, after)

    def test_nothing_matches_a_missing_fingerprint(self, reads):
        assert not _fingerprints_match(None, _input_fingerprint(str(reads)))
        assert not _fingerprints_match(_input_fingerprint(str(reads)), None)


class TestOlderSidecars:
    """Sidecars written before the directory was dropped carry ``path``."""

    def test_an_old_entry_matches_by_its_basename(self, reads):
        current = _input_fingerprint(str(reads))
        old_style = [{"path": f"/somewhere/else/{reads.name}",
                      "size": current[0]["size"],
                      "mtime_ns": current[0]["mtime_ns"]}]
        assert _fingerprints_match(old_style, current)

    def test_an_old_entry_for_different_reads_still_refuses(self, reads):
        current = _input_fingerprint(str(reads))
        old_style = [{"path": f"/somewhere/else/{reads.name}",
                      "size": current[0]["size"] + 1,
                      "mtime_ns": current[0]["mtime_ns"]}]
        assert not _fingerprints_match(old_style, current)
