"""Tests for combining per-segment artefacts into the merged view.

Per-well FASTQs and BAMs are linked from each segment into demux_output so
the merged run has one place to look. Leaving an earlier run's file in place
makes the merged view and the segment disagree about the same well: the plate
map counts reads from read_df.csv, which is rewritten every run, while a
pileup reads the per-well FASTQ, which was not.
"""

import os

import pytest

from usortm.cli.demux_cmd import _link_or_copy_tree


def _tree(root, files):
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


class TestStaleArtefactsAreReplaced:

    def test_an_older_file_is_overwritten(self, tmp_path):
        """The regression: a 53-read FASTQ from a subsampled run survived a
        full run that produced 741."""
        src = _tree(tmp_path / "seg" / "wells", {"fastqs/3A9.fastq": "741 reads"})
        dest = _tree(tmp_path / "merged" / "wells", {"fastqs/3A9.fastq": "53 reads"})

        _link_or_copy_tree(src, dest)
        assert (dest / "fastqs" / "3A9.fastq").read_text() == "741 reads"

    def test_the_merged_copy_is_linked_to_the_segment(self, tmp_path):
        src = _tree(tmp_path / "seg" / "wells", {"fastqs/1A1.fastq": "reads"})
        dest = tmp_path / "merged" / "wells"

        _link_or_copy_tree(src, dest)
        a = os.stat(src / "fastqs" / "1A1.fastq")
        b = os.stat(dest / "fastqs" / "1A1.fastq")
        assert (a.st_ino, a.st_dev) == (b.st_ino, b.st_dev), "should be hard-linked"

    def test_nested_paths_are_preserved(self, tmp_path):
        src = _tree(tmp_path / "seg" / "wells", {
            "fastqs/1A1.fastq": "a", "consensus/1A1.bam": "b",
        })
        dest = tmp_path / "merged" / "wells"

        _link_or_copy_tree(src, dest)
        assert (dest / "fastqs" / "1A1.fastq").read_text() == "a"
        assert (dest / "consensus" / "1A1.bam").read_text() == "b"

    def test_two_segments_do_not_clobber_each_other(self, tmp_path):
        """Sort plates are unique to one segment, so both sets must survive."""
        s1 = _tree(tmp_path / "s1" / "wells", {"fastqs/1A1.fastq": "seg1"})
        s2 = _tree(tmp_path / "s2" / "wells", {"fastqs/9A1.fastq": "seg2"})
        dest = tmp_path / "merged" / "wells"

        _link_or_copy_tree(s1, dest)
        _link_or_copy_tree(s2, dest)
        assert (dest / "fastqs" / "1A1.fastq").read_text() == "seg1"
        assert (dest / "fastqs" / "9A1.fastq").read_text() == "seg2"

    def test_a_missing_source_is_a_no_op(self, tmp_path):
        dest = tmp_path / "merged"
        _link_or_copy_tree(tmp_path / "absent", dest)
        assert not dest.exists()
