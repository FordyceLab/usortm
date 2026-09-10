"""Merging segment outputs, over a directory a previous merge already filled.

The merge clears the merged directories and then hard-links each segment's
files into them.  Both halves can fail quietly: the clear is
``ignore_errors=True``, and the unlink that follows swallows OSError.  What
is left is a target that is already a hard link to its source, and then
``os.link`` raises FileExistsError and ``shutil.copy2`` raises SameFileError
for the same reason -- a file cannot be copied over itself.

That crashed a merge 100 minutes into a run, and left the merged view holding
a mixture of this run's outputs and an earlier run's.  It costs milliseconds
to test.
"""
import os

import pytest

from usortm.cli.demux_cmd import _link_or_copy_tree


def _segment(tmp_path, name, files):
    src = tmp_path / "segments" / name / "wells" / "consensus"
    src.mkdir(parents=True)
    for fname, text in files.items():
        (src / fname).write_text(text)
    return tmp_path / "segments" / name / "wells"


def test_merging_into_an_empty_directory(tmp_path):
    src = _segment(tmp_path, "segA", {"1A1.bam": "a", "1A1_align.bam": "b"})
    dest = tmp_path / "demux_output" / "wells"
    _link_or_copy_tree(src, dest)
    assert (dest / "consensus" / "1A1.bam").read_text() == "a"
    assert (dest / "consensus" / "1A1_align.bam").read_text() == "b"


def test_merging_twice_is_not_an_error(tmp_path):
    """The second merge finds its own hard links already in place.

    os.link raises FileExistsError, and copy2 then raises SameFileError
    because source and target are the same inode.  Neither is a real problem:
    the file that should be there already is.
    """
    src = _segment(tmp_path, "segA", {"1A1.bam": "a"})
    dest = tmp_path / "demux_output" / "wells"

    _link_or_copy_tree(src, dest)
    _link_or_copy_tree(src, dest)          # must not raise

    assert (dest / "consensus" / "1A1.bam").read_text() == "a"


def test_a_stale_target_is_replaced_by_this_run(tmp_path):
    """A file left by an earlier run must not survive the merge."""
    dest = tmp_path / "demux_output" / "wells" / "consensus"
    dest.mkdir(parents=True)
    (dest / "1A1.bam").write_text("from an earlier run")

    src = _segment(tmp_path, "segA", {"1A1.bam": "from this run"})
    _link_or_copy_tree(src, tmp_path / "demux_output" / "wells")

    assert (dest / "1A1.bam").read_text() == "from this run"


def test_a_target_that_cannot_be_removed_is_reported(tmp_path, caplog):
    """The failure that starts the chain must not be silent.

    A read-only directory stops the unlink; the merge should say so rather
    than carry on and fail later for a reason that does not name this one.
    """
    dest_wells = tmp_path / "demux_output" / "wells"
    dest = dest_wells / "consensus"
    dest.mkdir(parents=True)
    (dest / "1A1.bam").write_text("stale")
    src = _segment(tmp_path, "segA", {"1A1.bam": "fresh"})

    os.chmod(dest, 0o500)                  # no write: unlink will fail
    try:
        _link_or_copy_tree(src, dest_wells)
    finally:
        os.chmod(dest, 0o700)

    # Either it replaced the file or it said why it could not.  Silence with
    # a stale file left behind is the outcome this forbids.
    replaced = (dest / "1A1.bam").read_text() == "fresh"
    complained = any("1A1.bam" in r.getMessage() for r in caplog.records)
    assert replaced or complained


def test_the_right_file_already_being_there_is_not_a_crash(tmp_path):
    """The crash, in the two conditions that produce it.

    A previous merge left the target as a hard link to this very source, and
    the unlink that would clear it fails.  os.link then raises
    FileExistsError and copy2 raises SameFileError, because a file cannot be
    copied over itself.  But the file that should be there already is: there
    is nothing to do and nothing to fail about.

    This is what crashed a merge 100 minutes into a run.
    """
    src = _segment(tmp_path, "segA", {"5N21.bam": "x"})
    dest_wells = tmp_path / "demux_output" / "wells"
    dest = dest_wells / "consensus"
    dest.mkdir(parents=True)
    os.link(src / "consensus" / "5N21.bam", dest / "5N21.bam")
    assert os.path.samefile(src / "consensus" / "5N21.bam", dest / "5N21.bam")

    os.chmod(dest, 0o500)                  # the unlink cannot succeed
    try:
        _link_or_copy_tree(src, dest_wells)     # must not raise
    finally:
        os.chmod(dest, 0o700)

    assert (dest / "5N21.bam").read_text() == "x"


def test_every_segment_file_arrives(tmp_path):
    """Two segments, no overlapping wells: everything lands."""
    a = _segment(tmp_path, "segA", {"1A1.bam": "a", "1A2.bam": "a2"})
    b = _segment(tmp_path, "segB", {"7A1.bam": "b", "7A2.bam": "b2"})
    dest = tmp_path / "demux_output" / "wells"
    _link_or_copy_tree(a, dest)
    _link_or_copy_tree(b, dest)
    got = sorted(p.name for p in (dest / "consensus").iterdir())
    assert got == ["1A1.bam", "1A2.bam", "7A1.bam", "7A2.bam"]
