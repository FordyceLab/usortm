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


def _wa_csv(path, rows):
    """A well_assignments.csv as demux writes one."""
    import csv

    cols = ["plate", "well", "reads", "variant", "consensus_fraction",
            "cons_check", "flank_check", "protein_check",
            "assignment_confidence", "n_flagged_positions",
            "max_mismatch_frac"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            base = {c: "" for c in cols}
            base.update({"plate": 1, "consensus_fraction": 0.99,
                         "cons_check": "Perfect Match", "flank_check": "OK",
                         "max_mismatch_frac": 0.01, "reads": 500})
            base.update(r)
            w.writerow(base)


def test_the_parent_is_not_a_hit(tmp_path):
    """Every plate carries it, and it is not a variant of anything.

    Taken as one, a 376-member library merged to 377 hits and reported 100.3%
    coverage.
    """
    from usortm.cli.merge import _build_merged_pick_list, _load_well_assignments

    path = tmp_path / "wa.csv"
    _wa_csv(path, [{"well": "A1", "variant": "V1"},
                   {"well": "A2", "variant": "Parent"},
                   {"well": "A3", "variant": "unassigned"}])
    wells = _load_well_assignments(path)

    picks = _build_merged_pick_list({1: wells}, {"V1": 0}, "C")
    got = {p["variant"] for p in picks if not p.get("empty")}
    assert got == {"V1"}


def test_a_well_the_tiers_discard_is_not_merged_as_recovered(tmp_path):
    """The quality rule has to survive the loader to be a rule at all.

    The loader copied six columns and left the flank and worst-column figures
    behind, so the test downstream did not fail -- it passed, on a missing
    flank reading as intact and a missing disagreement as clean.  Asserted
    through the loader for that reason, not against hand-built dicts.
    """
    from usortm.cli.merge import _load_well_assignments, _passes_tier

    path = tmp_path / "wa.csv"
    _wa_csv(path, [
        {"well": "A1", "variant": "CLEAN"},
        {"well": "A2", "variant": "FLANK", "flank_check": "3' mismatch"},
        {"well": "A3", "variant": "MIXED", "max_mismatch_frac": 0.40},
        {"well": "A4", "variant": "ERR", "cons_check": "Error"},
    ])
    wells = {w["variant"]: w for w in _load_well_assignments(path)}
    designed = {"CLEAN", "FLANK", "MIXED", "ERR"}

    assert _passes_tier(wells["CLEAN"], "C", designed)
    assert not _passes_tier(wells["FLANK"], "C", designed)
    assert not _passes_tier(wells["MIXED"], "C", designed)
    assert not _passes_tier(wells["ERR"], "C", designed)


def test_the_loader_keeps_what_the_rule_reads(tmp_path):
    """Named directly, since dropping them again would be silent."""
    from usortm.cli.merge import _load_well_assignments

    path = tmp_path / "wa.csv"
    _wa_csv(path, [{"well": "A1", "variant": "V1",
                    "flank_check": "5' mismatch",
                    "max_mismatch_frac": 0.33}])
    well = _load_well_assignments(path)[0]
    assert well["flank_check"] == "5' mismatch"
    assert well["max_mismatch_frac"] == 0.33


def test_a_blank_mismatch_column_survives_the_loader(tmp_path):
    """Runs that never measured it write the column empty, not zero."""
    from usortm.cli.merge import _load_well_assignments, _passes_tier

    path = tmp_path / "wa.csv"
    _wa_csv(path, [{"well": "A1", "variant": "V1", "max_mismatch_frac": ""}])
    well = _load_well_assignments(path)[0]
    assert well["max_mismatch_frac"] is None
    # Not measured is not the same as mixed, so the well still counts.
    assert _passes_tier(well, "C", {"V1"})
