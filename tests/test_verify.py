"""Tests for checking wells against their expected constructs.

The sequence comparisons are tested directly.  The full check -- reads
through the pipeline and each well tested against its expectation -- is in
test_cli_integration, which needs the demux toolchain.
"""

from usortm.demux.expected_plate import ExpectedPlate, ExpectedWell
from usortm.demux.verify import (
    WellVerdict,
    _mark_swaps,
    describe_differences,
    describe_protein_changes,
    extract_insert,
    tally,
)

F5 = "GGATCCTTAAGCACTCAATGCCGTAGGA"
F3 = "TTGACCATGGTTGAGCTCAAGCTTGCAT"
INSERT = "ATGAAACGTGCAATTGGCTAA"


class TestExtractInsert:

    def test_between_the_flanks(self):
        assert extract_insert(F5 + INSERT + F3, F5, F3) == INSERT

    def test_ambiguity_codes_become_unread(self):
        cons = F5 + INSERT[:4] + "R" + INSERT[5:9] + "n" + INSERT[10:] + F3
        got = extract_insert(cons, F5, F3)
        assert got[4] == "N" and got[9] == "N"

    def test_missing_flank_gives_none(self):
        assert extract_insert(F5 + INSERT, F5, F3) is None
        assert extract_insert("", F5, F3) is None


class TestDifferences:

    def test_substitution(self):
        obs = INSERT[:3] + "T" + INSERT[4:]
        assert describe_differences(INSERT, obs) == ["A4T"]

    def test_codon_substitution_is_one_run(self):
        obs = INSERT[:3] + "GGC" + INSERT[6:]
        assert describe_differences(INSERT, obs) == ["4-6 AAA>GGC"]

    def test_unread_base_is_not_a_change(self):
        obs = INSERT[:3] + "N" + INSERT[4:]
        assert describe_differences(INSERT, obs) == []

    def test_deletion_and_insertion(self):
        assert describe_differences(INSERT, INSERT[:6] + INSERT[9:]) == ["del7-9"]
        assert describe_differences(INSERT, INSERT[:6] + "CCC" + INSERT[6:]) == ["ins6^7 CCC"]

    def test_protein_changes_in_frame(self):
        # AAA (K2) -> GAA (E2)
        obs = INSERT[:3] + "G" + INSERT[4:]
        assert describe_protein_changes(INSERT, obs) == ["K2E"]

    def test_silent_change_has_no_protein_change(self):
        # AAA -> AAG, both K
        obs = INSERT[:5] + "G" + INSERT[6:]
        assert describe_protein_changes(INSERT, obs) == []

    def test_out_of_frame_gives_no_protein_changes(self):
        assert describe_protein_changes(INSERT, INSERT[:6] + INSERT[7:]) == []


class TestSwapsAndTally:

    def _v(self, label, expected, observed, verdict="wrong construct"):
        return WellVerdict(plate=1, well=label, label=label, expected=expected,
                           verdict=verdict, observed=observed)

    def test_pairs_are_marked(self):
        a, b = self._v("A1", "x", "y"), self._v("A2", "y", "x")
        c = self._v("A3", "z", "x")
        _mark_swaps([a, b, c])
        assert a.note == "swapped with A2"
        assert b.note == "swapped with A1"
        assert c.note == ""

    def test_tally_keeps_verdict_order(self):
        vs = [self._v("A1", "x", "x", "match"), self._v("A2", "x", "", "no reads"),
              self._v("A3", "x", "x", "match")]
        assert list(tally(vs).items()) == [("match", 2), ("no reads", 1)]


def test_expected_plate_references_are_named_once():
    """The check writes one reference per construct; wells sharing a sequence
    must resolve to the same one."""
    w1 = ExpectedWell(1, "A1", "wt", INSERT, "A1", 2)
    w2 = ExpectedWell(1, "A2", "wt again", INSERT, "A2", 3)
    plate = ExpectedPlate(wells={w1.key: w1, w2.key: w2}, layout="384", source=None)
    assert plate.reference_name(w1) == plate.reference_name(w2) == "wt"
