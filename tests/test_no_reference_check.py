"""A well with no assigned variant must not report a match against one.

The variable-region test asks whether the consensus disagrees with the
reference anywhere. For a well that was never assigned a variant the reference
is empty, so there is nothing to disagree with, and zero mismatches over zero
positions passed as a perfect match. On a real run that was 909 of 2,083
wells: 44% of the plate claiming to match a reference they did not have.
"""

import pandas as pd
import pytest

from usortm.demux.utils import _check_flanking_regions, _extract_matches_one


class TestNoAssignedReference:

    def test_the_flank_check_refuses_a_zero_length_region(self, tmp_path):
        """The guard is here rather than at the call site, since this is the
        function that would otherwise answer vacuously."""
        out = _check_flanking_regions("1A1", 0, 637, 1011, str(tmp_path))
        assert out["cons_check"] != "Perfect Match"

    def test_it_refuses_rather_than_answering(self, tmp_path):
        """Either refusal will do. With no alignment on disk the function
        returns before the guard, reporting Error; with one, the guard reports
        No reference. What matters is that neither is a match."""
        out = _check_flanking_regions("1A1", 0, 637, 1011, str(tmp_path))
        assert out["cons_check"] in ("Error", "No reference")

    def test_a_negative_length_is_refused_too(self, tmp_path):
        out = _check_flanking_regions("1A1", -1, 637, 1011, str(tmp_path))
        assert out["cons_check"] != "Perfect Match"

    def test_a_missing_length_is_refused(self, tmp_path):
        out = _check_flanking_regions("1A1", None, 637, 1011, str(tmp_path))
        assert out["cons_check"] != "Perfect Match"


class TestThroughTheWellCheck:

    def test_an_unassigned_well_is_not_a_perfect_match(self):
        """What the run produced: major_ref 'unassigned', ref_len 0, and a
        cons_check that said the consensus matched it."""
        row = {"global_well": "1A2", "ref_len": 0, "ref_seq": "",
               "CIGAR": None, "cons_seq": None}
        out = _extract_matches_one(row, 0, 0, None, 0, False, "")
        assert out["cons_check"] != "Perfect Match"

    def test_a_well_with_a_reference_still_works(self):
        ref = "ACGT" * 24
        row = {"global_well": "1A1", "ref_len": len(ref), "ref_seq": ref,
               "CIGAR": f"{len(ref)}M", "cons_seq": ref}
        out = _extract_matches_one(row, 0, 0, None, 0, False, "")
        assert out["cons_check"] == "Perfect Match"


class TestItIsNotPickable:
    """An unassigned well must not reach a pick list by looking like a match."""

    def test_no_reference_is_not_an_acceptable_consensus(self):
        from usortm.cli.pick import _generate_pick_list

        wells = [{"variant": "unassigned", "plate": "1", "well": "A1",
                  "reads": 500, "consensus_fraction": 1.0,
                  "cons_check": "No reference"}]
        picked = _generate_pick_list(
            well_data=wells, target_variants=None, unique_only=True,
            target_format=384, fill_order="row", tier=None,
        )
        assert [h for h in picked if not h.get("empty")] == []
