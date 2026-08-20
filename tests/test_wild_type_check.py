"""Tests for recognising a well that carries the unmutated parent.

A mutational library is built from one sequence and does not contain it, so the
assignment step has to give a parental well some variant it never carried, and
every check against that variant then reports a mismatch. Called an error it
reads as a damaged well. It is an intact one carrying no mutation.
"""

import pandas as pd
import pytest
from Bio.Seq import Seq

from usortm.demux.protein_call import derive_wt_insert
from usortm.demux.utils import _extract_matches_one, extract_matches

FLANK_5P = "AC" * 30          # 60 bp
FLANK_3P = "GT" * 30          # 60 bp
WT_INSERT = ("ATGGCTGAAGGCAACACCCTGATTAGCGTTGACTATGAAATCTTTGGCAAAGTGCAGGGC"
             "GTTTTCTTCCGCAAACACACC")[:81]          # 27 codons
WT_PROTEIN = str(Seq(WT_INSERT).translate())


def _variant(codon, new):
    i = (codon - 1) * 3
    return WT_INSERT[:i] + new + WT_INSERT[i + 3:]


LIBRARY = [_variant(c, n) for c in range(2, 11) for n in ("GCG", "TTT", "ATG",
                                                          "TAG")]


def _row(cons_insert, assigned_insert, cigar=None):
    """One well_df row: a consensus, and the variant it was assigned."""
    cons = FLANK_5P + cons_insert + FLANK_3P
    return pd.Series({
        "global_well": "1A1",
        "ref_len": len(assigned_insert),
        "ref_seq": assigned_insert,
        "cigar": cigar,
        "CIGAR": cigar or f"{len(cons)}M",
        "cons_seq": cons,
    })


def _check(cons_insert, assigned_insert, wt_protein=WT_PROTEIN):
    return _extract_matches_one(
        _row(cons_insert, assigned_insert), len(FLANK_5P), len(FLANK_3P),
        None, 0, False, wt_protein,
    )


class TestDerivingTheParent:

    def test_recovered_from_a_scan_library(self):
        assert derive_wt_insert(LIBRARY) == WT_INSERT

    def test_the_parent_is_not_itself_a_member(self):
        """Which is the whole reason a parental well cannot be assigned."""
        assert WT_INSERT not in LIBRARY


class TestParentalWells:

    def test_a_parental_well_is_named_as_one(self):
        """Assigned some variant, but carrying no mutation at all."""
        out = _check(WT_INSERT, _variant(5, "GCG"))
        assert out["cons_check"] == "Wild Type"
        assert out["protein_check"] == "WT"

    def test_synonymous_differences_still_count_as_parental(self):
        """The library encodes protein changes, so a silent difference from
        the parent's codons is still the parent's protein."""
        syn = WT_INSERT[:3] + "GCC" + WT_INSERT[6:]   # Ala GCT -> GCC
        if str(Seq(syn).translate()) != WT_PROTEIN:
            pytest.skip("chosen codon is not synonymous in this sequence")
        assert _check(syn, _variant(5, "GCG"))["cons_check"] == "Wild Type"

    def test_a_well_carrying_its_assigned_variant_is_untouched(self):
        assigned = _variant(5, "GCG")
        out = _check(assigned, assigned)
        assert out["cons_check"] != "Wild Type"

    def test_a_well_carrying_some_other_variant_is_not_parental(self):
        out = _check(_variant(7, "TAG"), _variant(5, "GCG"))
        assert out["cons_check"] != "Wild Type"

    def test_a_damaged_insert_is_not_parental(self):
        """A truncation is an error and must keep saying so."""
        out = _check(WT_INSERT[:30], _variant(5, "GCG"))
        assert out["cons_check"] != "Wild Type"


class TestWhenItDoesNothing:

    def test_without_a_parent_nothing_is_reclassified(self):
        """A library that is not a scan has no recoverable parent."""
        out = _check(WT_INSERT, _variant(5, "GCG"), wt_protein="")
        assert out["cons_check"] != "Wild Type"

    def test_a_good_match_is_never_overridden(self):
        """Checked explicitly: a well matching its variant must not be
        relabelled even if that variant's protein equals the parent's."""
        assigned = _variant(5, "GCG")
        out = _extract_matches_one(
            _row(assigned, assigned), len(FLANK_5P), len(FLANK_3P), None, 0,
            False, str(Seq(assigned).translate()),
        )
        assert out["cons_check"] != "Wild Type"


class TestThroughExtractMatches:

    def _run(self, rows, **kw):
        kw.setdefault("flank_5p_len", len(FLANK_5P))
        kw.setdefault("flank_3p_len", len(FLANK_3P))
        return extract_matches(pd.DataFrame(rows), workers=1, **kw)

    def test_the_library_drives_it(self):
        out = self._run(
            [_row(WT_INSERT, _variant(5, "GCG")),
             _row(_variant(5, "GCG"), _variant(5, "GCG"))],
            library_inserts=LIBRARY,
        )
        assert out["cons_check"].tolist()[0] == "Wild Type"
        assert out["cons_check"].tolist()[1] != "Wild Type"

    def test_omitting_the_library_leaves_everything_alone(self):
        out = self._run([_row(WT_INSERT, _variant(5, "GCG"))])
        assert out["cons_check"].tolist()[0] != "Wild Type"

    def test_a_non_scan_library_reclassifies_nothing(self):
        """No consensus among the members means no parent to compare to."""
        import numpy as np
        rng = np.random.default_rng(0)
        diverse = ["".join(rng.choice(list("ACGT"), 81)) for _ in range(40)]
        out = self._run([_row(WT_INSERT, _variant(5, "GCG"))],
                        library_inserts=diverse)
        assert out["cons_check"].tolist()[0] != "Wild Type"

    def test_without_flank_lengths_the_insert_cannot_be_located(self):
        """The variable region is found by its distance from each end, so a run
        without --vector-fasta has nothing to translate and nothing to
        compare."""
        out = extract_matches(
            pd.DataFrame([_row(WT_INSERT, _variant(5, "GCG"))]),
            library_inserts=LIBRARY, workers=1,
        )
        assert out["cons_check"].tolist()[0] != "Wild Type"


class TestDownstreamMeaning:
    """'Wild Type' is not in the set pick accepts, so a parental well is still
    not picked for a variant it does not carry -- it is only named correctly."""

    def test_it_is_not_an_acceptable_pick(self):
        from usortm.cli.pick import _generate_pick_list

        wells = [{
            "variant": "V1", "plate": "1", "well": "A1", "reads": 500,
            "consensus_fraction": 0.99, "cons_check": "Wild Type",
        }]
        picked = _generate_pick_list(
            well_data=wells, target_variants=None, unique_only=True,
            target_format=384, fill_order="row", tier=None,
        )
        assert [h for h in picked if not h.get("empty")] == []
