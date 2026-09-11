"""The report counts recovery by the limit the pick was held to.

The tier table, the recovery curve and the plate maps share one predicate,
carries_designed_sequence, which judged the worst column against the fixed
mixed-template threshold whatever the pick had done.  A round 1 picked at
--max-disagreement 0.10 filled 331 wells; the tier table on the same page
said 342 recovered at tier C, and "Library recovered 374 of 376" stood over
a merged plate showing 364.  The report now sets the limit once from the
project's state and every test reads it.
"""
import pytest

from usortm.report import plates, summary
from usortm.report.plates import (carries_designed_sequence,
                                  disagreement_limit_from_project,
                                  set_applied_disagreement_limit)


@pytest.fixture(autouse=True)
def _reset_limit():
    set_applied_disagreement_limit(None)
    yield
    set_applied_disagreement_limit(None)


def _well(mmf, variant="G45A"):
    return {"variant": variant, "reads": 500, "consensus_fraction": 1.0,
            "cons_check": "Perfect Match", "flank_check": "OK",
            "max_mismatch_frac": mmf}


DESIGNED = {"G45A"}


def test_without_a_limit_the_mixed_threshold_decides():
    assert carries_designed_sequence(_well(0.19), DESIGNED)        # watch: counts
    assert not carries_designed_sequence(_well(0.40), DESIGNED)    # mixed: does not


def test_an_explicit_limit_overrides_the_default():
    assert not carries_designed_sequence(_well(0.19), DESIGNED, max_disagreement=0.10)
    assert carries_designed_sequence(_well(0.10), DESIGNED, max_disagreement=0.10)
    assert carries_designed_sequence(_well(None), DESIGNED, max_disagreement=0.10)


def test_the_applied_limit_is_read_when_none_is_passed():
    set_applied_disagreement_limit(0.10)
    assert not carries_designed_sequence(_well(0.19), DESIGNED)
    assert carries_designed_sequence(_well(0.05), DESIGNED)


def test_the_limit_comes_from_the_merge_when_every_round_shares_one():
    project = {"merged": {"max_disagreement": {"1": 0.10, "2": 0.10}},
               "workflow_steps": {"pick": {"max_disagreement": None}}}
    assert disagreement_limit_from_project(project) == 0.10


def test_rounds_that_differ_fall_back_to_round_ones_pick():
    project = {"merged": {"max_disagreement": {"1": 0.10, "2": 0.25}},
               "workflow_steps": {"pick": {"max_disagreement": 0.10}}}
    assert disagreement_limit_from_project(project) == 0.10
    assert disagreement_limit_from_project(project, round_num=2) is None


def test_a_named_round_uses_that_rounds_pick():
    project = {"rounds": {"2": {"workflow_steps": {"pick": {"max_disagreement": 0.05}}}}}
    assert disagreement_limit_from_project(project, round_num=2) == 0.05
    assert disagreement_limit_from_project(project) is None


def test_the_note_states_the_limit_that_was_applied():
    assert "25%" in summary.recovery_note()
    set_applied_disagreement_limit(0.10)
    note = summary.recovery_note()
    assert "10%" in note and "limit the pick was held to" in note
    assert "25%" not in note


def test_the_tier_table_counts_by_the_applied_limit():
    """The table that said 342 over a plate filled to 331."""
    from usortm.cli.report import _compute_quality_bins

    wells = [
        {"plate": "2", "well": "K7", "variant": "G45A", "reads": 516,
         "consensus_fraction": 1.0, "cons_check": "Perfect Match",
         "flank_check": "OK", "max_mismatch_frac": 0.192},          # watch
        {"plate": "1", "well": "B1", "variant": "T26F", "reads": 381,
         "consensus_fraction": 1.0, "cons_check": "Perfect Match",
         "flank_check": "OK", "max_mismatch_frac": 0.05},           # clean
    ]
    tiers = _compute_quality_bins(wells, 2, designed={"G45A", "T26F"})["recovery_tiers"]
    assert tiers["C"]["count"] == 2                                  # 25%: both count

    set_applied_disagreement_limit(0.10)
    tiers = _compute_quality_bins(wells, 2, designed={"G45A", "T26F"})["recovery_tiers"]
    assert tiers["C"]["count"] == 1                                  # 10%: the watch well is out
    assert tiers["A"]["count"] == 1


def test_the_report_command_sets_the_limit_from_the_project():
    """The setting is only worth anything if the command makes it."""
    import inspect

    from usortm.cli import report

    src = inspect.getsource(report)
    assert "set_applied_disagreement_limit(" in src
    assert "disagreement_limit_from_project(" in src
