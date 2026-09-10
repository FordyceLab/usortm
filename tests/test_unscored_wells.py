"""A well with no measurement is not a well that passed.

The column scan returns NaN for a well it could not read, and every
comparison against NaN is false, so such a well fell past both thresholds and
was classified clean.  Round 2 held 22 of them.  One, 1A22 carrying V36M, was
reported clean while 67% of its reads disagreed with the reference -- an
error visible in the 3' flank of its pileup.
"""
import math

import pytest

from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH,
                                column_agreement_class)


def test_a_well_that_could_not_be_scored_is_not_clean():
    assert column_agreement_class(float("nan")) == "unknown"


@pytest.mark.parametrize("value", [None, "", "not a number", float("nan")])
def test_every_absent_measurement_reads_the_same_way(value):
    """None, empty and NaN all mean the same thing and must not diverge."""
    assert column_agreement_class(value) == "unknown"


def test_a_real_measurement_still_classifies():
    assert column_agreement_class(0.0) == "clean"
    assert column_agreement_class(MIXED_TEMPLATE_WATCH) == "clean"
    assert column_agreement_class(MIXED_TEMPLATE_WATCH + 0.01) == "watch"
    assert column_agreement_class(MIXED_TEMPLATE_THRESHOLD) == "watch"
    assert column_agreement_class(MIXED_TEMPLATE_THRESHOLD + 0.01) == "mixed"
    assert column_agreement_class(0.6736) == "mixed"


def test_a_numpy_nan_reads_the_same_as_a_python_one():
    """The value arrives from a DataFrame column, not from a literal."""
    pd = pytest.importorskip("pandas")
    col = pd.Series([0.05, None, 0.6736])
    got = [column_agreement_class(v) for v in col]
    assert got == ["clean", "unknown", "mixed"]
    assert math.isnan(col[1])
