"""Every stage the dashboard lists must actually be entered.

A stage is timed by the one that follows it, so a stage the pipeline never
enters is not merely blank -- its time is silently added to whichever stage
came before. On a real run that made "Generating barcode config" read 3m 53s
when it was really config plus the alignment plus both barcode demuxes, and
left three rows with no time at all.

Checked against the source rather than by running a demux, because running one
needs dorado, minimap2, samtools and real reads, and this is a wiring question
that does not need any of them.
"""

import inspect
import re

from usortm.demux import pipeline
from usortm.demux.live import STAGES

DECLARED = {key for key, _ in STAGES}
SOURCE = inspect.getsource(pipeline)
ENTERED = set(re.findall(r'set_stage\(\s*"([a-z_]+)"\s*\)', SOURCE))

#: Set by LiveReport itself when a run begins, so the pipeline never sets it.
SET_ON_CONSTRUCTION = {"deps"}


class TestEveryStageIsEntered:

    def test_no_declared_stage_is_left_unentered(self):
        missing = DECLARED - ENTERED - SET_ON_CONSTRUCTION
        assert not missing, (
            f"declared but never entered, so their time lands on the "
            f"preceding stage: {sorted(missing)}"
        )

    def test_the_stages_that_do_the_work_are_all_there(self):
        """Named explicitly: these four were the ones missing, and they are
        between them most of a run's wall clock."""
        for key in ("align", "fbc", "rbc", "consensus"):
            assert key in ENTERED, f"{key} is not entered by the pipeline"

    def test_the_run_is_marked_finished(self):
        assert "done" in ENTERED


class TestNoStrayStages:

    def test_nothing_is_entered_that_the_page_cannot_show(self):
        """A key the list does not contain leaves the page with no active
        step, which reads as a stalled run."""
        stray = ENTERED - DECLARED
        assert not stray, f"entered but not declared: {sorted(stray)}"


class TestOrder:

    def test_the_declared_order_has_no_duplicates(self):
        keys = [key for key, _ in STAGES]
        assert len(keys) == len(set(keys))

    def test_every_stage_has_a_label(self):
        for key, name in STAGES:
            assert name and name.strip(), f"{key} has no label"
