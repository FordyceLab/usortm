"""The summary page, and the things it must not claim.

Most of these guard against a section describing a different run: an artefact
left by an earlier demux outlives the one that made it, and reads as current
unless something checks the dates.
"""
import json
import os
import time

import pytest

from usortm.report.charts import cmap_hex, depth_colour
from usortm.report.summary import (WELLS_PER_PLATE, measured_parameters,
                                   render_summary)


def _well(plate, well, variant, reads=200, mismatch=0.02, cons="Perfect Match"):
    return {"plate": str(plate), "well": well, "variant": variant,
            "reads": reads, "consensus_fraction": 0.99, "cons_check": cons,
            "flank_check": "OK", "max_mismatch_frac": mismatch}


@pytest.fixture
def run(tmp_path):
    """A project directory with the files the page reads."""
    demux = tmp_path / "demux_output"
    refs = demux / "reference_fasta" / "single_ref_fastas"
    refs.mkdir(parents=True)
    for name in ("V1", "V2", "Parent"):
        (refs / f"{name}.fasta").write_text(f">{name}\nACGT\n")
    (demux / "well_assignments.csv").write_text("plate,well\n1,A1\n")
    (tmp_path / "report").mkdir()
    return tmp_path


def test_page_renders_without_a_plotting_library(run):
    """The page is inline SVG and HTML; nothing is imported to draw it."""
    html = render_summary(
        {"library_size": 2, "round": 1, "skew": 2},
        {"input_reads": 100, "aligned_reads": 90, "demuxed_reads": 80},
        [_well(1, "A1", "V1"), _well(1, "A2", "V2")],
        run, tiers={"A": {"count": 2, "pct": 100.0},
                    "B": {"count": 2, "pct": 100.0},
                    "C": {"count": 2, "pct": 100.0}},
        library_size=2,
    )
    assert "<svg" in html
    assert "bokeh" not in html.lower()
    assert "uSort-M Summary" in html


def test_a_pick_older_than_the_demux_is_not_drawn(run):
    """A pick built against earlier wells describes a different plate."""
    pick_dir = run / "pick"
    pick_dir.mkdir()
    (pick_dir / "pick_list.json").write_text(json.dumps(
        [{"variant": "V1", "target_well": "A1", "source_plate": "1",
          "source_well": "A1", "reads": 100, "consensus_fraction": 0.99}]))
    # Age it behind the well assignments it would claim to describe.
    old = time.time() - 3600
    os.utime(pick_dir / "pick_list.json", (old, old))

    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert "no current pick" in html
    assert "filled" not in html.split("Pick plate")[1][:400]


def test_a_current_pick_is_drawn(run):
    pick_dir = run / "pick"
    pick_dir.mkdir()
    (pick_dir / "pick_list.json").write_text(json.dumps(
        [{"variant": "V1", "target_well": "A1", "source_plate": "1",
          "source_well": "A1", "reads": 100, "consensus_fraction": 0.99}]))

    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert "1 filled" in html
    assert "no current pick" not in html


def test_read_length_says_what_it_covers(run):
    """A per-segment histogram must not read as the whole run."""
    html = render_summary(
        {"library_size": 2, "round": 1},
        {"input_reads": 1_000_000,
         "read_len_hist": {"counts": [1, 5, 2], "bin_size": 100,
                           "median": 150, "n_reads": 200_000}},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert "one segment of 1,000,000" in html


def test_read_length_is_silent_when_it_covers_the_run(run):
    html = render_summary(
        {"library_size": 2, "round": 1},
        {"input_reads": 200_000,
         "read_len_hist": {"counts": [1, 5, 2], "bin_size": 100,
                           "median": 150, "n_reads": 200_000}},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert "one segment of" not in html


def test_sorted_wells_come_from_the_plate_count(run):
    """Wells sorted is plate capacity, not the planned total or wells seen."""
    wells = [_well(1, "A1", "V1"), _well(2, "A1", "V2")]
    m = measured_parameters(wells, {"V1", "V2"}, n_plates=2, library_size=2)
    assert m["n_sorted"] == 2 * WELLS_PER_PLATE
    assert m["sampling"] == 2 * WELLS_PER_PLATE / 2


def test_off_target_counts_only_clean_library_members(run):
    """A well holding the parent is off target however cleanly it reads."""
    wells = [_well(1, "A1", "V1"), _well(1, "A2", "Parent"),
             _well(1, "A3", "unassigned")]
    m = measured_parameters(wells, {"V1", "V2"}, n_plates=1, library_size=2)
    assert m["n_grown"] == 3
    assert m["n_on_target"] == 1
    assert m["p_incorrect"] == pytest.approx(2 / 3)


def test_a_mixed_well_is_off_target(run):
    wells = [_well(1, "A1", "V1", mismatch=0.5)]
    m = measured_parameters(wells, {"V1"}, n_plates=1, library_size=1)
    assert m["n_on_target"] == 0


def test_depth_colour_runs_light_to_dark():
    """The ramp the plate maps and the depth histogram share."""
    assert depth_colour(0) == "#ffffff"
    assert cmap_hex(0.0) != cmap_hex(1.0)
    # Deeper wells are darker, so the green channel falls.
    assert int(cmap_hex(1.0)[3:5], 16) < int(cmap_hex(0.2)[3:5], 16)
