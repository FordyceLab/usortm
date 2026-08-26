"""The summary page, and the things it must not claim.

Most of these guard against a section describing a different run: an artefact
left by an earlier demux outlives the one that made it, and reads as current
unless something checks the dates.
"""
import json
import os
import time
from pathlib import Path

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
    # No plate is drawn for it: the counts a plate would carry are absent.
    assert "not recovered, " not in html
    assert "blank by design." not in html


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
    assert m["sorted_sampling"] == 2 * WELLS_PER_PLATE / 2


def test_fold_sampling_counts_the_wells_that_grew(run):
    """The axis is cultures sampled, not wells sorted into.

    A well that never grew was never a sample of the library, and counting it
    would put the run's point further right than its own data supports.
    """
    wells = [_well(1, "A1", "V1"), _well(1, "A2", "V2"),
             _well(1, "A3", "V1", reads=3)]      # below the depth, did not grow
    m = measured_parameters(wells, {"V1", "V2"}, n_plates=1, library_size=2)
    assert m["n_grown"] == 2
    assert m["sampling"] == 1.0          # two grown over a library of two
    assert m["sorted_sampling"] == WELLS_PER_PLATE / 2


def test_growth_is_not_applied_twice(run):
    """With growth in the axis, the curves must not also apply p_grow.

    Taking it in both places puts the run's own point above its own curve,
    which is how the axis and the parameters last disagreed.
    """
    from usortm.report.summary import recovery_curves
    wells = [_well(1, f"A{i}", "V1") for i in range(1, 9)]
    m = measured_parameters(wells, {"V1"}, n_plates=1, library_size=2)
    assert m["p_grow"] < 0.1            # few of a plate's wells grew
    curves = recovery_curves(2, 2.0, m, observed_pct=90.0)
    if not curves:
        pytest.skip("simulation unavailable")
    # A curve that had growth applied twice would sit far below one that did
    # not; at high sampling with no off-target loss it should approach full
    # recovery instead.
    assert max(curves["measured"]["means"]) > 90.0


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


def test_a_watch_well_is_marked_on_the_plate(run):
    """A well between the two thresholds is drawn, not only counted.

    It was reported in the hover and in the parameter table while looking
    identical to a clean well on the map, which is where wells are scanned.
    """
    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1", mismatch=0.15),
         _well(1, "A2", "V2", mismatch=0.02)],
        run, library_size=2)
    assert " watch" in html
    assert "worth checking" in html


def test_a_watch_mark_stacks_with_what_the_well_holds(run):
    """The two corners say different things, so both must be free to appear."""
    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "Parent", mismatch=0.15)], run, library_size=2)
    cells = [c for c in html.split("<") if c.startswith('i class="w parent')]
    assert cells, "a parent well in the watch band should still be drawn"
    assert "watch" in cells[0], "and should carry the watch mark as well"


def test_a_clean_well_carries_no_mark(run):
    html = render_summary(
        {"library_size": 1, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1", mismatch=0.02)], run, library_size=1)
    cells = [c for c in html.split("<") if c.startswith('i class="w"')
             or c.startswith('a class="w"')]
    assert cells, "a clean library well is drawn without a corner"


def test_the_ramp_matches_the_one_the_plate_maps_use():
    """The report interpolates the ramp itself, so it must not drift.

    matplotlib is an optional extra and cannot be required to render a report,
    so charts.cmap_hex reimplements get_custom_cmap's stops.  Two
    implementations of one scale is exactly the arrangement that goes quietly
    wrong, so they are compared here across the range.
    """
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.colors as mcolors

    from usortm.demux.viz import get_custom_cmap

    reference = get_custom_cmap()
    worst = 0
    for i in range(101):
        t = i / 100
        ref = mcolors.rgb2hex(reference(t)[:3])
        mine = cmap_hex(t)
        worst = max(worst, *(abs(int(ref[j:j + 2], 16) - int(mine[j:j + 2], 16))
                             for j in (1, 3, 5)))
    # Both endpoints are exact; between them the two interpolations differ by
    # rounding only, which is well under a step the eye resolves.
    assert cmap_hex(0.0) == mcolors.rgb2hex(reference(0.0)[:3])
    assert cmap_hex(1.0) == mcolors.rgb2hex(reference(1.0)[:3])
    assert worst <= 4, f"ramp drifted by {worst}/255 from get_custom_cmap()"


def test_a_silent_change_is_not_the_designed_sequence(run):
    """Synonymous is still different DNA, and DNA is the deliverable."""
    from usortm.cli.report import _compute_quality_bins
    wells = [{"plate": "1", "well": "A1", "variant": "V1", "reads": 200,
              "consensus_fraction": 0.99, "cons_check": "Silent Mutation",
              "flank_check": "OK", "max_mismatch_frac": 0.02}]
    bins = _compute_quality_bins(wells, 1)
    assert bins["recovery_tiers"]["C"]["count"] == 0


def test_a_silent_change_is_not_its_own_category(run):
    """It is counted with the rest of what differs from the design."""
    html = render_summary(
        {"library_size": 1, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1", cons="Silent Mutation")], run, library_size=1)
    assert "Silent mutation" not in html
    # Counted as a mutation, with every other way a well can hold something
    # other than the sequence designed for it.
    assert ">Mutation</td>" in html


def test_pick_does_not_take_a_silent_change(run):
    import csv

    from usortm.cli.pick import _generate_pick_list
    from usortm.cli.pick import _load_well_assignments as pick_load
    path = run / "wa2.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plate", "well", "reads", "variant",
                    "consensus_fraction", "cons_check", "flank_check",
                    "n_flagged_positions", "max_mismatch_frac"])
        w.writerow([1, "A1", 200, "V1", 1.0, "Silent Mutation", "OK", 0, 0.02])
    picks = _generate_pick_list(pick_load(path), None, True, 384, "row",
                                library_order={"V1": 0})
    assert not [p for p in picks if not p.get("empty")]


def test_the_pick_plate_carries_the_watch_mark(run):
    """A picked well shows the mark its source well has on the demux map.

    Resolved from the wells rather than the pick list: a list written before
    picks carried the fraction has no field to read.
    """
    pick_dir = run / "pick"
    pick_dir.mkdir()
    (pick_dir / "pick_list.json").write_text(json.dumps(
        [{"variant": "V1", "target_well": "A1", "source_plate": "1",
          "source_well": "A1", "reads": 100, "consensus_fraction": 0.99}]))

    html = render_summary(
        {"library_size": 1, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1", mismatch=0.15)], run, library_size=1)
    pick_section = html.split('<div class="pcol">')[-1]
    assert "w watch" in pick_section


def _tiers():
    return {"A": {"count": 2, "pct": 100.0}, "B": {"count": 2, "pct": 100.0},
            "C": {"count": 2, "pct": 100.0}}


def test_a_sections_note_is_out_of_the_flow(run):
    """Two sections stand side by side, and a note between a heading and its
    table is as tall as it happens to wrap: the longer of the two started its
    table lower than the other, and no two rows lined up.
    """
    html = render_summary(
        {"library_size": 2, "round": 1, "skew": 2}, {"input_reads": 100},
        [_well(1, "A1", "V1"), _well(1, "A2", "V2")], run,
        tiers=_tiers(), library_size=2)
    for title in ("Library recovery", "What the wells contain"):
        assert (f'<div class="head"><h2>{title}</h2>'
                f'<details class="info">') in html
    # Nothing stands between a heading and the table it belongs to.
    row = html.split('<div class="cols contain">')[1]
    assert '<p class="note">' not in row.split("</table>")[0]


def test_the_tier_rule_stays_on_the_page(run):
    """The rule is what the tier counts mean.  The button holds it; it does
    not drop it.
    """
    html = render_summary(
        {"library_size": 2, "round": 1, "skew": 2}, {"input_reads": 100},
        [_well(1, "A1", "V1"), _well(1, "A2", "V2")], run,
        tiers=_tiers(), library_size=2)
    pop = html.split("<h2>Library recovery</h2>")[1].split("</details>")[0]
    assert "consensus exceeds 90% agreement" in pop
    assert "25% of reads disagree" in pop
    assert "Tiers are cumulative" in pop


def test_a_note_button_says_what_it_opens(run):
    """A <summary> takes focus and opens from the keyboard, but the control
    itself is one character, so the label carries the section's name.
    """
    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert '<summary aria-label="About demux plate maps">' in html


def test_a_missing_pick_says_so_on_the_page(run):
    """With no plate to draw, the note is the section rather than a gloss on
    it, and saying why nothing is here belongs in the flow.
    """
    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=2)
    assert '<h2>Pick plate</h2><p class="note">Not shown' in html


def test_a_well_links_below_the_page_it_is_on(run):
    """Safari reads a file:// page's directory and no further up.

    The page therefore sits at the top of the run and reaches down to the
    pileups; a link that climbs out with ``../`` loads as "can't open the
    page" for every well.
    """
    pileups = run / "pick" / "pileup"
    pileups.mkdir(parents=True)
    (pileups / "well_1_A1.html").write_text("<html></html>")

    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=2)

    assert 'href="pick/pileup/well_1_A1.html"' in html
    assert 'href="../' not in html


def test_plates_are_stepped_not_tabbed(run):
    """One control for however many plates, rather than one control each."""
    wells = [_well(p, "A1", "V1") for p in range(1, 5)]
    html = render_summary(
        {"library_size": 1, "round": 1}, {"input_reads": 100},
        wells, run, library_size=1)
    assert 'class="stepper"' in html
    assert 'class="tabs"' not in html
    assert "/4</span>" in html          # the count names the total
    assert 'data-step="-1"' in html and 'data-step="1"' in html


def test_a_single_plate_needs_no_stepper(run):
    """With one plate there is nowhere to step to."""
    html = render_summary(
        {"library_size": 1, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1")], run, library_size=1)
    assert 'class="stepper"' not in html


def test_the_information_icon_is_drawn_not_typed():
    """A letter cannot be centred in a 15px circle; a dot and a stem can."""
    css = (Path(__file__).parents[1] / "src" / "usortm" / "report"
           / "summary.css").read_text()
    assert ".info > summary::before" in css
    assert ".info > summary::after" in css
    # One grey for the border and the mark, so it reads as a single object.
    assert "border:1px solid currentColor" in css


def test_the_sampling_gauge_fills_with_depth():
    """One dot at a fold, five well past it, amber until the curve flattens."""
    import re

    from usortm.report.summary import (SAMPLING_STEPS, _dot_row,
                                       _sampling_dots)

    total = len(SAMPLING_STEPS) + 1

    def level(fold):
        # The card draws every band as an example, so the page carries more
        # filled dots than the reading. Take the level the gauge states.
        m = re.search(r"Sampling depth (\d+) of (\d+)", _sampling_dots(fold))
        assert m, "the gauge does not state its level"
        assert int(m.group(2)) == total
        return int(m.group(1))

    assert level(1.0) == 1
    assert level(2.5) == 2
    assert level(3.5) == 3
    assert level(6.0) == 4
    assert level(9.0) == total

    # One row fills exactly what it is asked for, and draws the rest, so the
    # scale's length is readable at any depth.
    assert _dot_row(3, total).count('<i class="on">') == 3
    assert _dot_row(3, total).count("<i") == total
    assert _dot_row(1, total).count("<i") == total

    # Amber where the curve is still climbing steeply, green past 3x.
    assert "dots warn" in _dot_row(1, total)
    assert "dots warn" in _dot_row(2, total)
    assert "dots good" in _dot_row(3, total)


def test_the_sampling_card_draws_every_band():
    """The card shows what each band looks like, with this run's marked."""
    from usortm.report.summary import (SAMPLING_STEPS, _sampling_bands,
                                       _sampling_dots)

    bands = _sampling_bands()
    assert [b[0] for b in bands] == list(range(1, len(SAMPLING_STEPS) + 2))
    assert bands[0][1] == "under 2"
    assert bands[-1][1] == "8 and over"

    html = _sampling_dots(6.0)
    for _, label in bands:
        assert f">{label}</b>" in html
    # Exactly one band is this run's.
    assert html.count('class="now"') == 1
    assert '<b class="now">5 to 8</b>' in html
    # The examples sit in the card, not beside the gauge on the page.
    assert 'class="gaugescale"' in html


def test_the_sampling_gauge_card_is_reachable_and_not_read_twice():
    """The picture is hidden from the label, which carries it in words."""
    from usortm.report.summary import _sampling_dots

    html = _sampling_dots(6.0)
    assert "6.0 wells that grew per designed variant" in html
    # The rule the card draws is spoken for a reader who cannot see it.
    assert "amber below 3" in html
    assert 'aria-label="Sampling depth 4 of 5.' in html
    # Drawn, not left to a title attribute, which never appeared on a glyph
    # this small.  Its own class: .tip is the plate maps' well hover, which
    # is display:none until script adds .on.
    assert 'class="gaugetip"' in html
    assert 'class="tip"' not in html
    assert "title=" not in html
    assert 'aria-hidden="true"' in html
    # Reachable without a pointer.
    assert 'tabindex="0"' in html


def test_the_sampling_gauge_reaches_the_top_metrics(run):
    """The gauge is rendered under the fold-sampling figure, not alongside."""
    html = render_summary(
        {"library_size": 2, "round": 1}, {"input_reads": 100},
        [_well(1, "A1", "V1"), _well(1, "A2", "V2")], run, library_size=2)
    assert "Fold sampling" in html
    assert 'class="dots' in html
    # Inside the metric's own block, after its value.
    i = html.index("Fold sampling")
    assert 'class="dots' in html[i:i + 400]


def test_no_rendered_class_is_hidden_by_another_rule():
    """A class the page renders must not be one another rule sets display:none.

    The sampling gauge's card was first written as ``.tip``, which the plate
    maps' well hover already owned further down the stylesheet: that rule is
    ``display:none`` until script adds ``.on``, and it won on order alone, so
    the card never painted while every other sign of it looked right. The
    failure is silent, so it is worth a test rather than an eye.
    """
    import re

    css = (Path(__file__).parents[1] / "src" / "usortm" / "report"
           / "summary.css").read_text()
    # Comments first: the note explaining this very collision contains the
    # words display:none, and a scan that keeps it blames the rule above.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    hidden = set()
    for m in re.finditer(r"display\s*:\s*none", css):
        # Walk back to the rule's brace, then to the end of the rule before
        # it. A brace-matching regex trips over the @media blocks.
        brace = css.rfind("{", 0, m.start())
        if brace < 0:
            continue
        start = max(css.rfind("}", 0, brace), css.rfind("{", 0, brace)) + 1
        for part in css[start:brace].split(","):
            part = part.strip()
            if re.fullmatch(r"\.[A-Za-z0-9_-]+", part):
                hidden.add(part[1:])

    # The guard is only worth having if it sees the rule that caused the bug.
    assert "tip" in hidden, "scan found no bare display:none class rule"

    rendered = {"gauge", "gaugetip", "dots"}
    clash = rendered & hidden
    assert not clash, f"rendered class hidden by another rule: {clash}"
