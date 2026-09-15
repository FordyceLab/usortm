"""Render the run summary page.

Every figure comes from the run's own output.  A section with nothing behind it
is omitted rather than filled in, and a figure that describes a different run
says so instead of being drawn -- an artefact left by an earlier run outlives
the one that made it, and reads as current unless something checks.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                column_agreement_class)

from .charts import (TIER_READS, bar, colorbar, read_depth_chart,
                     read_length_chart, recovery_chart, skew_chart)
from .plates import (carries_designed_sequence, demux_plate_maps,
                     pick_plate, pileup_links)
from .reorder import (failure_rows, outcome_rows, replicate_rows,
                      reorder_plate)

#: Parameters the manuscript reports for the hAcyP2 library.  Only the PCR
#: failure rate is still read from here; see measured_parameters().
PUBLISHED = {"skew": 2, "p_incorrect": 0.35, "p_grow": 0.67, "p_fail": 0.025}

#: Wells per sort plate, which sets how many were sorted for a given plate
#: count.  The barcode scheme addresses 16 rows by 24 columns.
WELLS_PER_PLATE = 16 * 24

_CSS_PATH = Path(__file__).with_name("summary.css")

#: What the re-order section is counting.  Stated once here because the
#: section, the plate tab and the recovery row all describe the same wells.
_REORDER_NOTE = (
    "The dropouts, re-ordered as synthesised constructs and assembled into "
    "known wells. Each was picked as several colonies, so a construct is "
    "recovered when any one of its wells holds it, and the colonies are "
    "counted apart because one failing far more than the others points at "
    "the picking rather than the constructs."
)


def _designed_variants(project_dir) -> set:
    """The library's members, from the per-variant references demux wrote."""
    ref_dir = os.path.join(str(project_dir), "demux_output",
                           "reference_fasta", "single_ref_fastas")
    names = {os.path.basename(f)[:-6] for f in glob.glob(f"{ref_dir}/*.fasta")}
    names.discard("Parent")
    return names


def _current_pick(project_dir) -> Optional[List[dict]]:
    """The pick for this project, or None when there is none current.

    A merged pick is preferred when there is one: it spans the rounds, and
    with a re-order round the single-round pick names only what the sort
    found, leaving the wells the re-order filled empty on the plate.

    pick writes its list from one demux's well assignments and a later demux
    leaves that file in place, so age is what separates a pick describing
    these wells from one describing earlier ones.  A merged pick is measured
    against every round's wells, since it draws on all of them.
    """
    root = str(project_dir)
    rounds = sorted(glob.glob(os.path.join(root, "rounds", "*",
                                           "demux_output",
                                           "well_assignments.csv")))
    own = os.path.join(root, "demux_output", "well_assignments.csv")
    sources = [p for p in [own] + rounds if os.path.exists(p)]
    if not sources:
        return None
    newest = max(os.path.getmtime(p) for p in sources)

    for candidate, against in (
        (os.path.join(root, "merged", "pick_list.json"), newest),
        (os.path.join(root, "pick", "pick_list.json"),
         os.path.getmtime(own) if os.path.exists(own) else 0),
    ):
        if not os.path.exists(candidate):
            continue
        if os.path.getmtime(candidate) < against:
            continue
        try:
            with open(candidate) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            continue
    return None


def estimate_skew(well_data: Sequence[dict],
                  designed: set) -> Optional[dict]:
    """Estimate library skew from how many wells carried each designed variant.

    Read depth measures how much a well was sequenced, not how abundant its
    variant was, so the observable is the number of wells carrying each
    variant.  Those counts are Poisson draws about the library's abundances,
    which is the model :mod:`usortm.qc.skew` fits for read counts from a
    sequenced pool, applied here to well counts.  Sampling noise spreads the
    counts on its own, so the raw ratio overstates skew;
    :func:`~usortm.qc.skew.measure_skew` deconvolves the Poisson component and
    fits the dropout fraction separately, which keeps variants absent from the
    library out of the skew term.

    Skew is Q90/Q10 throughout, the same ratio
    :func:`usortm.simulate.sample.generate_pool` takes, so the estimate can be
    handed straight to the simulation the recovery curve is drawn from.

    A sort yields fewer wells per variant than a sequenced pool yields reads,
    below the depth that function calls sufficient, so the interval is wide.
    At 2.9 wells per variant over 376 variants the median estimate is 1.9 for
    a true skew of 2 and 3.9 for a true 4, and the 95% interval covers truth
    in 12 of 12 seeds (``tests/test_skew.py::test_skew_from_well_counts``).

    Returns None when the fit cannot run, otherwise ``skew``, its ``ci``,
    ``mean_wells`` and ``dropout``.
    """
    if not designed:
        return None
    seen: Dict[str, int] = {v: 0 for v in designed}
    for w in well_data:
        variant = w.get("variant")
        if variant in seen and (w.get("reads") or 0) >= TIER_READS["C"]:
            seen[variant] += 1
    if not any(seen.values()):
        return None
    try:
        from usortm.qc.skew import VariantCounts, measure_skew
        stats = measure_skew(VariantCounts(counts=seen))
    except Exception:
        # scipy missing, or the likelihood did not converge.  The page falls
        # back to the planned skew rather than dropping the figure.
        return None
    lo, hi = stats.q90_q10_ci
    finite = all(v == v and abs(v) != float("inf") for v in (lo, hi))
    return {
        "skew": float(stats.q90_q10_corrected),
        "ci": (float(lo), float(hi)) if finite else None,
        "mean_wells": float(stats.mean_depth),
        "dropout": float(stats.dropout_fraction),
        # What was counted and what was fitted to it, so the figure showing
        # the fit does not have to count the wells a second time.
        "counts": list(seen.values()),
        "stats": stats,
    }


def measured_parameters(well_data: Sequence[dict], designed: set,
                        n_plates: int, library_size: int) -> dict:
    """The simulation's parameters, measured from this run.

    Named as the manuscript names them.  Sorting efficiency is the share of
    sorted wells that returned reads worth calling; off-target variation is the
    share of those whose contents are not a library member cleanly read.  PCR
    failure cannot be separated from sorting efficiency when growth is judged
    from read counts, so the published value is carried rather than re-derived
    and applied twice.
    """
    sorted_wells = n_plates * WELLS_PER_PLATE
    grown = [w for w in well_data if (w.get("reads") or 0) >= TIER_READS["C"]]
    # Fold sampling counts the wells that grew, not the wells that were sorted:
    # a well that never produced a culture was never a sample of the library,
    # and the curve is about what sampling those cultures recovers.  Sorting
    # efficiency is reported beside it rather than folded into the axis.
    on_target = [w for w in grown if carries_designed_sequence(w, designed)]
    n_grown = len(grown) or 1
    return {
        "n_sorted": sorted_wells,
        "n_plates": n_plates,
        "n_grown": len(grown),
        "n_on_target": len(on_target),
        "sampling": len(grown) / library_size if library_size else None,
        "sorted_sampling": sorted_wells / library_size if library_size else None,
        "p_grow": len(grown) / sorted_wells if sorted_wells else 0.0,
        "p_incorrect": 1 - len(on_target) / n_grown,
        "p_fail": PUBLISHED["p_fail"],
    }


def recovery_curves(library_size: int, skew: float, measured: dict,
                    observed_pct: Optional[float]) -> dict:
    """Simulate recovery against sampling depth on this run's parameters.

    Returns an empty dict when the simulation cannot run, so the page omits the
    figure rather than drawing a curve with nothing behind it.
    """
    try:
        import numpy as np

        from usortm.simulate.sortm import sortm
    except Exception:
        return {}

    folds = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    if measured.get("sampling"):
        folds = sorted(set(folds) | {round(measured["sampling"], 1)})

    def run(pg, pf, pi, sk):
        means, stds = [], []
        for fs in folds:
            try:
                r = sortm(n_sims=30, lib_size=library_size, fold_sampling=fs,
                          skew=sk, p_grow=pg, p_fail=pf, p_incorrect=pi,
                          return_correct=True, seed=42)
            except Exception:
                return None, None
            means.append(round(float(np.mean(r) / library_size * 100), 2))
            stds.append(round(float(np.std(r) / library_size * 100), 2))
        return means, stds

    # p_grow is 1 here.  The axis counts wells that grew, so the growth loss
    # is already in it; applying it again would take it twice and put the run's
    # own point above its own curve.  The curve is therefore about the library
    # rather than the sort: given cultures, how much of it comes back.
    m_means, m_stds = run(1.0, measured["p_fail"],
                          measured["p_incorrect"], skew)
    if m_means is None:
        return {}
    return {
        "fold_samplings": folds,
        "measured": {"means": m_means, "stds": m_stds},
        "sampling": measured.get("sampling"),
        "observed": observed_pct,
    }


def recovery_note() -> str:
    """The rule the tier table counts by, stating the limit actually applied.

    The limit is the report's applied disagreement limit -- the one the pick
    and merge were run with -- so the table and the plate beside it are
    described by the same number.  Where none was recorded it is the
    mixed-template threshold, as before.
    """
    from .plates import applied_disagreement_limit

    limit = applied_disagreement_limit()
    if limit is None:
        clause = (f'have no position where more than '
                  f'{MIXED_TEMPLATE_THRESHOLD:.0%} of reads disagree')
    else:
        clause = (f'have no position where more than {limit:.0%} of reads '
                  f'disagree, the limit the pick was held to')
    from .plates import excluded_count

    text = (f'Variants with at least one well at the tier\'s depth whose '
            f'consensus exceeds 90% agreement. That well must also carry no '
            f'error call, have intact flanks, and {clause}. Tiers are '
            f'cumulative.')
    n = excluded_count()
    if n:
        text += (f' {n} well{"s" if n != 1 else ""} the merge was told to '
                 f'leave out (a read-level check found a second clone below '
                 f'that limit) {"are" if n != 1 else "is"} not counted.')
    return text


def _stat(label, value, unit="", extra=""):
    u = f'<span class="u">{unit}</span>' if unit else ""
    return (f'<div><div class="k">{label}</div>'
            f'<div class="v">{value}{u}</div>{extra}</div>')


#: Fold sampling read as a five-step gauge.  Each entry is the depth a step
#: starts at; the simulation behind the recovery curve is what sets them, in
#: that recovery climbs steeply to about 5x and flattens after roughly 8x.
SAMPLING_STEPS = (2.0, 3.0, 5.0, 8.0)


def _dot_row(level: int, n: int) -> str:
    """*level* of *n* dots filled, in the tone that level carries.

    Shared by the gauge and by the examples in its card, so the two cannot
    come to disagree about what a given level looks like.
    """
    tone = "good" if level >= 3 else "warn"
    dots = "".join('<i class="on"></i>' if i < level else "<i></i>"
                   for i in range(n))
    return f'<span class="dots {tone}">{dots}</span>'


def _sampling_bands() -> list:
    """The depth each gauge level covers, as (level, label) pairs."""
    steps = SAMPLING_STEPS
    out = [(1, f"under {steps[0]:g}")]
    out += [(i + 2, f"{steps[i]:g} to {steps[i + 1]:g}")
            for i in range(len(steps) - 1)]
    out.append((len(steps) + 1, f"{steps[-1]:g} and over"))
    return out


def _sampling_dots(fold: float) -> str:
    """A filled-dot gauge for how deeply the library was sampled.

    Five dots, filled to the step this run reached.  The steps follow the
    recovery curve's shape rather than round numbers: recovery climbs steeply
    to about 5x and gains little after 8x, so the gauge separates the depths
    where sorting more plates still pays from the depths where it does not.
    Amber below 3x and green at or above it.

    The card shows every band drawn rather than named, with this run's own
    marked, so the reading is a comparison rather than an arithmetic step.
    What a given depth is predicted to recover depends on the library's skew
    and the sort's off-target rate, so the figure itself is left to the
    recovery curve, which is drawn on this run's own parameters.
    """
    n = len(SAMPLING_STEPS) + 1
    level = 1 + sum(1 for t in SAMPLING_STEPS if fold >= t)
    steps = ", ".join(f"{t:g}" for t in SAMPLING_STEPS)
    # Drawn rather than left to the title attribute, which never appeared:
    # the dots are 7px tall, and a tooltip on a non-interactive element of
    # that size is not reliably offered.
    lead = (f"{fold:.1f} wells that grew per designed variant. The "
            f"recovery curve below gives what this depth is predicted to "
            f"recover.")
    scale = "".join(
        f'{_dot_row(lv, n)}'
        f'<b{" class=\"now\"" if lv == level else ""}>{label}</b>'
        for lv, label in _sampling_bands())
    # The card is a picture, so the words a screen reader gets have to carry
    # the same thing: where the steps fall and which side of them this run is.
    spoken = (f"Sampling depth {level} of {n}. {lead} "
              f"The gauge fills at {steps}, and is amber below "
              f"{SAMPLING_STEPS[1]:g}.")
    return (f'<div class="gauge" tabindex="0" role="img" '
            f'aria-label="{spoken}">'
            f'{_dot_row(level, n)}'
            f'<span class="gaugetip" aria-hidden="true">{lead}'
            f'<span class="gaugescale">{scale}</span></span>'
            f'</div>')


def _qc_mask_note(demux_summary: dict) -> str:
    """What the run's well checks forgave, as a sentence, or nothing.

    Read from the run rather than from the project's configuration, so the
    page describes the checks that produced these counts.
    """
    mask = (demux_summary or {}).get("qc_mask") or []
    if not mask:
        return ""
    changes = ", ".join(f"{m['base']} at {m['position']}"
                        for m in sorted(mask, key=lambda m: m["position"]))
    return (f' {changes} is read as a sequencing artefact and does not '
            f'count against a well, where it is a minority of the reads.'
            if len(mask) == 1 else
            f' {changes} are read as sequencing artefacts and do not count '
            f'against a well, where they are a minority of its reads.')


def _section(title: str, note: str = "", control: str = "") -> str:
    """A section heading, with its explanation folded into a button beside it.

    Sections sit side by side, and a note left in the flow is as tall as it
    happens to wrap: the longer of two notes pushed its table down until the
    two tables' rows no longer lined up.  Out of the flow a note cannot move
    anything, and the page carries less prose for the same explanation.
    """
    if not (note or control):
        return f"<h2>{title}</h2>"
    info = ""
    if note:
        info = (f'<details class="info">'
                f'<summary aria-label="About {title.lower()}"></summary>'
                f'<div class="pop">{note}</div></details>')
    return f'<div class="head"><h2>{title}</h2>{info}{control}</div>'


def _plate_stepper(plates: Sequence[str]) -> str:
    """Step through the plate maps one at a time.

    A button per plate is a wall of them by fourteen, and the maps are read in
    order far more often than jumped between.  The count sits in the heading
    row so the two plate sections keep their titles on one line.
    """
    if len(plates) < 2:
        return ""
    return (
        f'<div class="stepper" data-n="{len(plates)}">'
        f'<button type="button" data-step="-1" aria-label="Previous plate">'
        f'&minus;</button>'
        f'<span class="count"><b id="plateAt">1</b>/'
        f'<b id="plateOf">{len(plates)}</b></span>'
        f'<button type="button" data-step="1" aria-label="Next plate">+'
        f'</button></div>'
    )


def render_summary(project: dict, demux_summary: dict,
                   well_data: Sequence[dict], project_dir,
                   tiers: Optional[dict] = None,
                   library_size: Optional[int] = None,
                   reorder: Optional[dict] = None,
                   pick_check: Optional[dict] = None) -> str:
    """The summary page for one run, as HTML.

    *pick_check* carries the sequenced pick plate when there is one: its
    ``verdicts`` and ``summary`` from :mod:`usortm.verify`, the ``rows`` of
    that round's demux keyed by plate and well, its ``links`` to pileups, and
    the ``layout`` the merge placed, keyed by destination well.

    *reorder* carries a re-order round when one has been sequenced: its
    ``summary`` and ``verdicts`` from :mod:`usortm.verify`, the ``rows`` of its
    own demux keyed by plate and well, and its ``links`` to pileups.  It is
    laid into this page rather than given one of its own, because the
    constructs it buys are the ones this sort missed and the two make one
    recovery figure.
    """
    ro = reorder or None
    # Confirmed here that the sort did not already have: the re-order buys the
    # dropouts, so in practice this is all of them, but a construct recovered
    # by both must not be counted twice in the total.
    rescued = set()
    if ro:
        rescued = set(ro["summary"]["confirmed"]) - {
            w.get("variant") for w in well_data
            if (w.get("reads") or 0) >= TIER_READS["C"]
            and carries_designed_sequence(w, _designed_variants(project_dir))
        }
    lib = library_size or project.get("library_size") or 0
    designed = _designed_variants(project_dir)
    plates = {str(w["plate"]) for w in well_data}
    n_plates = len(plates) or int(project.get("n_plates") or 0) or 1
    links = pileup_links(project_dir)

    deep = [w for w in well_data if (w.get("reads") or 0) >= TIER_READS["C"]]
    depths = [int(w.get("reads") or 0) for w in well_data]

    inp = demux_summary.get("input_reads") or 0
    aligned = demux_summary.get("aligned_reads") or 0
    demuxed = demux_summary.get("demuxed_reads") or 0

    stats = [_stat("Input reads", f"{inp:,}")]
    if inp:
        stats.append(_stat("Aligned to reference", f"{aligned:,}",
                           f" {100 * aligned / inp:.1f}%"))
        stats.append(_stat("Demuxed", f"{demuxed:,}",
                           f" {100 * demuxed / inp:.1f}%"))
    # The well count is not a metric of its own: fold sampling states it as
    # the figure it divides, and the two stood side by side saying it twice.
    if lib:
        fold = len(deep) / lib
        # The wells the figure divides, rather than the library it divides
        # by: "of 376" beside it reads as a fraction of the library, which
        # it is not.
        stats.append(_stat(
            "Fold sampling", f"{fold:.1f}",
            f" {len(deep):,} wells &ge;{TIER_READS['C']} reads",
            _sampling_dots(fold)))
    tier_c = (tiers or {}).get("C", {}).get("count")
    if tier_c is not None and lib:
        # The whole library's figure, not the sort's: a re-order round exists
        # to finish this number, so leaving it at the sort's would report the
        # project as further behind than it is.
        total = tier_c + len(rescued)
        extra = (f'<div class="hint">{tier_c} sorted &middot; '
                 f'{len(rescued)} re-ordered</div>') if rescued else ""
        stats.append(_stat("Library recovered", f"{total}", f" of {lib}",
                           extra))

    # --- library recovery ---
    recovery_html = ""
    if tiers and lib:
        rows = []
        # Which tier a pick was taken at, so the row that decided the plate is
        # distinguishable from the two that only describe it.
        picked_tier = ((project.get("workflow_steps") or {})
                       .get("pick") or {}).get("tier")
        for key, tone in (("A", "good"), ("B", ""), ("C", "warn")):
            t = tiers.get(key) or {}
            pct = t.get("pct", 0.0)
            sel = ' class="sel"' if picked_tier == key else ""
            mark = (' <span class="u">Selected tier</span>'
                    if picked_tier == key else "")
            rows.append(
                f'<tr{sel}><td><span class="chip {key.lower()}">Tier {key}'
                f'</span></td>'
                f'<td class="name">&ge;{TIER_READS[key]} reads{mark}</td>'
                f'<td>{t.get("count", 0)} <span class="u">{pct:.1f}%</span></td>'
                f'<td>{bar(pct, tone)}</td></tr>')
        if rescued:
            pct = 100 * len(rescued) / lib
            rows.append(
                f'<tr><td><span class="chip r">Re-order</span></td>'
                f'<td class="name">assembled and confirmed</td>'
                f'<td>{len(rescued)} <span class="u">{pct:.1f}%</span></td>'
                f'<td>{bar(pct, "good")}</td></tr>')
        missing = lib - (tiers.get("C") or {}).get("count", 0) - len(rescued)
        miss_pct = 100 * missing / lib if lib else 0.0
        rows.append(
            f'<tr><td colspan="2" class="name">Not recovered</td>'
            f'<td>{missing} <span class="u">{miss_pct:.1f}%</span></td>'
            f'<td>{bar(miss_pct, "bad")}</td></tr>')
        note = recovery_note()
        if rescued:
            note += (' The re-order row is the dropouts bought back as '
                     'synthesised constructs, held to the same test in the '
                     'well they were assembled into.')
        recovery_html = (
            f'   <div>\n  {_section("Library recovery", note)}\n'
            f'  <table><tr><th>Tier</th><th>Threshold</th><th>Variants</th>'
            f'<th style="width:34%"></th></tr>{"".join(rows)}</table>\n'
            f'   </div>\n')

    # --- what the wells contain ---
    contents_html = ""
    if deep:
        # The rule the plate maps flag on and the recovery curve is drawn
        # on.  Split four ways this table tested only the consensus call, so
        # a well could be counted here as a library member while the map
        # flagged it and the curve left it out.
        buckets = {"designed": 0, "parent": 0, "mutation": 0}
        for w in deep:
            if (w.get("variant") or "") == "Parent":
                buckets["parent"] += 1
            elif carries_designed_sequence(w, designed):
                buckets["designed"] += 1
            else:
                buckets["mutation"] += 1
        labels = [("designed", "Variant in library", "good"),
                  ("parent", "Parent (unmutated)", "warn"),
                  ("mutation", "Mutation", "bad")]
        rows = []
        for key, label, tone in labels:
            n = buckets[key]
            pct = 100 * n / len(deep)
            rows.append(f'<tr><td class="name">{label}</td>'
                        f'<td>{n:,} <span class="u">{pct:.1f}%</span></td>'
                        f'<td>{bar(pct, tone)}</td></tr>')
        note = (f'Over the {len(deep):,} wells with at least '
                f'{TIER_READS["C"]} reads, by the same test the demux plate '
                f'maps flag on.{_qc_mask_note(demux_summary)}')
        contents_html = (
            f'   <div>\n  {_section("What the wells contain", note)}\n'
            f'  <table><tr><th>Outcome</th><th>Wells</th>'
            f'<th style="width:34%"></th></tr>{"".join(rows)}</table>\n'
            f'   </div>\n')

    tables_html = ""
    if recovery_html or contents_html:
        tables_html = (f'  <div class="cols contain">\n{recovery_html}'
                       f'{contents_html}  </div>\n')

    # --- the re-order round ---
    reorder_html = ""
    if ro:
        rs = ro["summary"]
        n_rep = len(rs["by_replicate"]) or 1
        left = (
            f'   <div>\n  {_section("Re-order round", _REORDER_NOTE)}\n'
            f'  <table><tr><th>Well</th><th>Count</th>'
            f'<th style="width:34%"></th></tr>{outcome_rows(rs)}</table>\n'
            f'  <table><tr><th>Colony</th><th>Held it</th><th>Other</th>'
            f'<th>Empty</th></tr>{replicate_rows(rs)}</table>\n'
            f'   </div>\n')
        fails = failure_rows(ro["verdicts"], ro["rows"])
        if fails:
            note = ('Why a well did not hold the construct ordered for it. '
                    'A well can fail more than one test at once, so each is '
                    'counted under the first that applies.')
            right = (
                f'   <div>\n  {_section("Why wells failed", note)}\n'
                f'  <table><tr><th>Reason</th><th>Wells</th>'
                f'<th style="width:34%"></th></tr>{fails}</table>\n'
                f'   </div>\n')
        else:
            right = (f'   <div>\n  {_section("Why wells failed")}\n'
                     f'  <p class="note">Every well held its construct.</p>\n'
                     f'   </div>\n')
        reorder_html = (f'  <div class="cols contain">\n{left}{right}'
                        f'  </div>\n')

    # --- figures ---
    measured = measured_parameters(well_data, designed, n_plates, lib)
    observed = (tiers or {}).get("C", {}).get("pct")
    # Skew from the run rather than from the plan.  The planned value is what
    # was ordered, not what arrived, and the curve is drawn to describe this
    # run.  The planned value is kept for the table alongside it.
    skew_est = estimate_skew(well_data, designed)
    planned_skew = float(project.get("skew") or 2)
    skew = skew_est["skew"] if skew_est else planned_skew
    measured["skew"] = skew
    measured["planned_skew"] = planned_skew
    measured["skew_estimate"] = skew_est
    curves = recovery_curves(lib, skew, measured, observed)

    # The two histograms share one column, stacked, so the row is two panels
    # wide and the curve beside them can stand as tall as the pair.
    stacked = []
    hist_html = read_length_chart(demux_summary.get("read_len_hist") or {}, inp)
    if hist_html:
        # No note: the chart's own line already says what it covers, and the
        # heading says what it is.
        stacked.append(f'      <div>\n      {_section("Read length")}\n'
                       f'      {hist_html}\n      </div>')
    depth_html = read_depth_chart(depths)
    if depth_html:
        head = _section("Read depth per well",
                        "Filled by the same scale as the plate maps.")
        stacked.append(f'      <div>\n      {head}\n'
                       f'      {depth_html}\n      </div>')

    panels = []
    if stacked:
        panels.append(f'    <div>\n{"".join(stacked)}\n    </div>')

    if curves:
        # The plot marks this run and the key names the mark, so the reading
        # is on the figure already; the line under it repeated in numbers what
        # the red dot says in place, beside a caption saying something else.
        # The numbers keep, in the card the key opens.
        info = _simulation_info(project, measured, lib)
        curve_html = recovery_chart(curves, info)
        note = (f'Variants recovered against fold sampling of the '
                f'{measured["n_grown"]:,} wells that grew, of '
                f'{measured["n_sorted"]:,} sorted on {n_plates} plates.')
        panels.append(
            f'    <div>\n      {_section("Recovery curve", note)}\n'
            f'      {curve_html}\n    </div>')

    # The evidence behind the skew the curve was drawn on, beside the curve
    # rather than inside its card: a fitted width is worth seeing against the
    # counts it was fitted to.
    skew_html = skew_chart(skew_est, lib)
    if skew_html:
        note = ('Bars are the designed variants seen at each number of '
                'wells. The line is the same count under the fitted model, '
                'a log-normal abundance sampled by Poisson. Skew is the '
                'width the fit gives the log-normal, so most of the spread '
                'in the bars is sampling rather than library.')
        panels.append(
            f'    <div>\n      {_section("Wells per variant", note)}\n'
            f'      {skew_html}\n    </div>')

    figures_html = ""
    if panels:
        figures_html = f'  <div class="quad">\n{"".join(panels)}\n  </div>\n'

    # --- plates ---
    maps = demux_plate_maps(well_data, designed, links)
    # How cleanly each well reads, so a pick carries the mark its source well
    # has on the demux map.
    well_class = {
        f'{w["plate"]}_{w["well"]}':
            column_agreement_class(w.get("max_mismatch_frac"))
        for w in well_data
    }
    # A merged pick names its source plate by round, "R1_8" or "R2_1", and
    # both rounds have a plate 1: flattened into one map the rounds' wells
    # would collide and a pick could link to the other round's reads.
    pick_links = dict(links)
    for key, href in links.items():
        pick_links[f"R1_{key}"] = href
    for key, href in ((ro or {}).get("links") or {}).items():
        pick_links[f"R2_{key}"] = href
    pick = pick_plate(_current_pick(project_dir), pick_links, well_class)

    # Headings and the plate tabs sit above the row so the two grids start on
    # the same line: the demux map carries a row of tabs and the pick plate
    # does not, which otherwise drops one grid below the other.  The depth ramp
    # stands to the left of both, once, since they share a scale.
    plates_html = ""
    if maps:
        # With no plate to draw, the note is the section rather than a gloss on
        # it, and saying why nothing is here belongs on the page.
        if pick["grid"]:
            pick_head = _section("Pick plate", pick["note"])
        else:
            pick_head = (f'{_section("Pick plate")}'
                         f'<p class="note">{pick["note"]}</p>')
        # Heading, plate and legend share a column with the plate they belong
        # to, and each row of the grid begins together.  The ramp stands in the
        # middle column: it belongs to both plates, and between them it
        # separates the two without a rule that would say they are measured
        # differently.
        # One plate area with a tab per round rather than two sections: the
        # two plates are read one against the other, and a second grid below
        # put the sort's plate and the re-order's a page apart.
        ro_plate = (reorder_plate(ro["verdicts"], ro["rows"], ro.get("links"))
                    if ro else None)
        stepper = _plate_stepper(maps["plates"])
        if ro_plate and ro_plate["grids"]:
            tabs = ('<div class="tabs plateswitch">'
                    '<button type="button" class="tab on" data-set="sort">'
                    'Sort</button>'
                    '<button type="button" class="tab" data-set="reorder">'
                    'Re-order</button></div>')
            note = (f'<b>Sort.</b> {maps["note"]}</p>'
                    f'<p><b>Re-order.</b> {ro_plate["note"]}')
            demux_head = _section("Plate maps", note, tabs + stepper)
            left_grid = (
                f'<div class="plateset" data-set="sort">{maps["grids"]}</div>'
                f'<div class="plateset" data-set="reorder" hidden>'
                f'{ro_plate["grids"]}</div>')
            left_leg = (
                f'<div class="plateset" data-set="sort">{maps["legend"]}</div>'
                f'<div class="plateset" data-set="reorder" hidden>'
                f'{ro_plate["legend"]}</div>')
        else:
            demux_head = _section("Demux plate maps", maps["note"], stepper)
            left_grid = (f'<div class="plateset" data-set="sort">'
                         f'{maps["grids"]}</div>')
            left_leg = (f'<div class="plateset" data-set="sort">'
                        f'{maps["legend"]}</div>')
        plates_html = (
            f'  <div class="platewrap">\n'
            f'    <div class="phead left">{demux_head}</div>\n'
            f'    <div class="cbcol">{colorbar()}<div class="cblab">reads'
            f'<br>per well</div></div>\n'
            f'    <div class="phead right">{pick_head}</div>\n'
            f'    <div class="pgrid left">{left_grid}</div>\n'
            f'    <div class="pgrid right">{pick["grid"]}</div>\n'
            f'    <div class="pleg left">{left_leg}</div>\n'
            f'    <div class="pleg right">{pick["legend"]}</div>\n'
            f'  </div>\n')

    # --- the pick plate, as sequenced ---------------------------------------
    # The last question the run answers: the plate the robot built, barcoded
    # and sequenced, each well judged against the variant the merge put in it.
    pickcheck_html = ""
    pc = pick_check or None
    if pc and pc.get("verdicts"):
        from .plates import verdict_plate
        from usortm.verify import CONFIRMED, EMPTY, WRONG

        vp = verdict_plate(pc["verdicts"], pc["rows"], pc.get("links"),
                           pc.get("layout"))
        n = sum(vp["counts"].values()) or 1
        rows_html = "".join(
            f'<tr><td class="name">{label}</td>'
            f'<td>{vp["counts"][s]:,} <span class="u">{100 * vp["counts"][s] / n:.1f}%</span></td>'
            f'<td>{bar(100 * vp["counts"][s] / n, tone)}</td></tr>'
            for s, label, tone in ((CONFIRMED, "Holds the variant placed there", "good"),
                                   (WRONG, "Holds something else, or reads unclean", "bad"),
                                   (EMPTY, "Too few reads to call", "warn")))
        failed = [v for v in pc["verdicts"] if v.status != CONFIRMED]
        fail_rows = ""
        if failed:
            from usortm.verify import failure_reason
            lay = pc.get("layout") or {}
            items = []
            for v in sorted(failed, key=lambda v: (v.well[0], int(v.well[1:]))):
                row = pc["rows"].get(f"{v.plate}_{v.well}")
                src = lay.get(v.well, {})
                where = f'{src.get("source_plate", "")} {src.get("source_well", "")}'.strip()
                items.append(
                    f'<tr><td class="name">{v.well}</td><td>{v.expected}</td>'
                    f'<td>{v.observed or "nothing"}</td>'
                    f'<td>{failure_reason(row, v) or ""}</td>'
                    f'<td>{where}</td><td>{v.reads:,}</td></tr>')
            fail_rows = (f'  <table><tr><th>Well</th><th>Placed</th><th>Read as</th>'
                         f'<th>Reason</th><th>From</th><th>Reads</th></tr>'
                         f'{"".join(items)}</table>\n')
        note = ("Each well of the destination plate, sequenced after picking and "
                "judged against the variant the merge placed there by the same "
                "test the pick used. A well that holds its variant is a glycerol "
                "stock; one that does not is re-picked or made by mutagenesis.")
        pickcheck_html = (
            f'  <div class="cols contain">\n'
            f'   <div>\n  {_section("Pick plate, as sequenced", note)}\n'
            f'  <table><tr><th>Well</th><th>Count</th><th style="width:34%"></th></tr>'
            f'{rows_html}</table>\n{fail_rows}   </div>\n'
            f'   <div>\n  {_section("", vp["note"])}\n'
            f'    <div class="pgrid">{vp["grid"]}</div>\n'
            f'    <div class="pleg">{vp["legend"]}</div>\n'
            f'   </div>\n'
            f'  </div>\n')

    versions = demux_summary.get("versions") or {}
    ver_rows = "".join(
        f'<tr><td class="name">{k}</td>'
        f'<td>{(v or {}).get("version") or "&mdash;"}</td></tr>'
        for k, v in sorted(versions.items())
    ) or '<tr><td class="name" colspan="2">Not recorded for this run</td></tr>'

    css = _CSS_PATH.read_text()
    rnd = project.get("round", 1)
    name = os.path.basename(os.path.normpath(str(project_dir)))

    # The insert's measured length, which says whether the construct that came
    # back is the one that was designed.
    lo = demux_summary.get("seq_len_min")
    hi = demux_summary.get("seq_len_max")
    if lo is None or hi is None:
        seq_len = f"{project.get('seq_length', 'N/A')} bp"
    elif lo == hi:
        seq_len = f"{lo} bp"
    else:
        seq_len = f"{lo}–{hi} bp"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>uSort-M summary</title>
<style>{css}</style>
</head><body><main>
  <button id="themeToggle" class="themeToggle" type="button"
          aria-label="Switch between light and dark">&#9681;</button>
  <h1>uSort-M Summary</h1>
  <div class="meta">project: <b>{name}</b> &middot; round {rnd}</div>
  <p class="sub">Library size {lib} designed variants &middot; {n_plates} sort
     plates &middot; insert {seq_len}</p>

  <div class="stats">{"".join(stats)}</div>

{tables_html}{reorder_html}{figures_html}{plates_html}{pickcheck_html}
  <h2>Provenance</h2>
  <table><tr><th>Component</th><th>Version</th></tr>{ver_rows}</table>

  <footer>Rebuild with <code>usortm report {name}/</code>.</footer>
</main>
<script>
var tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
document.addEventListener("mouseover", function (e) {{
  var w = e.target.closest("[data-tip]");
  if (!w) {{ tip.classList.remove("on"); return; }}
  tip.innerHTML = w.dataset.tip;
  tip.classList.add("on");
}});
document.addEventListener("mousemove", function (e) {{
  if (!tip.classList.contains("on")) return;
  var pad = 14, r = tip.getBoundingClientRect();
  var x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > window.innerWidth) x = e.clientX - r.width - pad;
  if (y + r.height > window.innerHeight) y = e.clientY - r.height - pad;
  tip.style.left = (x + window.scrollX) + "px";
  tip.style.top = (y + window.scrollY) + "px";
}});
/* A <details> closes only from the control that opened it, which leaves a note
   standing over the page after the reader has moved on. */
document.addEventListener("click", function (e) {{
  document.querySelectorAll("details.info[open]").forEach(function (d) {{
    if (!d.contains(e.target)) d.open = false;
  }});
}});
document.addEventListener("keydown", function (e) {{
  if (e.key !== "Escape") return;
  document.querySelectorAll("details.info[open]").forEach(function (d) {{
    d.open = false;
  }});
}});

(function () {{
  // The tab picks which round's plates the area shows and the stepper walks
  // whichever that is.  One block rather than two: the stepper's total is a
  // property of the set on show -- the sort has a plate per sort plate, the
  // re-order one per picked colony -- so a stepper that did not know about
  // the tab counted the wrong plates.
  var sets = [...document.querySelectorAll(".plateset")];
  if (!sets.length) return;
  var box = document.querySelector(".stepper");
  var sw = document.querySelector(".plateswitch");
  var at = document.getElementById("plateAt");
  var total = document.getElementById("plateOf");
  var active = sets[0].dataset.set;
  // Where each set was left, so coming back to one returns to the plate that
  // was being read rather than to its first.
  var at_i = {{}};

  function platesOf(name) {{
    var s = sets.find(function (x) {{ return x.dataset.set === name; }});
    return s ? [...s.querySelectorAll(".plate[data-p]")] : [];
  }}

  function show() {{
    var maps = platesOf(active);
    var i = at_i[active] || 0;
    sets.forEach(function (s) {{ s.hidden = s.dataset.set !== active; }});
    maps.forEach(function (g, k) {{ g.hidden = k !== i; }});
    if (!box) return;
    // Nothing to step on a set of one, so the control goes rather than
    // sitting disabled beside it.
    box.hidden = maps.length < 2;
    if (at) at.textContent = i + 1;
    if (total) total.textContent = maps.length;
    box.querySelectorAll("button").forEach(function (b) {{
      var next = i + Number(b.dataset.step);
      // Disabled at the ends rather than wrapping: wrapping from the last
      // plate to the first reads as a jump to one that was not asked for.
      b.disabled = next < 0 || next >= maps.length;
    }});
  }}

  if (box) {{
    box.addEventListener("click", function (e) {{
      var b = e.target.closest("button[data-step]");
      if (!b || b.disabled) return;
      var maps = platesOf(active);
      var i = at_i[active] || 0;
      at_i[active] = Math.min(maps.length - 1,
                              Math.max(0, i + Number(b.dataset.step)));
      show();
    }});
  }}

  if (sw) {{
    sw.addEventListener("click", function (e) {{
      var b = e.target.closest("button[data-set]");
      if (!b) return;
      active = b.dataset.set;
      sw.querySelectorAll("button").forEach(function (x) {{
        x.classList.toggle("on", x === b);
      }});
      show();
    }});
  }}

  show();
}})();

/* The palette follows the system until someone says otherwise, and the choice
   is remembered: a report is looked at more than once, and re-picking it every
   time is worse than not offering it. */
(function () {{
  var KEY = "usortm-theme";
  var root = document.documentElement;
  var saved = null;
  try {{ saved = localStorage.getItem(KEY); }} catch (e) {{}}
  if (saved) root.setAttribute("data-theme", saved);
  var btn = document.getElementById("themeToggle");
  if (!btn) return;
  btn.addEventListener("click", function () {{
    var dark = root.getAttribute("data-theme") === "dark"
      || (!root.getAttribute("data-theme")
          && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {{ localStorage.setItem(KEY, next); }} catch (e) {{}}
  }});
}})();
</script>
</body></html>
"""


def _simulation_info(project, measured, library_size) -> str:
    """The parameters the curve was computed on, folded into its key.

    The table alone.  What the run recovered at its own depth is the marked
    point on the curve, and the counts the skew was fitted to are the figure
    beside it, so restating either here sent the reader to a card for
    something the plot next to it already showed.
    """
    skew = float(measured.get("planned_skew") or project.get("skew") or 2)

    def row(label, value):
        return f'<tr><td class="name">{label}</td><td>{value}</td></tr>'

    est = measured.get("skew_estimate")
    if est:
        skew_row = row(
            "Library skew, Q90/Q10",
            f"{measured['skew']:.1f} estimated, "
            f"{measured['planned_skew']:g} planned")
    else:
        skew_row = row("Library skew, Q90/Q10", f"{skew:g} planned")

    model = "".join([
        row("Library size", f"{library_size}"),
        skew_row,
        row("Off-target variation", f"{measured['p_incorrect']:.2f}"),
        row("Sorting efficiency", f"{measured['p_grow']:.2f}"),
        row("PCR failure", f"{measured['p_fail']:.3f}"),
    ])

    return f"""<details class="info keyinfo">
        <summary aria-label="The curve\'s parameters"></summary>
        <div class="pop">
          <table class="params">
            <tr><th>Parameter</th><th>Value</th></tr>{model}
          </table>
        </div>
      </details>"""
