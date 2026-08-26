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

from usortm.cli.report import NOT_THE_DESIGNED_SEQUENCE
from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH, column_agreement_class)

from .charts import (TIER_READS, bar, colorbar, read_depth_chart,
                     read_length_chart, recovery_chart)
from .plates import demux_plate_maps, pick_plate, pileup_links

#: Parameters the manuscript reports for the hAcyP2 library.  Only the PCR
#: failure rate is still read from here; see measured_parameters().
PUBLISHED = {"skew": 2, "p_incorrect": 0.35, "p_grow": 0.67, "p_fail": 0.025}

#: Wells per sort plate, which sets how many were sorted for a given plate
#: count.  The barcode scheme addresses 16 rows by 24 columns.
WELLS_PER_PLATE = 16 * 24

_CSS_PATH = Path(__file__).with_name("summary.css")


def _designed_variants(project_dir) -> set:
    """The library's members, from the per-variant references demux wrote."""
    ref_dir = os.path.join(str(project_dir), "demux_output",
                           "reference_fasta", "single_ref_fastas")
    names = {os.path.basename(f)[:-6] for f in glob.glob(f"{ref_dir}/*.fasta")}
    names.discard("Parent")
    return names


def _current_pick(project_dir) -> Optional[List[dict]]:
    """The pick for this run, or None when there is none current.

    pick writes its list from one demux's well assignments and a later demux
    leaves that file in place, so age is what separates a pick describing these
    wells from one describing earlier ones.
    """
    pick_json = os.path.join(str(project_dir), "pick", "pick_list.json")
    well_csv = os.path.join(str(project_dir), "demux_output",
                            "well_assignments.csv")
    if not (os.path.exists(pick_json) and os.path.exists(well_csv)):
        return None
    if os.path.getmtime(pick_json) < os.path.getmtime(well_csv):
        return None
    try:
        with open(pick_json) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def estimate_skew(well_data: Sequence[dict],
                  designed: set) -> Optional[dict]:
    """Estimate library skew from how many wells carried each designed variant.

    Read depth measures how much a well was sequenced, not how abundant its
    variant was, so the observable is the number of wells carrying each
    variant.  Those counts are Poisson draws about the library's abundances,
    which is the model :mod:`usortm.qc.skew` fits for read counts from a
    sequenced pool, applied here to well counts.  Sampling noise spreads the
    counts on its own, so the raw ratio of the 90th to the 10th percentile
    overstates skew; :func:`~usortm.qc.skew.measure_skew` deconvolves the
    Poisson component and fits the dropout fraction separately, which keeps
    variants absent from the library out of the skew term.

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
    on_target = [
        w for w in grown
        if w.get("variant") in designed
        and (w.get("consensus_fraction") or 0) > 0.9
        and w.get("cons_check", "") not in NOT_THE_DESIGNED_SEQUENCE
        and (w.get("flank_check", "OK") or "OK") == "OK"
        and column_agreement_class(w.get("max_mismatch_frac")) != "mixed"
    ]
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


def _at(folds, values, x):
    """*values* interpolated at *x*."""
    if x is None or not folds:
        return None
    if x <= folds[0]:
        return values[0]
    for i in range(1, len(folds)):
        if x <= folds[i]:
            f = (x - folds[i - 1]) / (folds[i] - folds[i - 1])
            return values[i - 1] + f * (values[i] - values[i - 1])
    return values[-1]


def _stat(label, value, unit="", extra=""):
    u = f'<span class="u">{unit}</span>' if unit else ""
    return (f'<div><div class="k">{label}</div>'
            f'<div class="v">{value}{u}</div>{extra}</div>')


#: Fold sampling read as a five-step gauge.  Each entry is the depth a step
#: starts at; the simulation behind the recovery curve is what sets them, in
#: that recovery climbs steeply to about 5x and flattens after roughly 8x.
SAMPLING_STEPS = (2.0, 3.0, 5.0, 8.0)


def _sampling_dots(fold: float) -> str:
    """A filled-dot gauge for how deeply the library was sampled.

    Five dots, filled to the step this run reached.  Amber below 3x, where the
    curve predicts a large share of the library is missed however clean the
    sort is, and green at or above it.  The gauge repeats the number beside it
    rather than adding anything to it: it is there to be read without stopping
    to compare against a threshold.
    """
    level = 1 + sum(1 for t in SAMPLING_STEPS if fold >= t)
    tone = "good" if level >= 3 else "warn"
    dots = "".join(f'<i class="on"></i>' if i < level else "<i></i>"
                   for i in range(len(SAMPLING_STEPS) + 1))
    return (f'<div class="dots {tone}" role="img" '
            f'aria-label="Sampling depth {level} of '
            f'{len(SAMPLING_STEPS) + 1}">{dots}</div>')


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
        f'<span class="count"><b id="plateAt">1</b>/{len(plates)}</span>'
        f'<button type="button" data-step="1" aria-label="Next plate">+'
        f'</button></div>'
    )


def render_summary(project: dict, demux_summary: dict,
                   well_data: Sequence[dict], project_dir,
                   tiers: Optional[dict] = None,
                   library_size: Optional[int] = None) -> str:
    """The summary page for one run, as HTML."""
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
        stats.append(_stat("Aligned", f"{aligned:,}",
                           f" {100 * aligned / inp:.1f}%"))
        stats.append(_stat("Demuxed", f"{demuxed:,}",
                           f" {100 * demuxed / inp:.1f}%"))
    stats.append(_stat(f"Wells &ge;{TIER_READS['C']} reads", f"{len(deep):,}"))
    if lib:
        fold = len(deep) / lib
        # No unit: the figure is wells per designed variant, and "of 376"
        # beside it reads as a fraction of the library, which it is not.
        stats.append(_stat("Fold sampling", f"{fold:.1f}&#215;",
                           extra=_sampling_dots(fold)))
    tier_c = (tiers or {}).get("C", {}).get("count")
    if tier_c is not None and lib:
        stats.append(_stat("Library recovered", f"{tier_c}", f" of {lib}"))

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
        missing = lib - (tiers.get("C") or {}).get("count", 0)
        miss_pct = 100 * missing / lib if lib else 0.0
        rows.append(
            f'<tr><td colspan="2" class="name">Not recovered</td>'
            f'<td>{missing} <span class="u">{miss_pct:.1f}%</span></td>'
            f'<td>{bar(miss_pct, "bad")}</td></tr>')
        note = (f'Variants with at least one well at the tier\'s depth whose '
                f'consensus exceeds 90% agreement, carries no error call, has '
                f'intact flanks, and no position where more than '
                f'{MIXED_TEMPLATE_THRESHOLD:.0%} of reads disagree. Tiers are '
                f'cumulative.')
        recovery_html = (
            f'   <div>\n  {_section("Library recovery", note)}\n'
            f'  <table><tr><th>Tier</th><th>Threshold</th><th>Variants</th>'
            f'<th style="width:34%"></th></tr>{"".join(rows)}</table>\n'
            f'   </div>\n')

    # --- what the wells contain ---
    contents_html = ""
    if deep:
        buckets = {"designed": 0, "parent": 0, "uncalled": 0, "other": 0}
        for w in deep:
            variant = w.get("variant") or ""
            cons = w.get("cons_check") or ""
            if variant == "Parent":
                buckets["parent"] += 1
            elif variant == "unassigned" or variant not in designed:
                buckets["uncalled"] += 1
            elif cons in ("Perfect Match", "Match", ""):
                buckets["designed"] += 1
            else:
                # A silent change lands here with the rest: it is not the
                # sequence that was designed, whatever it encodes.
                buckets["other"] += 1
        labels = [("designed", "Variant in library", "good"),
                  ("parent", "Parent (unmutated)", "warn"),
                  ("uncalled", "Insert not readable", "bad"),
                  ("other", "Sequence differs from design", "bad")]
        rows = []
        for key, label, tone in labels:
            n = buckets[key]
            pct = 100 * n / len(deep)
            rows.append(f'<tr><td class="name">{label}</td>'
                        f'<td>{n:,} <span class="u">{pct:.1f}%</span></td>'
                        f'<td>{bar(pct, tone)}</td></tr>')
        note = (f'Over the {len(deep):,} wells with at least '
                f'{TIER_READS["C"]} reads. Each well\'s consensus, translated '
                f'and compared to the parent.')
        contents_html = (
            f'   <div>\n  {_section("What the wells contain", note)}\n'
            f'  <table><tr><th>Outcome</th><th>Wells</th>'
            f'<th style="width:34%"></th></tr>{"".join(rows)}</table>\n'
            f'   </div>\n')

    tables_html = ""
    if recovery_html or contents_html:
        tables_html = (f'  <div class="cols contain">\n{recovery_html}'
                       f'{contents_html}  </div>\n')

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
        info = _simulation_info(project, measured, deep, lib)
        curve_html = recovery_chart(curves, info)
        pred = _at(curves["fold_samplings"], curves["measured"]["means"],
                   measured.get("sampling"))
        hint = ""
        if pred is not None and measured.get("sampling"):
            hint = (f'      <div class="hint">at '
                    f'{measured["sampling"]:.1f}&#215;: {pred:.0f}% predicted')
            if observed is not None:
                hint += f", {observed:.1f}% recovered"
            hint += ".</div>\n"
        note = (f'Variants recovered against fold sampling of the '
                f'{measured["n_grown"]:,} wells that grew, of '
                f'{measured["n_sorted"]:,} sorted on {n_plates} plates.')
        panels.append(
            f'    <div>\n      {_section("Recovery curve", note)}\n'
            f'      {curve_html}\n{hint}    </div>')

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
    pick = pick_plate(_current_pick(project_dir), links, well_class)

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
        demux_head = _section("Demux plate maps", maps["note"],
                              _plate_stepper(maps["plates"]))
        plates_html = (
            f'  <div class="platewrap">\n'
            f'    <div class="phead left">{demux_head}</div>\n'
            f'    <div class="cbcol">{colorbar()}<div class="cblab">reads'
            f'<br>per well</div></div>\n'
            f'    <div class="phead right">{pick_head}</div>\n'
            f'    <div class="pgrid left">{maps["grids"]}</div>\n'
            f'    <div class="pgrid right">{pick["grid"]}</div>\n'
            f'    <div class="pleg left">{maps["legend"]}</div>\n'
            f'    <div class="pleg right">{pick["legend"]}</div>\n'
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

{tables_html}{figures_html}{plates_html}
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
  var box = document.querySelector(".stepper");
  if (!box) return;
  var maps = [...document.querySelectorAll(".plate[data-p]")];
  var at = document.getElementById("plateAt");
  var i = 0;
  function show() {{
    maps.forEach(function (g, k) {{ g.hidden = k !== i; }});
    if (at) at.textContent = i + 1;
    box.querySelectorAll("button").forEach(function (b) {{
      var next = i + Number(b.dataset.step);
      // Disabled at the ends rather than wrapping: wrapping from the last
      // plate to the first reads as a jump to a plate that was not asked for.
      b.disabled = next < 0 || next >= maps.length;
    }});
  }}
  box.addEventListener("click", function (e) {{
    var b = e.target.closest("button[data-step]");
    if (!b || b.disabled) return;
    i = Math.min(maps.length - 1, Math.max(0, i + Number(b.dataset.step)));
    show();
  }});
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


def _simulation_info(project, measured, deep, library_size) -> str:
    """The conditions the curve was computed under, folded into its key.

    Only this run's values.  The published ones were dropped with the curve
    they belonged to, which described a different library at a different skew
    and off-target rate.
    """
    n_deep = len(deep) or 1
    mixed = watch = bad_flank = err = low = 0
    for w in deep:
        klass = column_agreement_class(w.get("max_mismatch_frac"))
        if klass == "mixed":
            mixed += 1
        elif klass == "watch":
            watch += 1
        if (w.get("flank_check") or "OK") != "OK":
            bad_flank += 1
        if w.get("cons_check") in ("Error", "Other Error"):
            err += 1
        if (w.get("consensus_fraction") or 0) <= 0.9:
            low += 1
    skew = float(measured.get("planned_skew") or project.get("skew") or 2)

    def row(label, value):
        return (f'<tr><td class="name">{label}</td><td>{value}</td></tr>')

    est = measured.get("skew_estimate")
    if est:
        skew_row = row(
            "Library skew",
            f"{measured['skew']:.1f} estimated, "
            f"{measured['planned_skew']:g} planned")
    else:
        skew_row = row("Library skew", f"{skew:g} planned")
    model = "".join([
        row("Library size", f"{library_size}"),
        skew_row,
        row("Off-target variation", f"{measured['p_incorrect']:.2f}"),
        row("Sorting efficiency", f"{measured['p_grow']:.2f}"),
        row("PCR failure", f"{measured['p_fail']:.3f}"),
    ])

    def wrow(label, count, share=""):
        return (f'<tr><td class="name">{label}</td><td>{count}</td>'
                f'<td>{share}</td></tr>')

    wells = "".join([
        wrow(f"Sorted, {measured['n_plates']} plates",
             f"{measured['n_sorted']:,}"),
        wrow(f"Grew, &ge;{TIER_READS['C']} reads", f"{measured['n_grown']:,}",
             f"{100 * measured['p_grow']:.0f}%"),
        wrow("On-target", f"{measured['n_on_target']:,}",
             f"{100 * (1 - measured['p_incorrect']):.0f}%"),
        wrow(f"A position past {MIXED_TEMPLATE_THRESHOLD:.0%}", f"{mixed:,}",
             f"{100 * mixed / n_deep:.0f}%"),
        wrow(f"A position at {MIXED_TEMPLATE_WATCH:.0%}&ndash;"
             f"{MIXED_TEMPLATE_THRESHOLD:.0%}", f"{watch:,}",
             f"{100 * watch / n_deep:.0f}%"),
        wrow("Flank failed", f"{bad_flank:,}",
             f"{100 * bad_flank / n_deep:.0f}%"),
        wrow("Called an error", f"{err:,}", f"{100 * err / n_deep:.0f}%"),
        wrow("Agreement &le;90%", f"{low:,}", f"{100 * low / n_deep:.0f}%"),
    ])

    # Where the skew came from and how wide the estimate is.  At a few wells
    # per variant the counts are mostly Poisson, so the interval carries the
    # caveat rather than an adjective.
    skew_note = ""
    if est:
        ci = ""
        if est["ci"]:
            ci = (f', 95% interval {est["ci"][0]:.1f}&ndash;'
                  f'{est["ci"][1]:.1f}')
        dropout = ""
        if est["dropout"] >= 0.005:
            dropout = (f' A further {est["dropout"]:.0%} of the library is '
                       f'estimated absent rather than rare.')
        skew_note = (
            f'<p>Skew is estimated from the number of wells carrying each '
            f'designed variant, {est["mean_wells"]:.1f} on average'
            f'{ci}. Poisson sampling spreads those counts on its own and is '
            f'deconvolved from the estimate.{dropout}</p>')

    return f"""<details class="info keyinfo">
        <summary aria-label="About the simulation"></summary>
        <div class="pop">
          <p>The curve uses this run's own parameters.</p>
          {skew_note}
          <table class="params">
            <tr><th>Parameter</th><th>Value</th></tr>{model}
          </table>
          <table class="params">
            <tr><th>Wells</th><th>Count</th><th>Share</th></tr>{wells}
          </table>
          <p>Sorting efficiency is measured from read counts, so wells lost to
             PCR failure sit inside it. The last five rows are shares of the
             wells that grew and overlap; a well can fail more than one.</p>
        </div>
      </details>"""
