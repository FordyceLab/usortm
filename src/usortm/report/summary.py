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
                                MIXED_TEMPLATE_WATCH, column_agreement_class)

from .charts import (TIER_READS, bar, colorbar_h, read_depth_chart,
                     read_length_chart, recovery_chart)
from .plates import demux_plate_maps, pick_plate, pileup_links

#: Parameters the manuscript reports for the hAcyP2 library, drawn beside this
#: run's own so the two are comparable at a glance.
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
        and w.get("cons_check", "") not in ("Other Error", "Error")
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
    """Simulate recovery on this run's parameters and on the published ones.

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

    # p_grow is 1 on both curves.  The axis counts wells that grew, so the
    # growth loss is already in it; applying it again would take it twice and
    # put the run's own point above its own curve.  What the two curves then
    # compare is the library rather than the sort: given cultures, how much of
    # it comes back.
    d_means, d_stds = run(1.0, PUBLISHED["p_fail"],
                          PUBLISHED["p_incorrect"], PUBLISHED["skew"])
    m_means, m_stds = run(1.0, measured["p_fail"],
                          measured["p_incorrect"], skew)
    if d_means is None or m_means is None:
        return {}
    return {
        "fold_samplings": folds,
        "design": {"means": d_means, "stds": d_stds},
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


def _stat(label, value, unit=""):
    u = f'<span class="u">{unit}</span>' if unit else ""
    return (f'<div><div class="k">{label}</div>'
            f'<div class="v">{value}{u}</div></div>')


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
        recovery_html = (
            f'   <div>\n  <h2>Library recovery</h2>\n'
            f'  <p class="note">Variants with at least one well at the tier\'s '
            f'depth whose consensus exceeds 90% agreement, carries no error '
            f'call, has intact flanks, and whose worst column disagrees by no '
            f'more than {MIXED_TEMPLATE_THRESHOLD:.0%}. Tiers are '
            f'cumulative.</p>\n'
            f'  <table><tr><th>Tier</th><th>Threshold</th><th>Variants</th>'
            f'<th style="width:34%"></th></tr>{"".join(rows)}</table>\n'
            f'   </div>\n')

    # --- what the wells contain ---
    contents_html = ""
    if deep:
        buckets = {"designed": 0, "silent": 0, "parent": 0, "uncalled": 0,
                   "other": 0}
        for w in deep:
            variant = w.get("variant") or ""
            cons = w.get("cons_check") or ""
            if variant == "Parent":
                buckets["parent"] += 1
            elif variant == "unassigned" or variant not in designed:
                buckets["uncalled"] += 1
            elif cons == "Silent Mutation":
                buckets["silent"] += 1
            elif cons in ("Perfect Match", "Match", ""):
                buckets["designed"] += 1
            else:
                buckets["other"] += 1
        labels = [("designed", "Variant in library", "good"),
                  ("silent", "Silent mutation", "good"),
                  ("parent", "Parent (unmutated)", "warn"),
                  ("uncalled", "Insert not readable", "bad"),
                  ("other", "Other mismatch", "bad")]
        rows = []
        for key, label, tone in labels:
            n = buckets[key]
            pct = 100 * n / len(deep)
            rows.append(f'<tr><td class="name">{label}</td>'
                        f'<td>{n:,} <span class="u">{pct:.1f}%</span></td>'
                        f'<td>{bar(pct, tone)}</td></tr>')
        contents_html = (
            f'   <div>\n  <h2>What the wells contain</h2>\n'
            f'  <p class="note">Over the {len(deep):,} wells with at least '
            f'{TIER_READS["C"]} reads. Each well\'s consensus, translated and '
            f'compared to the parent.</p>\n'
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
    curves = recovery_curves(lib, float(project.get("skew") or 2), measured,
                             observed)

    # The two histograms share a column so the row is three wide: a fourth
    # panel wraps onto a second row and leaves the first two thirds empty.
    stacked = []
    hist_html = read_length_chart(demux_summary.get("read_len_hist") or {}, inp)
    if hist_html:
        stacked.append(f'      <div>\n      <h2>Read length</h2>\n'
                       f'      <p class="note">Distribution of read lengths.'
                       f'</p>\n      {hist_html}\n      </div>')
    depth_html = read_depth_chart(depths)
    if depth_html:
        stacked.append(f'      <div>\n      <h2>Read depth per well</h2>\n'
                       f'      <p class="note">Filled by the same scale as the '
                       f'plate maps.</p>\n      {depth_html}\n      </div>')

    panels = []
    if stacked:
        panels.append(f'    <div>\n{"".join(stacked)}\n    </div>')

    curve_html = recovery_chart(curves) if curves else ""
    if curve_html:
        pred = _at(curves["fold_samplings"], curves["measured"]["means"],
                   measured.get("sampling"))
        pub = _at(curves["fold_samplings"], curves["design"]["means"],
                  measured.get("sampling"))
        hint = ""
        if pred is not None and measured.get("sampling"):
            hint = (f'      <div class="hint">at '
                    f'{measured["sampling"]:.1f}&#215;: {pred:.0f}% predicted')
            if observed is not None:
                hint += f", {observed:.1f}% recovered"
            if pub is not None:
                hint += f". Published parameters give {pub:.0f}%"
            hint += ".</div>\n"
        panels.append(
            f'    <div>\n      <h2>Recovery curve</h2>\n'
            f'      <p class="note">Variants recovered against fold sampling '
            f'of the {measured["n_grown"]:,} wells that grew, of '
            f'{measured["n_sorted"]:,} sorted on {n_plates} plates.</p>\n'
            f'      {curve_html}\n{hint}    </div>')
        panels.append(_parameters_panel(project, measured, curves, deep, lib))

    figures_html = ""
    if panels:
        figures_html = f'  <div class="quad">\n{"".join(panels)}\n  </div>\n'

    # --- plates ---
    maps_html = demux_plate_maps(well_data, designed, links)
    pick_html = pick_plate(_current_pick(project_dir), links)

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

{tables_html}{figures_html}  <div class="cols plates">
   <div>
{maps_html}   </div>
   <div>
{pick_html}   </div>
  </div>
  <p class="note">Both plates are filled on the same scale.</p>
  {colorbar_h()}
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
document.querySelectorAll(".tab").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll(".tab").forEach(function (o) {{
      o.classList.toggle("on", o === b); }});
    document.querySelectorAll(".plate[data-p]").forEach(function (g) {{
      g.hidden = g.dataset.p !== b.dataset.p; }});
  }});
}});

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


def _parameters_panel(project, measured, curves, deep, library_size) -> str:
    """The conditions the curves were computed under, beside them.

    Two tables rather than prose: these are values to be compared down a
    column, and a sentence makes the reader pick each one out of it.
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
    skew = float(project.get("skew") or 2)

    def row(label, mine, published=""):
        pub = f"<td>{published}</td>" if published != "" else "<td></td>"
        return (f'<tr><td class="name">{label}</td>'
                f'<td>{mine}</td>{pub}</tr>')

    model = "".join([
        row("Library size", f"{library_size}", "&mdash;"),
        row("Library skew", f"{skew:g}", f"{PUBLISHED['skew']:g}"),
        row("Off-target variation", f"{measured['p_incorrect']:.2f}",
            f"{PUBLISHED['p_incorrect']:.2f}"),
        row("Sorting efficiency", f"{measured['p_grow']:.2f}",
            f"{PUBLISHED['p_grow']:.2f}"),
        row("PCR failure", f"{measured['p_fail']:.3f}",
            f"{PUBLISHED['p_fail']:.3f}"),
    ])

    def wrow(label, count, share=""):
        return (f'<tr><td class="name">{label}</td><td>{count}</td>'
                f'<td>{share}</td></tr>')

    wells = "".join([
        wrow(f"Sorted, {measured['n_plates']} plates",
             f"{measured['n_sorted']:,}"),
        wrow(f"Grew, &ge;{TIER_READS['C']} reads", f"{measured['n_grown']:,}",
             f"{100 * measured['p_grow']:.0f}% of sorted"),
        wrow("On-target", f"{measured['n_on_target']:,}",
             f"{100 * (1 - measured['p_incorrect']):.0f}% of grown"),
        wrow(f"Mixed template &gt;{MIXED_TEMPLATE_THRESHOLD:.0%}", f"{mixed:,}",
             f"{100 * mixed / n_deep:.0f}% of grown"),
        wrow(f"Worth checking {MIXED_TEMPLATE_WATCH:.0%}&ndash;"
             f"{MIXED_TEMPLATE_THRESHOLD:.0%}", f"{watch:,}",
             f"{100 * watch / n_deep:.0f}% of grown"),
        wrow("Flank failed", f"{bad_flank:,}",
             f"{100 * bad_flank / n_deep:.0f}% of grown"),
        wrow("Called an error", f"{err:,}",
             f"{100 * err / n_deep:.0f}% of grown"),
        wrow("Agreement &le;90%", f"{low:,}",
             f"{100 * low / n_deep:.0f}% of grown"),
    ])

    return f"""    <div>
      <h2>Simulation</h2>
      <p class="note">The parameters behind the curves, measured here and as
         published for the hAcyP2 library.</p>
      <table class="params">
        <tr><th>Parameter</th><th>This run</th><th>Published</th></tr>
        {model}
      </table>
      <table class="params">
        <tr><th>Wells</th><th>Count</th><th>Share</th></tr>
        {wells}
      </table>
      <p class="hint">Sorting efficiency is measured from read counts, so wells
         lost to PCR failure sit inside it. The last five rows overlap; a well
         can fail more than one.</p>
    </div>"""
