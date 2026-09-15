"""Plate maps for the summary page.

A well is a cell filled by read depth, with a corner triangle carrying what it
contains -- colour alone cannot say two things about one square.  Where a
pileup has been rendered the cell is a link to it, so a plate map is the way
into the reads rather than only a picture of them.
"""
from __future__ import annotations

import html
import os
import re
from typing import Dict, List, Optional, Sequence

from usortm.cli.report import NOT_THE_DESIGNED_SEQUENCE
from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH, column_agreement_class)

from .charts import TIER_READS, depth_colour


#: The largest per-position disagreement a well may carry and still count as
#: holding its designed sequence, for every test in this report.  None means
#: the mixed-template threshold.  Set once per report from the project's
#: state (see :func:`set_applied_disagreement_limit`), so the tier table, the
#: recovery curve, the plate maps and the pick plate are judged by the rule
#: the pick and merge applied.  Before this the tier table counted a variant
#: recovered at 25% while the plate beside it, picked at 10%, left it empty.
_APPLIED_LIMIT: Optional[float] = None
_UNSET = object()


def set_applied_disagreement_limit(limit: Optional[float]) -> None:
    """Fix the disagreement limit the report's tests are judged by."""
    global _APPLIED_LIMIT
    _APPLIED_LIMIT = limit


#: How many wells the merge was told to leave out (``--exclude-wells``), for
#: the note under the tier table.  The wells themselves are marked on their
#: rows (``excluded``) by the report as it loads them, and every test here
#: treats a marked well as not holding its designed sequence.
_EXCLUDED_COUNT = 0


def set_excluded_count(n: int) -> None:
    global _EXCLUDED_COUNT
    _EXCLUDED_COUNT = int(n or 0)


def excluded_count() -> int:
    return _EXCLUDED_COUNT


def excluded_wells_by_round(project: dict) -> Dict[int, set]:
    """``{round: {(plate, WELL)}}`` from the merge's recorded exclusions."""
    out: Dict[int, set] = {}
    for name in ((project.get("merged") or {}).get("excluded_wells") or []):
        m = re.fullmatch(r"R(\d+)_(\d+)([A-Pa-p]\d{1,2})", str(name).strip())
        if m:
            out.setdefault(int(m.group(1)), set()).add((m.group(2), m.group(3).upper()))
    return out


def stamp_excluded(rows: Sequence[dict], project: dict, round_num: int) -> int:
    """Mark the rows of *round_num* the merge excluded; return how many."""
    wanted = excluded_wells_by_round(project).get(int(round_num), set())
    n = 0
    for w in rows:
        if (str(w.get("plate")), str(w.get("well") or "").upper()) in wanted:
            w["excluded"] = True
            n += 1
    return n


def applied_disagreement_limit() -> Optional[float]:
    return _APPLIED_LIMIT


def disagreement_limit_from_project(project: dict,
                                    round_num: Optional[int] = None
                                    ) -> Optional[float]:
    """The limit the project's picks were held to, read from its state.

    The merge records the limit it applied to each round; a pick records the
    one it applied to its own.  With a round named, that round's pick decides;
    without one, the merge's limits decide where every round shares one, and
    otherwise round 1's pick.  None where nothing was recorded, which is the
    mixed-template threshold.
    """
    def pick_limit(r):
        if r == 1:
            step = (project.get("workflow_steps") or {}).get("pick") or {}
        else:
            step = ((project.get("rounds") or {}).get(str(r)) or {}) \
                .get("workflow_steps", {}).get("pick") or {}
        return step.get("max_disagreement")

    if round_num is not None:
        return pick_limit(round_num)
    merged = (project.get("merged") or {}).get("max_disagreement")
    if isinstance(merged, dict) and merged:
        values = {v for v in merged.values() if v is not None}
        if len(values) == 1:
            return values.pop()
    return pick_limit(1)


def carries_designed_sequence(w: dict, designed: set,
                              max_disagreement=_UNSET) -> bool:
    """Whether a well holds the member it was assigned, read cleanly.

    One rule for the plate maps' flag and for the parameters the recovery
    curve is drawn on.  A well flagged on the map and counted as on-target in
    the same report is a contradiction a reader has no way to settle, and the
    two tests were written out twice before.  Mirrors the tier test in
    :func:`usortm.cli.report._compute_quality_bins`.

    A well missing flank or agreement data is not failed for missing it: the
    fields arrive from stages that a run may not have reached.

    *max_disagreement* is the largest per-position disagreement allowed.
    Left unset it is the report's applied limit (:data:`_APPLIED_LIMIT`);
    None is the mixed-template threshold.
    """
    if w.get("variant") not in designed:
        return False
    if w.get("excluded"):
        # Left out of the pick by name (merge --exclude-wells): a check made
        # outside the pipeline found the well wanting, and the report must
        # not count recovered what the plate does not hold.
        return False
    if (w.get("consensus_fraction") or 0) <= 0.9:
        return False
    if w.get("cons_check", "") in NOT_THE_DESIGNED_SEQUENCE:
        return False
    if (w.get("flank_check", "OK") or "OK") != "OK":
        return False
    limit = _APPLIED_LIMIT if max_disagreement is _UNSET else max_disagreement
    mmf = w.get("max_mismatch_frac")
    if limit is None:
        return column_agreement_class(mmf) != "mixed"
    if mmf in (None, ""):
        return True
    return float(mmf) <= limit


ROWS = "ABCDEFGHIJKLMNOP"
COLS = 24

#: Where pileups are looked for, most specific first: a picked hit, then a
#: flagged mutation, then a general pass.  Each path serves twice, as the
#: place on disk under the run directory and as the href the page carries,
#: which holds because the page is written at the top of that directory.
PILEUP_SOURCES = (
    "pick/pileup",
    "demux_output/mutation/pileup",
    "demux_output/pileups/pileup",
)


def pileup_links(project_dir) -> Dict[str, str]:
    """Map ``"<plate>_<well>"`` to the page showing that well's reads.

    Three commands write pileups for the same well -- pick writes the hits,
    the mutation pass writes the flagged, and ``usortm pileups`` writes them
    all -- into three directories.  The newest wins.

    Taking the first directory that had one instead meant a page written by an
    earlier command shadowed a later one: a pileup rendered months ago sat in
    front of the same well re-rendered this morning, and the link opened the
    old page while the new one sat unread beside it.  Which command produced a
    page says nothing about how current it is; when it was written does.
    """
    newest: Dict[str, float] = {}
    links: Dict[str, str] = {}
    summary_newest: Dict[str, float] = {}
    summaries: Dict[str, str] = {}
    for rel in PILEUP_SOURCES:
        directory = os.path.join(str(project_dir), rel)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not (name.startswith("well_") and name.endswith(".html")):
                continue
            # A summary sits beside its pileup as well_<key>_summary.html.
            # Read as a pileup it would register a well named "<key>_summary",
            # which is no well at all.
            is_summary = name.endswith("_summary.html")
            end = -len("_summary.html") if is_summary else -len(".html")
            key = name[len("well_"):end]
            try:
                when = os.path.getmtime(os.path.join(directory, name))
            except OSError:
                continue
            seen, out = ((summary_newest, summaries) if is_summary
                         else (newest, links))
            if key not in seen or when > seen[key]:
                seen[key] = when
                out[key] = f"{rel}/{name}"

    # The summary is the page to open first; it links on to the pileup.  A
    # well without one -- rendered before summaries existed -- keeps its
    # direct link rather than losing it.
    merged = dict(links)
    merged.update(summaries)
    return merged


def _well_tip(plate, label, well, has_pileup) -> str:
    """The hover block for one well.

    Reports the codon agreement the call rests on rather than the share of
    reads assigned to a reference: the pipeline assigns a well once and marks
    every read in it with that call, so the read share is 100% everywhere and
    says nothing.
    """
    if well is None:
        return html.escape(
            f'<div style="line-height:1.2">'
            f'<div style="font-size:13px;">Plate {plate} &middot; '
            f'<b>{label}</b></div>'
            f'<div style="margin-top:4px;">no reads</div></div>', quote=True)

    reads = int(well.get("reads") or 0)
    agree = float(well.get("consensus_fraction") or 0.0)
    variant = well.get("variant") or "unassigned"
    cons = well.get("cons_check") or ""
    klass = column_agreement_class(well.get("max_mismatch_frac"))
    worst = well.get("max_mismatch_frac")

    lines = [f'<div style="font-size:11px;color:#666;margin-top:4px;">'
             f'Reads: {reads:,} &nbsp;|&nbsp; Codon agreement: {agree:.1%}</div>']
    if cons:
        lines.append(f'<div style="font-size:11px;color:#666;">{cons}</div>')
    if worst not in (None, ""):
        tone = {"mixed": "#dc2626", "watch": "#d97706"}.get(klass, "#6b7280")
        note = {"mixed": "mixed template", "watch": "worth checking"}.get(
            klass, "clean")
        lines.append(
            f'<div style="font-size:11px;color:{tone};margin-top:2px;">'
            f'{float(worst):.0%} of reads disagree at one position '
            f'&mdash; {note}</div>')
    if well.get("excluded"):
        lines.append(
            '<div style="font-size:11px;color:#dc2626;margin-top:2px;">'
            'left out of the pick: a read-level check found a second clone'
            '</div>')
    if has_pileup:
        lines.append('<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
                     'Click to view pileup</div>')
    return html.escape(
        f'<div style="line-height:1.2">'
        f'<div style="font-size:13px;">Plate {plate} &middot; <b>{label}</b></div>'
        f'<div style="margin-top:4px;">{variant}</div>'
        f'{"".join(lines)}</div>', quote=True)


def demux_plate_maps(well_data: Sequence[dict], designed: set,
                     links: Dict[str, str]) -> str:
    """One tabbed map per sort plate."""
    by_plate: Dict[str, Dict[str, dict]] = {}
    for w in well_data:
        by_plate.setdefault(str(w["plate"]), {})[w["well"]] = w
    if not by_plate:
        return ""

    def plate_key(p):
        try:
            return (0, int(p), "")
        except ValueError:
            return (1, 0, p)

    plates = sorted(by_plate, key=plate_key)
    grids = []
    for i, plate in enumerate(plates):
        wells = by_plate[plate]
        cells = []
        for letter in ROWS:
            for col in range(1, COLS + 1):
                label = f"{letter}{col}"
                w = wells.get(label)
                depth = int((w or {}).get("reads") or 0)
                cls = "w"
                if w is not None and depth >= TIER_READS["C"]:
                    variant = w.get("variant") or ""
                    klass = column_agreement_class(
                        w.get("max_mismatch_frac"))
                    # One flag for every way a well can fail to hold its
                    # designed sequence.  Drawn apart they were four colours
                    # over most of the plate, and which of them a well had is
                    # a question for a well, not for a plate.  The parent
                    # keeps its own: it carries no mutation, and a plate of
                    # parent wells is a sorting problem rather than a
                    # sequencing one.
                    if variant == "Parent":
                        cls += " parent"
                    elif not carries_designed_sequence(w, designed):
                        cls += " mut"
                    # Independent of what the well holds, and drawn as the
                    # well's edge: a parent well can also read uncleanly, and
                    # one corner cannot say both.
                    if klass == "watch":
                        cls += " watch"
                href = links.get(f"{plate}_{label}")
                tip = _well_tip(plate, label, w, bool(href))
                style = f"--f:{depth_colour(depth)}"
                if href:
                    cells.append(f'<a class="{cls}" href="{href}" '
                                 f'target="_blank" rel="noopener" '
                                 f'style="{style}" data-tip="{tip}"></a>')
                else:
                    cells.append(f'<i class="{cls}" style="{style}" '
                                 f'data-tip="{tip}"></i>')

        grids.append(
            f'<div class="plate" data-p="{plate}"{"" if i == 0 else " hidden"}>'
            f'<div class="grid"><div class="cols24">{"".join(cells)}</div>'
            f'</div></div>')

    n_linked = sum(1 for k in links if k.split("_")[0] in by_plate)
    note = (f"Wells link to their pileup; {n_linked:,} have one."
            if n_linked else
            "Wells link to their pileup once <code>usortm pick</code> has "
            "generated them.")
    return {
        "note": (f"Read depth per well. A red corner marks a mutation: the "
                 f"well does not hold the sequence designed for it. That "
                 f"covers a well matching no library member, a consensus "
                 f"differing from the design, a mixed template, and failed "
                 f"flanks. Amber marks the parent, which carries no mutation. "
                 f"A blue edge marks a well worth checking. {note}"),
        # The page steps through these one at a time rather than offering a
        # button per plate: fourteen buttons is a wall, and a plate map is read
        # in sequence far more often than jumped to.
        "plates": plates,
        "grids": "".join(grids),
        "legend": (
            '<div class="legend">'
            '<span class="ls"><i class="swatch mut"></i>mutation</span>'
            '<span class="ls"><i class="swatch parent"></i>parent</span>'
            '<span class="ls"><i class="swatch watch"></i>worth checking</span>'
            '</div>'),
    }


def verdict_plate(verdicts, rows: Dict[str, dict],
                  links: Optional[Dict[str, str]] = None,
                  layout: Optional[Dict[str, dict]] = None) -> dict:
    """The sequenced pick plate, each well marked by whether it holds what
    the merge put there.

    Filled by read depth like the other maps; a wrong well is drawn as a
    mutation, an empty one as not recovered, and a well the pick never used
    is blank.  The hover names the intended variant, what was read, the
    reason where it failed, and the source well it was picked from.
    """
    from usortm.verify import CONFIRMED, EMPTY, WRONG, failure_reason

    links = links or {}
    layout = layout or {}
    by_well = {v.well.upper(): v for v in verdicts}
    counts = {CONFIRMED: 0, WRONG: 0, EMPTY: 0}
    cells = []
    for letter in ROWS:
        for col in range(1, COLS + 1):
            label = f"{letter}{col}"
            v = by_well.get(label)
            if v is None:
                cells.append('<i class="w blank" data-tip="not on the plate"></i>')
                continue
            counts[v.status] = counts.get(v.status, 0) + 1
            row = rows.get(f"{v.plate}_{v.well}")
            reason = failure_reason(row, v) if v.status != CONFIRMED else ""
            depth = int((row or {}).get("reads") or 0)
            cls = "w"
            if v.status == EMPTY:
                cls += " none"
            elif reason == "flank mismatch":
                cls += " flank"
            elif v.status == WRONG:
                cls += " mut"
            lay = layout.get(label, {})
            src = f"{lay.get('source_plate', '')} {lay.get('source_well', '')}".strip()
            if lay.get("bench_well"):
                src += f" (colony plate {lay.get('bench_plate')} {lay.get('bench_well')})"
            lines = [f'<div style="font-size:13px;"><b>{label}</b></div>',
                     f'<div style="margin-top:4px;">placed <b>{v.expected}</b>'
                     + (f' from {src}' if src else '') + '</div>']
            if v.status == CONFIRMED:
                lines.append('<div style="font-size:11px;color:#1baf7a;margin-top:4px;">'
                             'holds it</div>')
            else:
                lines.append(f'<div style="font-size:11px;color:#666;margin-top:4px;">'
                             f'read as {v.observed or "nothing"}</div>')
                if reason:
                    lines.append(f'<div style="font-size:11px;color:#dc2626;">{reason}</div>')
            if row is not None:
                worst = row.get("max_mismatch_frac")
                agree = (f' &nbsp;|&nbsp; worst column {float(worst):.0%}'
                         if worst not in (None, "") and worst == worst else "")
                lines.append(f'<div style="font-size:11px;color:#666;margin-top:2px;">'
                             f'Reads: {depth:,}{agree}</div>')
            tip = html.escape(f'<div style="line-height:1.2">{"".join(lines)}</div>',
                              quote=True)
            style = f"--f:{depth_colour(depth)}"
            href = links.get(f"{v.plate}_{v.well}")
            if href:
                cells.append(f'<a class="{cls}" href="{href}" target="_blank" '
                             f'rel="noopener" style="{style}" data-tip="{tip}"></a>')
            else:
                cells.append(f'<i class="{cls}" style="{style}" data-tip="{tip}"></i>')
    n = sum(counts.values())
    return {
        "counts": counts,
        "note": (f"The destination plate as sequenced, each well judged against the "
                 f"variant the merge placed there: {counts[CONFIRMED]} of {n} hold it, "
                 f"{counts[WRONG]} hold something else or read unclean, "
                 f"{counts[EMPTY]} returned too few reads to call."),
        "grid": (f'<div class="grid"><div class="cols24">{"".join(cells)}'
                 f'</div></div>'),
        "legend": (
            '<div class="legend">'
            '<span class="ls"><i class="swatch mut"></i>holds something else</span>'
            '<span class="ls"><i class="swatch flank"></i>flank mismatch</span>'
            '<span class="ls"><i class="swatch none"></i>too few reads</span>'
            '<span class="ls"><i class="swatch blank"></i>not on the plate</span>'
            '</div>'),
    }


def pick_plate(pick_list: Optional[List[dict]],
               links: Dict[str, str],
               well_class: Optional[Dict[str, str]] = None) -> str:
    """The destination plate as pick built it, or a note saying why not.

    *pick_list* is None when no pick exists or when the one on disk predates
    this demux; the section then says so rather than rendering a plate that
    describes different wells.

    *well_class* maps ``"<plate>_<well>"`` to how cleanly that well reads, so a
    picked well carries the same mark it has on the demux map.  Taken from the
    wells rather than from the pick list, which need not carry the fraction.
    """
    well_class = well_class or {}
    if pick_list is None:
        return {"note": ("Not shown: no current pick for this run. Run "
                         "<code>usortm pick</code> to populate it."),
                "grid": "", "legend": ""}

    by_target = {p["target_well"]: p for p in pick_list if p.get("target_well")}
    if not by_target:
        return {"note": "", "grid": "", "legend": ""}

    filled = empty = blank = 0
    cells = []
    for letter in ROWS:
        for col in range(1, COLS + 1):
            label = f"{letter}{col}"
            slot = by_target.get(label)
            if slot is None:
                blank += 1
                cells.append('<i class="w blank" data-tip="blank by design">'
                             '</i>')
                continue
            variant = slot.get("variant") or ""
            if slot.get("empty"):
                empty += 1
                tip = html.escape(
                    f'<div style="line-height:1.2">'
                    f'<div style="font-size:13px;"><b>{label}</b></div>'
                    f'<div style="margin-top:4px;">{variant}</div>'
                    f'<div style="font-size:11px;color:#666;margin-top:4px;">'
                    f'not recovered</div></div>', quote=True)
                cells.append(f'<i class="w none" data-tip="{tip}"></i>')
                continue
            filled += 1
            reads = int(slot.get("reads") or 0)
            src = f'{slot.get("source_plate")}_{slot.get("source_well")}'
            klass = well_class.get(src) or column_agreement_class(
                slot.get("max_mismatch_frac"))
            extra = ""
            if klass == "watch":
                extra = ('<div style="font-size:11px;color:#2a78d6;'
                         'margin-top:2px;">worth checking</div>')
            tip = html.escape(
                f'<div style="line-height:1.2">'
                f'<div style="font-size:13px;">{label} &middot; '
                f'<b>{variant}</b></div>'
                f'<div style="margin-top:4px;">from plate '
                f'{slot.get("source_plate")} {slot.get("source_well")}</div>'
                f'<div style="font-size:11px;color:#666;margin-top:4px;">'
                f'Reads: {reads:,} &nbsp;|&nbsp; Codon agreement: '
                f'{float(slot.get("consensus_fraction") or 0):.1%}</div>'
                f'{extra}</div>', quote=True)
            href = links.get(src)
            style = f"--f:{depth_colour(reads)}"
            cls = "w watch" if klass == "watch" else "w"
            if href:
                cells.append(f'<a class="{cls}" href="{href}" target="_blank" '
                             f'rel="noopener" style="{style}" '
                             f'data-tip="{tip}"></a>')
            else:
                cells.append(f'<i class="{cls}" style="{style}" '
                             f'data-tip="{tip}"></i>')

    return {
        "note": (f"The destination plate as picked, filled by the read depth of "
                 f"each source well. {filled} filled, {empty} not recovered, "
                 f"{blank} blank by design."),
        "grid": (f'<div class="grid"><div class="cols24">{"".join(cells)}'
                 f'</div></div>'),
        "legend": (
            '<div class="legend">'
            '<span class="ls"><i class="swatch none"></i>not recovered</span>'
            '<span class="ls"><i class="swatch blank"></i>blank by design'
            '</span>'
            '<span class="ls"><i class="swatch watch"></i>worth checking</span>'
            '</div>'),
    }
