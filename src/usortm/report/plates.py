"""Plate maps for the summary page.

A well is a cell filled by read depth, with a corner triangle carrying what it
contains -- colour alone cannot say two things about one square.  Where a
pileup has been rendered the cell is a link to it, so a plate map is the way
into the reads rather than only a picture of them.
"""
from __future__ import annotations

import html
import os
from typing import Dict, List, Optional, Sequence

from usortm.cli.report import NOT_THE_DESIGNED_SEQUENCE
from usortm.demux.utils import (MIXED_TEMPLATE_THRESHOLD,
                                MIXED_TEMPLATE_WATCH, column_agreement_class)

from .charts import TIER_READS, depth_colour


def carries_designed_sequence(w: dict, designed: set) -> bool:
    """Whether a well holds the member it was assigned, read cleanly.

    One rule for the plate maps' flag and for the parameters the recovery
    curve is drawn on.  A well flagged on the map and counted as on-target in
    the same report is a contradiction a reader has no way to settle, and the
    two tests were written out twice before.  Mirrors the tier test in
    :func:`usortm.cli.report._compute_quality_bins`.

    A well missing flank or agreement data is not failed for missing it: the
    fields arrive from stages that a run may not have reached.
    """
    if w.get("variant") not in designed:
        return False
    if (w.get("consensus_fraction") or 0) <= 0.9:
        return False
    if w.get("cons_check", "") in NOT_THE_DESIGNED_SEQUENCE:
        return False
    if (w.get("flank_check", "OK") or "OK") != "OK":
        return False
    return column_agreement_class(w.get("max_mismatch_frac")) != "mixed"


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
    """Map ``"<plate>_<well>"`` to the page showing that well's reads."""
    links: Dict[str, str] = {}
    for rel in PILEUP_SOURCES:
        directory = os.path.join(str(project_dir), rel)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not (name.startswith("well_") and name.endswith(".html")):
                continue
            key = name[len("well_"):-len(".html")]
            links.setdefault(key, f"{rel}/{name}")
    return links


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
                 f"well's sequence is not the one designed for it, whether an "
                 f"insert that could not be read, a mixed template, or a "
                 f"difference from the design. Amber marks the parent, which "
                 f"carries no mutation. A blue edge marks a well worth "
                 f"checking. {note}"),
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
