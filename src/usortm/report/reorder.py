"""The re-order round, laid into the run's own summary page.

A re-order round is not a second experiment with its own page.  It finishes
the first one: the constructs it buys are exactly the ones the sort missed, so
its result belongs in the same recovery figure rather than beside it.  What it
adds is a different question -- every well here has an intended construct, so
the plate shows whether each assembly worked rather than what each well holds.
"""
from __future__ import annotations

import html
from collections import Counter
from typing import Dict, List, Optional, Sequence

from usortm.verify import CONFIRMED, EMPTY, WRONG, failure_reason

from .charts import bar, depth_colour
from .plates import COLS, ROWS


def _key(plate, well) -> str:
    return f"{plate}_{well}"


def outcome_rows(summary: dict) -> str:
    """Wells by outcome, as table rows."""
    n = summary["n_wells"] or 1
    out = []
    for status, label, tone in ((CONFIRMED, "Held the construct", "good"),
                                (WRONG, "Held something else", "bad"),
                                (EMPTY, "Nothing grew", "warn")):
        count = summary["wells"].get(status, 0)
        pct = 100 * count / n
        out.append(
            f'<tr><td class="name">{label}</td>'
            f'<td>{count:,} <span class="u">{pct:.1f}%</span></td>'
            f'<td>{bar(pct, tone)}</td></tr>')
    return "".join(out)


def replicate_rows(summary: dict) -> str:
    """Each picked colony's outcome, as table rows.

    Colonies are picked in replicate to survive a bad one, so the reason to
    read them apart is that a replicate failing far more often than its
    neighbours points at the picking or the plate rather than at the
    constructs.
    """
    out = []
    for n, tally in summary["by_replicate"].items():
        total = sum(tally.values()) or 1
        ok = tally.get(CONFIRMED, 0)
        out.append(
            f'<tr><td class="name">Colony {n}</td>'
            f'<td>{ok:,} <span class="u">{100 * ok / total:.0f}%</span></td>'
            f'<td>{tally.get(WRONG, 0):,}</td>'
            f'<td>{tally.get(EMPTY, 0):,}</td></tr>')
    return "".join(out)


def failure_rows(verdicts: Sequence, rows: Dict) -> str:
    """Why the wells that failed did, most common first.

    Empty when nothing failed, so the section is left out rather than drawn as
    a table of zeros.
    """
    counts = Counter()
    for v in verdicts:
        reason = failure_reason(rows.get(_key(v.plate, v.well)), v)
        if reason:
            counts[reason] += 1
    if not counts:
        return ""
    total = sum(counts.values())
    out = []
    for reason, count in counts.most_common():
        pct = 100 * count / total
        out.append(
            f'<tr><td class="name">{html.escape(reason)}</td>'
            f'<td>{count:,} <span class="u">{pct:.0f}%</span></td>'
            f'<td>{bar(pct, "bad")}</td></tr>')
    return "".join(out)


def _tip(v, row, reason) -> str:
    """The hover block for one intended well."""
    lines = [f'<div style="font-size:13px;">{v.well} &middot; colony '
             f'{v.replicate}</div>',
             f'<div style="margin-top:4px;">ordered <b>{v.expected}</b></div>']
    if v.status == CONFIRMED:
        lines.append('<div style="font-size:11px;color:#1baf7a;'
                     'margin-top:4px;">held the construct</div>')
    else:
        got = v.observed or "nothing"
        lines.append(f'<div style="font-size:11px;color:#666;margin-top:4px;">'
                     f'read as {got}</div>')
        if reason:
            lines.append(f'<div style="font-size:11px;color:#dc2626;">'
                         f'{reason}</div>')
    if row is not None:
        lines.append(f'<div style="font-size:11px;color:#666;margin-top:2px;">'
                     f'Reads: {int(row.get("reads") or 0):,}</div>')
    return html.escape(f'<div style="line-height:1.2">{"".join(lines)}</div>',
                       quote=True)


def reorder_plate(verdicts: Sequence, rows: Dict,
                  links: Optional[Dict[str, str]] = None) -> dict:
    """The re-ordered plate, marked by whether each assembly worked.

    Filled by read depth like the demux maps, so the two read alike, with a
    corner carrying the outcome.  Wells the order never reached are hatched
    rather than drawn empty: on this plate an untouched well and a well that
    failed to grow mean different things, and only one of them is a result.

    Returns ``note``, ``plates``, ``grids`` and ``legend``, as the demux maps
    do, so the page can step through several plates the same way.
    """
    links = links or {}
    by_plate: Dict[int, Dict[str, object]] = {}
    for v in verdicts:
        by_plate.setdefault(v.plate, {})[v.well] = v
    if not by_plate:
        return {"note": "", "plates": [], "grids": "", "legend": ""}

    plates = sorted(by_plate)
    grids = []
    for i, plate in enumerate(plates):
        wanted = by_plate[plate]
        cells = []
        for letter in ROWS:
            for col in range(1, COLS + 1):
                label = f"{letter}{col}"
                v = wanted.get(label)
                if v is None:
                    cells.append('<i class="w blank" '
                                 'data-tip="not part of the order"></i>')
                    continue
                row = rows.get(_key(plate, label))
                reason = failure_reason(row, v)
                depth = int((row or {}).get("reads") or 0)
                cls = "w"
                if v.status == WRONG:
                    cls += " mut"
                elif v.status == EMPTY:
                    cls += " none"
                style = f"--f:{depth_colour(depth)}"
                tip = _tip(v, row, reason)
                href = links.get(_key(plate, label))
                if href:
                    cells.append(f'<a class="{cls}" href="{href}" '
                                 f'target="_blank" rel="noopener" '
                                 f'style="{style}" data-tip="{tip}"></a>')
                else:
                    cells.append(f'<i class="{cls}" style="{style}" '
                                 f'data-tip="{tip}"></i>')
        grids.append(
            f'<div class="plate" data-p="{plate}"'
            f'{"" if i == 0 else " hidden"}>'
            f'<div class="grid"><div class="cols24">{"".join(cells)}</div>'
            f'</div></div>')

    return {
        "note": ("Every well the order reached, filled by read depth. A red "
                 "corner marks a well that held something other than the "
                 "construct ordered for it; a grey well grew too little to "
                 "call. Hatched wells were not part of the order."),
        "plates": plates,
        "grids": "".join(grids),
        "legend": (
            '<div class="legend">'
            '<span class="ls"><i class="swatch mut"></i>held something else'
            '</span>'
            '<span class="ls"><i class="swatch none"></i>nothing grew</span>'
            '<span class="ls"><i class="swatch blank"></i>not ordered</span>'
            '</div>'),
    }
