"""The plate map for ``usortm demux --expected``.

Drawn with :mod:`.platemap`: each well filled by read depth, a red corner
where it does not hold the construct expected in it, hatching where it was
expected empty, and a click through to the summary of its pileup against
that construct.  A 96-well plate barcoded as a LevSeq quadrant is drawn in
its own A1-H12 coordinates, rather than as the scattered quarter of a
384-well plate it occupies.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

from usortm.demux.expected_plate import QUADRANTS, ExpectedPlate, quadrant_to_384

from .platemap import Well, legend, plate_map_page, render_plate


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def quadrant_of_384(well: str) -> tuple:
    """``(quadrant, row96, col96)`` of a 384-well position; inverse of quadrant_to_384."""
    row = ord(well[0]) - ord("A") + 1
    col = int(well[1:])
    quadrant = (0 if row % 2 else 2) + (0 if col % 2 else 1)
    return quadrant, (row + 1) // 2, (col + 1) // 2


def _tip(title: str, v, state: str, linked: bool) -> str:
    """The hover card for one well, as HTML, in the summary page's form."""
    head = f'<div style="line-height:1.2"><div style="font-size:13px;">{_esc(title)}</div>'
    if v is None:
        body = "expected empty; no reads" if state == "empty" else "no reads"
        return f'{head}<div style="margin-top:4px;">{body}</div></div>'
    small = 'style="font-size:11px;color:#666;"'
    lines = [f'<div style="font-size:11px;color:#666;margin-top:4px;">'
             f'Expected: {_esc(v.expected or "empty")} &nbsp;|&nbsp; '
             f'Reads: {v.reads:,}</div>']
    if v.observed and v.observed != v.expected:
        lines.append(f'<div {small}>Holds: {_esc(v.observed)}</div>')
    if v.differences:
        lines.append(f'<div {small}>{_esc(" ".join(v.differences[:6]))}'
                     + (" …" if len(v.differences) > 6 else "") + "</div>")
    if v.protein_changes:
        lines.append(f'<div {small}>{_esc(" ".join(v.protein_changes[:6]))}</div>')
    if v.note:
        lines.append(f'<div {small}>{_esc(v.note)}</div>')
    if linked:
        lines.append('<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
                     'Click for the pileup summary</div>')
    return (f'{head}<div style="margin-top:4px;">{_esc(v.verdict)}</div>'
            f'{"".join(lines)}</div>')


def _state(plate: ExpectedPlate, key) -> str:
    exp = plate.wells.get(key)
    if exp is None:
        return "unlisted"
    return "empty" if exp.empty else "listed"


def _well(plate, by_key, links, key, row, col, title) -> Well:
    v = by_key.get(key)
    state = _state(plate, key)
    href = links.get(f"{key[0]}_{key[1]}")
    if v is None:
        flag = "blank" if state == "empty" else ""
    else:
        flag = "" if v.verdict == "match" else "mut"
    return Well(row=row, col=col, depth=v.reads if v else 0, flag=flag,
                tip=_tip(title, v, state, bool(href)), href=href)


def plate_maps(plate: ExpectedPlate, verdicts: Sequence,
               links: Optional[Dict[str, str]] = None) -> str:
    """The maps, one per plate or quadrant as the expected plate was laid out.

    Args:
        plate: The expectation.
        verdicts: WellVerdicts from :func:`usortm.demux.verify.verify_plate`.
        links: ``{"<plate>_<384-well>": href}`` of the page each well opens.
    """
    links = links or {}
    by_key = {(v.plate, v.well): v for v in verdicts}
    maps = []
    if plate.layout == "96":
        used = {(w.plate, quadrant_of_384(w.well)[0]) for w in plate.wells.values()}
        used |= {(v.plate, quadrant_of_384(v.well)[0]) for v in verdicts}
        for p, q in sorted(used):
            wells = []
            for r in range(1, 9):
                letter = chr(ord("A") + r - 1)
                for c in range(1, 13):
                    well = quadrant_to_384(q, r, c)
                    title = f"Plate {p} · {QUADRANTS[q]} {letter}{c} ({well})"
                    wells.append(_well(plate, by_key, links, (p, well), r, c, title))
            rb = (p - 1) * 4 + q + 1
            maps.append(render_plate(
                wells, rows=8, cols=12,
                caption=f"Plate {p} · {QUADRANTS[q]} quadrant (RB{rb:02d})"))
    else:
        plates = sorted({w.plate for w in plate.wells.values()} | {v.plate for v in verdicts})
        for p in plates:
            wells = []
            for r in range(1, 17):
                letter = chr(ord("A") + r - 1)
                for c in range(1, 25):
                    well = f"{letter}{c}"
                    wells.append(_well(plate, by_key, links, (p, well), r, c,
                                       f"Plate {p} · {well}"))
            maps.append(render_plate(wells, rows=16, cols=24, caption=f"Plate {p}"))
    return "".join(maps)


def write_plate_check_page(
    plate: ExpectedPlate,
    verdicts: Sequence,
    path,
    links: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the plate check page: the plate maps and their legend.

    Args:
        links: ``{"<plate>_<384-well>": href}`` of the page each well opens,
            relative to the page's own directory.  The page goes at the top of
            the run so every page it opens is beneath it: a ``file://`` page
            in Safari can read below its directory and not above it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = plate_maps(plate, verdicts, links) + legend(
        [("mut", "not the expected construct"), ("blank", "expected empty")],
        note="click a well for its pileup summary" if links else "")
    meta = (f"expected: <b>{_esc(Path(plate.source).name)}</b> &middot; "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    path.write_text(plate_map_page("uSort-M Plate Check", body, meta=meta))
    return path
