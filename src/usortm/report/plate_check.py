"""The plate map for ``usortm demux --expected``.

Drawn the way the summary page draws its demux maps (:mod:`.plates`): a well
is a cell filled by read depth, a red corner marks a well that does not hold
the construct expected in it, and each well with reads links to a pileup of
its reads against that construct.  The only difference is the grid: a 96-well
plate barcoded as a LevSeq quadrant is drawn in its own A1-H12 coordinates,
rather than as the scattered quarter of a 384-well plate it occupies.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

from usortm.demux.expected_plate import QUADRANTS, ExpectedPlate, quadrant_to_384

from .charts import colorbar, depth_colour

_CSS_PATH = Path(__file__).with_name("summary.css")

# A 96-well plate on the same terms as the 384-well maps: the same cells, at
# the size a quarter of a 384-well plate would give them.
_EXTRA_CSS = """
main { max-width:44rem; }
.cols12 { display:grid; grid-template-columns:repeat(12, 1fr); gap:2px;
          width:100%; }
.plate h3 { font-size:.85rem; font-weight:600; margin:1.2rem 0 .35rem;
            color:var(--text-secondary); }
/* The depth ramp beside its plate, the height of the grid, so its ends sit
   level with the plate's top and bottom rows.  The label hangs below the
   ramp rather than taking height from it. */
.prow { display:flex; gap:.7rem; align-items:stretch; }
.prow .grid { flex:0 1 26rem; min-width:0; }
.prow .grid.wide { flex-basis:40rem; }
.prow .cbcol { position:relative; padding:3px 0; }
.prow .cbcol .cblab { position:absolute; top:100%; left:0; margin-top:.3rem;
                      white-space:nowrap; text-align:left; }
.pc .legend { margin-top:1.4rem; }
"""

_SCRIPT = """
var tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
document.addEventListener("mouseover", function (e) {
  var w = e.target.closest("[data-tip]");
  if (!w) { tip.classList.remove("on"); return; }
  tip.innerHTML = w.dataset.tip;
  tip.classList.add("on");
});
document.addEventListener("mousemove", function (e) {
  if (!tip.classList.contains("on")) return;
  var pad = 14, r = tip.getBoundingClientRect();
  var x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > window.innerWidth) x = e.clientX - r.width - pad;
  if (y + r.height > window.innerHeight) y = e.clientY - r.height - pad;
  tip.style.left = (x + window.scrollX) + "px";
  tip.style.top = (y + window.scrollY) + "px";
});
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def quadrant_of_384(well: str) -> tuple:
    """``(quadrant, row96, col96)`` of a 384-well position; inverse of quadrant_to_384."""
    row = ord(well[0]) - ord("A") + 1
    col = int(well[1:])
    quadrant = (0 if row % 2 else 2) + (0 if col % 2 else 1)
    return quadrant, (row + 1) // 2, (col + 1) // 2


def _tip(title: str, v, state: str, has_pileup: bool) -> str:
    """The hover block for one well, in the summary page's form."""
    if v is None:
        body = "expected empty; no reads" if state == "empty" else "no reads"
        return _esc(f'<div style="line-height:1.2"><div style="font-size:13px;">'
                    f'{_esc(title)}</div><div style="margin-top:4px;">{body}</div></div>')
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
    if has_pileup:
        lines.append('<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
                     'Click to view pileup</div>')
    return _esc(f'<div style="line-height:1.2"><div style="font-size:13px;">'
                f'{_esc(title)}</div><div style="margin-top:4px;">{_esc(v.verdict)}</div>'
                f'{"".join(lines)}</div>')


def _cell(v, state: str, title: str, href: Optional[str]) -> str:
    """One well: depth fill, a red corner unless it holds what was expected."""
    depth = v.reads if v is not None else 0
    cls = "w"
    if state == "empty" and v is None:
        cls += " blank"
    elif v is not None and v.verdict != "match":
        cls += " mut"
    tip = _tip(title, v, state, bool(href))
    style = "" if "blank" in cls else f"--f:{depth_colour(depth)}"
    if href:
        return (f'<a class="{cls}" href="{_esc(href)}" target="_blank" '
                f'rel="noopener" style="{style}" data-tip="{tip}"></a>')
    return f'<i class="{cls}" style="{style}" data-tip="{tip}"></i>'


def _state(plate: ExpectedPlate, key) -> str:
    exp = plate.wells.get(key)
    if exp is None:
        return "unlisted"
    return "empty" if exp.empty else "listed"


def _plate_block(title: str, grid: str, wide: bool) -> str:
    """A plate's heading, then its grid with the depth ramp beside it."""
    return (f'<div class="plate"><h3>{_esc(title)}</h3><div class="prow">'
            f'<div class="grid{" wide" if wide else ""}">{grid}</div>'
            f'<div class="cbcol">{colorbar()}<div class="cblab">reads per well</div>'
            f'</div></div></div>')


def plate_maps(plate: ExpectedPlate, verdicts: Sequence,
               links: Optional[Dict[str, str]] = None) -> str:
    """The maps, one per plate or quadrant as the expected plate was laid out.

    Args:
        plate: The expectation.
        verdicts: WellVerdicts from :func:`usortm.demux.verify.verify_plate`.
        links: ``{"<plate>_<384-well>": href}`` of each well's pileup.
    """
    links = links or {}
    by_key = {(v.plate, v.well): v for v in verdicts}
    maps = []
    if plate.layout == "96":
        used = {(w.plate, quadrant_of_384(w.well)[0]) for w in plate.wells.values()}
        used |= {(v.plate, quadrant_of_384(v.well)[0]) for v in verdicts}
        for p, q in sorted(used):
            cells = []
            for r in range(1, 9):
                letter = chr(ord("A") + r - 1)
                for c in range(1, 13):
                    well = quadrant_to_384(q, r, c)
                    title = f"Plate {p} · {QUADRANTS[q]} {letter}{c} ({well})"
                    cells.append(_cell(by_key.get((p, well)), _state(plate, (p, well)),
                                       title, links.get(f"{p}_{well}")))
            rb = (p - 1) * 4 + q + 1
            maps.append(_plate_block(
                f"Plate {p} · {QUADRANTS[q]} quadrant (RB{rb:02d})",
                f'<div class="cols12">{"".join(cells)}</div>', wide=False))
    else:
        plates = sorted({w.plate for w in plate.wells.values()} | {v.plate for v in verdicts})
        for p in plates:
            cells = []
            for r in range(1, 17):
                letter = chr(ord("A") + r - 1)
                for c in range(1, 25):
                    well = f"{letter}{c}"
                    cells.append(_cell(by_key.get((p, well)), _state(plate, (p, well)),
                                       f"Plate {p} · {well}", links.get(f"{p}_{well}")))
            maps.append(_plate_block(
                f"Plate {p}", f'<div class="cols24">{"".join(cells)}</div>', wide=True))
    return "".join(maps)


def write_plate_check_page(
    plate: ExpectedPlate,
    verdicts: Sequence,
    path,
    links: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the plate check page: the plate maps and their legend.

    Args:
        links: ``{"<plate>_<384-well>": href}``, relative to the page's own
            directory.  The page goes at the top of the run so every pileup
            is beneath it: a ``file://`` page in Safari can read below its
            directory and not above it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    css = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""
    legend = ('<div class="legend">'
              '<span class="ls"><i class="swatch mut"></i>not the expected construct</span>'
              '<span class="ls"><i class="swatch blank"></i>expected empty</span>'
              + ('<span class="ls">click a well for its pileup</span>' if links else "")
              + '</div>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>uSort-M Plate Check</title>
<style>{css}{_EXTRA_CSS}</style></head>
<body><main class="pc">
  <h1>uSort-M Plate Check</h1>
  <div class="meta">expected: <b>{_esc(Path(plate.source).name)}</b> &middot;
    {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  {plate_maps(plate, verdicts, links)}
  {legend}
</main>
<script>{_SCRIPT}</script>
</body></html>
"""
    path.write_text(page)
    return path
