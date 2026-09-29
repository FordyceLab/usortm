"""The plate map for ``usortm demux --expected``.

Drawn the way the summary page draws its demux maps (:mod:`.plates`): a well
is a cell filled by read depth, a red corner marks a well that does not hold
the construct expected in it, and each well with reads opens its pileup, or
the summary of that pileup, against that construct.  The only difference is
the grid: a 96-well plate barcoded as a LevSeq quadrant is drawn in its own
A1-H12 coordinates, rather than as the scattered quarter of a 384-well plate
it occupies.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

from usortm.demux.expected_plate import QUADRANTS, ExpectedPlate, quadrant_to_384

from .charts import colorbar, depth_colour

_CSS_PATH = Path(__file__).with_name("summary.css")

_EXTRA_CSS = """
/* A shade lighter than the summary page's dark surface: this page is one
   plate on its own, and the wells read better off a less deep ground. */
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-1:#252523; --surface-2:#2e2e2b; --rule:#43433f; }
}
:root[data-theme="dark"] { --surface-1:#252523; --surface-2:#2e2e2b; --rule:#43433f; }
main { max-width:46rem; }
/* Row letters and column numbers around the wells.  The axis cells are the
   grid's own first row and column, so they cannot drift from the wells. */
.cols12, .cols24 { display:grid; gap:2px; width:100%; }
.cols12 { grid-template-columns:1.1rem repeat(12, 1fr); }
.cols24 { grid-template-columns:1.1rem repeat(24, 1fr); }
.ax { font-size:.62rem; color:var(--text-muted); text-align:center;
      align-self:center; line-height:1; font-variant-numeric:tabular-nums; }
.ax.top { height:1rem; display:flex; align-items:flex-end; justify-content:center; }
/* The depth ramp beside its plate, the height of the wells, so its ends sit
   level with rows A and H.  The label hangs below the ramp rather than taking
   height from it, and the ramp starts below the column numbers. */
.prow { display:flex; gap:.7rem; align-items:stretch; }
.prow .grid { flex:0 1 27rem; min-width:0; }
.prow .grid.wide { flex-basis:42rem; }
.prow .cbcol { position:relative; padding:calc(1rem + 5px) 0 3px; }
.prow .cbcol .cblab { position:absolute; top:100%; left:0; margin-top:.3rem;
                      white-space:nowrap; text-align:left; }
.pcap { font-size:.75rem; color:var(--text-muted); margin:.9rem 0 0; }
.plate + .plate { margin-top:2rem; }
.pc .legend { margin-top:.4rem; }
.w[data-links] { cursor:pointer; }
.w[data-links]:hover { outline:2px solid var(--series-1); outline-offset:1px; z-index:3; }
/* Light / dark, top right. */
.themetoggle { position:fixed; top:.9rem; right:1rem; z-index:60;
  font:inherit; font-size:.75rem; padding:.25rem .6rem; cursor:pointer;
  border:1px solid var(--rule); border-radius:4px;
  background:var(--surface-2); color:var(--text-secondary); }
/* The two pages a well opens, shown where it was clicked. */
.wellmenu { position:absolute; z-index:55; display:none; padding:.3rem;
  border:1px solid var(--rule); border-radius:5px; background:var(--surface-1);
  box-shadow:0 2px 8px rgba(0,0,0,.2); font-size:.8rem; }
.wellmenu.on { display:flex; gap:.25rem; }
.wellmenu a { padding:.2rem .55rem; border-radius:3px; color:var(--text-primary);
              text-decoration:none; }
.wellmenu a:hover { background:var(--surface-2); }
"""

_SCRIPT = """
var tip = document.createElement("div");
tip.className = "tip";
document.body.appendChild(tip);
document.addEventListener("mouseover", function (e) {
  var w = e.target.closest("[data-tip]");
  if (!w || menu.classList.contains("on")) { tip.classList.remove("on"); return; }
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

var menu = document.createElement("div");
menu.className = "wellmenu";
document.body.appendChild(menu);
function closeMenu() { menu.classList.remove("on"); }
document.addEventListener("click", function (e) {
  if (menu.contains(e.target)) { closeMenu(); return; }
  var w = e.target.closest("[data-links]");
  if (!w) { closeMenu(); return; }
  var links = JSON.parse(w.dataset.links);
  menu.innerHTML = "";
  [["pileup", "Pileup"], ["summary", "Summary"]].forEach(function (k) {
    if (!links[k[0]]) return;
    var a = document.createElement("a");
    a.href = links[k[0]]; a.target = "_blank"; a.rel = "noopener";
    a.textContent = k[1];
    menu.appendChild(a);
  });
  var r = w.getBoundingClientRect();
  menu.style.left = (r.left + window.scrollX) + "px";
  menu.style.top = (r.bottom + window.scrollY + 4) + "px";
  tip.classList.remove("on");
  menu.classList.add("on");
});
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") closeMenu();
});

var root = document.documentElement, btn = document.querySelector(".themetoggle");
function current() {
  return root.dataset.theme ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function label() { btn.textContent = current() === "dark" ? "Light" : "Dark"; }
try { var saved = localStorage.getItem("usortm-theme"); if (saved) root.dataset.theme = saved; }
catch (e) {}
label();
btn.addEventListener("click", function () {
  root.dataset.theme = current() === "dark" ? "light" : "dark";
  try { localStorage.setItem("usortm-theme", root.dataset.theme); } catch (e) {}
  label();
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


def _tip(title: str, v, state: str, has_links: bool) -> str:
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
    if has_links:
        lines.append('<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
                     'Click for the pileup and its summary</div>')
    return _esc(f'<div style="line-height:1.2"><div style="font-size:13px;">'
                f'{_esc(title)}</div><div style="margin-top:4px;">{_esc(v.verdict)}</div>'
                f'{"".join(lines)}</div>')


def _cell(v, state: str, title: str, links: Optional[dict]) -> str:
    """One well: depth fill, a red corner unless it holds what was expected."""
    import json

    depth = v.reads if v is not None else 0
    cls = "w"
    if state == "empty" and v is None:
        cls += " blank"
    elif v is not None and v.verdict != "match":
        cls += " mut"
    tip = _tip(title, v, state, bool(links))
    style = "" if "blank" in cls else f"--f:{depth_colour(depth)}"
    data = f' data-links="{_esc(json.dumps(links))}"' if links else ""
    return f'<i class="{cls}" style="{style}" data-tip="{tip}"{data}></i>'


def _state(plate: ExpectedPlate, key) -> str:
    exp = plate.wells.get(key)
    if exp is None:
        return "unlisted"
    return "empty" if exp.empty else "listed"


def _grid(rows: int, cols: int, cell_for) -> str:
    """Wells with their row letters and column numbers."""
    parts = ['<span class="ax top"></span>']
    parts += [f'<span class="ax top">{c}</span>' for c in range(1, cols + 1)]
    for r in range(1, rows + 1):
        letter = chr(ord("A") + r - 1)
        parts.append(f'<span class="ax">{letter}</span>')
        parts += [cell_for(r, letter, c) for c in range(1, cols + 1)]
    return f'<div class="cols{cols}">{"".join(parts)}</div>'


def _plate_block(caption: str, grid: str, wide: bool) -> str:
    """A plate's grid with the depth ramp beside it, and its caption below."""
    return (f'<div class="plate"><div class="prow">'
            f'<div class="grid{" wide" if wide else ""}">{grid}</div>'
            f'<div class="cbcol">{colorbar()}<div class="cblab">reads per well</div>'
            f'</div></div><p class="pcap">{_esc(caption)}</p></div>')


def plate_maps(plate: ExpectedPlate, verdicts: Sequence,
               links: Optional[Dict[str, dict]] = None) -> str:
    """The maps, one per plate or quadrant as the expected plate was laid out.

    Args:
        plate: The expectation.
        verdicts: WellVerdicts from :func:`usortm.demux.verify.verify_plate`.
        links: ``{"<plate>_<384-well>": {"pileup": href, "summary": href}}``.
    """
    links = links or {}
    by_key = {(v.plate, v.well): v for v in verdicts}
    maps = []
    if plate.layout == "96":
        used = {(w.plate, quadrant_of_384(w.well)[0]) for w in plate.wells.values()}
        used |= {(v.plate, quadrant_of_384(v.well)[0]) for v in verdicts}
        for p, q in sorted(used):
            def cell(r, letter, c, p=p, q=q):
                well = quadrant_to_384(q, r, c)
                title = f"Plate {p} · {QUADRANTS[q]} {letter}{c} ({well})"
                return _cell(by_key.get((p, well)), _state(plate, (p, well)),
                             title, links.get(f"{p}_{well}"))

            rb = (p - 1) * 4 + q + 1
            maps.append(_plate_block(f"Plate {p} · {QUADRANTS[q]} quadrant (RB{rb:02d})",
                                     _grid(8, 12, cell), wide=False))
    else:
        plates = sorted({w.plate for w in plate.wells.values()} | {v.plate for v in verdicts})
        for p in plates:
            def cell(r, letter, c, p=p):
                well = f"{letter}{c}"
                return _cell(by_key.get((p, well)), _state(plate, (p, well)),
                             f"Plate {p} · {well}", links.get(f"{p}_{well}"))

            maps.append(_plate_block(f"Plate {p}", _grid(16, 24, cell), wide=True))
    return "".join(maps)


def write_plate_check_page(
    plate: ExpectedPlate,
    verdicts: Sequence,
    path,
    links: Optional[Dict[str, dict]] = None,
) -> Path:
    """Write the plate check page: the plate maps and their legend.

    Args:
        links: ``{"<plate>_<384-well>": {"pileup": href, "summary": href}}``,
            relative to the page's own directory.  The page goes at the top of
            the run so every page it opens is beneath it: a ``file://`` page in
            Safari can read below its directory and not above it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    css = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""
    legend = ('<div class="legend">'
              '<span class="ls"><i class="swatch mut"></i>not the expected construct</span>'
              '<span class="ls"><i class="swatch blank"></i>expected empty</span>'
              + ('<span class="ls">click a well for its pileup and summary</span>'
                 if links else "")
              + '</div>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>uSort-M Plate Check</title>
<style>{css}{_EXTRA_CSS}</style></head>
<body><button type="button" class="themetoggle" aria-label="Switch light or dark">Dark</button>
<main class="pc">
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
