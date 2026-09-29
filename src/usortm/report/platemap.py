"""Plate maps in the house style, as a fragment or as a page of their own.

A well is a square filled by read depth on the house colormap
(:func:`.charts.depth_colour`), with the depth ramp standing beside the plate
at the height of its wells.  One status rides on each well as a red corner;
a well with nothing expected in it is hatched.  Rows are lettered and columns
numbered, the caption sits under the plate rather than over it, hovering a
well shows its detail, and a well with a page behind it links to that page.

The map is used two ways, and the split between the functions follows them:

Embedded in a larger page
    Include :func:`plate_map_assets` once, anywhere on the page, then place
    :func:`render_plate` fragments where the maps belong.  Everything the
    fragment draws is scoped under ``.upm``, so it restyles nothing else on
    the host page.  It reads the house colour tokens (``--rule``,
    ``--text-muted``, ...) when the host defines them and falls back to the
    house values when it does not.

A page of its own
    :func:`plate_map_page` wraps fragments with the house stylesheet, a light
    and dark switch in the top right, and a cross-fade between the two.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .charts import colorbar, depth_colour

_CSS_PATH = Path(__file__).with_name("summary.css")


@dataclass
class Well:
    """One well to draw.

    Attributes:
        row: 1-based row, A = 1.
        col: 1-based column.
        depth: Reads in the well; sets the fill.
        flag: ``"mut"`` for the red corner, ``"blank"`` for hatching, or empty.
        tip: Hover card, as HTML.  Escaped here; pass it unescaped.
        href: Page the well opens, relative to the page it is drawn on.
    """

    row: int
    col: int
    depth: int = 0
    flag: str = ""
    tip: str = ""
    href: Optional[str] = None


# ---------------------------------------------------------------------------
# The fragment
# ---------------------------------------------------------------------------

_FRAGMENT_CSS = """
.upm { --upm-edge:#a8a59b; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .upm { --upm-edge:var(--rule, #43433f); }
}
:root[data-theme="dark"] .upm { --upm-edge:var(--rule, #43433f); }
.upm-plate + .upm-plate { margin-top:2rem; }
.upm-row { display:flex; gap:.7rem; align-items:stretch; }
.upm-wells { flex:0 1 27rem; min-width:0; overflow-x:auto; padding:3px; margin:-3px; }
.upm-wells.wide { flex-basis:42rem; }
/* Row letters and column numbers are the grid's own first row and column,
   so they cannot drift from the wells they name. */
.upm-grid { display:grid; gap:2px; width:100%; }
.upm-grid.c12 { grid-template-columns:1.1rem repeat(12, 1fr); }
.upm-grid.c24 { grid-template-columns:1.1rem repeat(24, 1fr); }
.upm-ax { font-size:.62rem; color:var(--text-muted, #75736c); text-align:center;
          align-self:center; line-height:1; font-variant-numeric:tabular-nums; }
.upm-ax.top { height:1rem; display:flex; align-items:flex-end; justify-content:center; }
.upm .upm-w { position:relative; display:block; aspect-ratio:1; border-radius:2px;
          background:var(--f); border:1px solid var(--upm-edge);
          outline:2px solid transparent; outline-offset:1px; }
.upm a.upm-w { cursor:pointer; }
.upm a.upm-w:hover { outline-color:var(--series-1, #2a78d6); z-index:3; }
.upm .upm-w.upm-mut::after { content:""; position:absolute; inset:0; border-radius:1px;
                     clip-path:polygon(0 0, 100% 0, 0 100%); background:#dc2626; }
.upm .upm-w.upm-blank { background:repeating-linear-gradient(45deg,
    var(--surface-2, #f3f3f0) 0 3px, var(--surface-1, #fcfcfb) 3px 6px); }
/* The ramp stands the height of the wells: it starts below the column
   numbers, and its label hangs beneath it rather than taking height from it,
   so 0 and the ceiling sit level with the first and last rows. */
.upm-scale { position:relative; padding:calc(1rem + 5px) 0 3px; display:flex; }
.upm-scale .cbar { display:flex; gap:.4rem; align-items:stretch; font-size:.68rem;
                   color:var(--text-muted, #75736c); flex:1; }
.upm-scale .ramp { width:11px; border-radius:2px; border:1px solid var(--rule, #dedcd5); }
.upm-scale .ticks { position:relative; width:2.4rem; }
.upm-scale .ticks span { position:absolute; left:0; line-height:1;
                         transform:translateY(50%); white-space:nowrap; }
.upm-scale .lab { position:absolute; top:100%; left:0; margin-top:.3rem;
                  font-size:.66rem; color:var(--text-muted, #75736c); white-space:nowrap; }
.upm-cap { font-size:.75rem; color:var(--text-muted, #75736c); margin:.9rem 0 0; }
.upm-legend { display:flex; flex-wrap:wrap; align-items:center; gap:.65rem;
              font-size:.72rem; color:var(--text-muted, #75736c); margin:.4rem 0 0; }
.upm-legend span { display:inline-flex; align-items:center; gap:.25rem; }
.upm-legend i { width:11px; height:11px; border-radius:2px; display:inline-block;
                border:1px solid var(--upm-edge); }
.upm-legend i.upm-mut { background:#dc2626; border-color:#dc2626; }
.upm-legend i.upm-blank { background:repeating-linear-gradient(45deg,
    var(--surface-2, #f3f3f0) 0 3px, var(--surface-1, #fcfcfb) 3px 6px); }
.upm-tip { position:absolute; z-index:50; max-width:22rem; padding:.45rem .6rem;
  border-radius:5px; pointer-events:none; background:#fff; color:#111;
  border:1px solid #c8c8c8; box-shadow:0 2px 8px rgba(0,0,0,.16);
  font:12px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif;
  opacity:0; visibility:hidden; transition:opacity .1s ease, visibility 0s linear .1s; }
.upm-tip.on { opacity:1; visibility:visible; transition:opacity .1s ease, visibility 0s; }
.upm .upm-w { transition:border-color .22s ease, outline-color .12s ease; }
@media (prefers-reduced-motion: reduce) {
  .upm .upm-w, .upm-tip { transition:none; }
}
"""

# Written to run once however many fragments a page holds.
_FRAGMENT_SCRIPT = """
(function () {
  if (window.__upmTip) return;
  var tip = document.createElement("div");
  tip.className = "upm-tip";
  document.body.appendChild(tip);
  window.__upmTip = tip;
  document.addEventListener("mouseover", function (e) {
    var w = e.target.closest(".upm [data-tip]");
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
})();
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def plate_map_assets() -> str:
    """The style and script a page needs once to draw any number of maps."""
    return f"<style>{_FRAGMENT_CSS}</style>\n<script>{_FRAGMENT_SCRIPT}</script>"


def _cell(w: Optional[Well]) -> str:
    if w is None:
        return f'<i class="upm-w" style="--f:{depth_colour(0)}"></i>'
    # Classes of the component's own, so a host page's .w or .mut -- the
    # summary page's included -- cannot restyle a well.
    cls = f"upm-w upm-{w.flag}" if w.flag else "upm-w"
    style = "" if w.flag == "blank" else f"--f:{depth_colour(w.depth)}"
    tip = f' data-tip="{_esc(w.tip)}"' if w.tip else ""
    if w.href:
        return (f'<a class="{cls}" href="{_esc(w.href)}" target="_blank" '
                f'rel="noopener" style="{style}"{tip}></a>')
    return f'<i class="{cls}" style="{style}"{tip}></i>'


def render_plate(wells: Iterable[Well], rows: int = 8, cols: int = 12,
                 caption: str = "", scale: bool = True) -> str:
    """One plate: labelled wells, the depth ramp beside them, a caption below.

    Args:
        wells: The wells to draw; positions not given are drawn empty.
        rows: 8 for a 96-well plate, 16 for 384.
        cols: 12 or 24.
        caption: Shown under the plate.
        scale: Draw the depth ramp.  Leave it on for one of a row of plates
            sharing a scale only if that plate is the last in the row.

    Returns:
        An HTML fragment.  Needs :func:`plate_map_assets` once on the page.
    """
    by_pos = {(w.row, w.col): w for w in wells}
    parts = ['<span class="upm-ax top"></span>']
    parts += [f'<span class="upm-ax top">{c}</span>' for c in range(1, cols + 1)]
    for r in range(1, rows + 1):
        parts.append(f'<span class="upm-ax">{chr(ord("A") + r - 1)}</span>')
        parts += [_cell(by_pos.get((r, c))) for c in range(1, cols + 1)]
    grid = f'<div class="upm-grid c{cols}">{"".join(parts)}</div>'
    ramp = (f'<div class="upm-scale">{colorbar()}<div class="lab">reads per well'
            f'</div></div>' if scale else "")
    cap = f'<p class="upm-cap">{_esc(caption)}</p>' if caption else ""
    wide = " wide" if cols > 12 else ""
    return (f'<div class="upm upm-plate"><div class="upm-row">'
            f'<div class="upm-wells{wide}">{grid}</div>{ramp}</div>{cap}</div>')


def legend(items: Sequence[tuple], note: str = "") -> str:
    """A key under the maps.

    Args:
        items: ``(flag, label)`` pairs, *flag* being ``"mut"`` or ``"blank"``.
        note: Plain text after the swatches, such as what a click opens.
    """
    spans = "".join(f'<span><i class="upm-{_esc(f)}"></i>{_esc(label)}</span>'
                    for f, label in items)
    if note:
        spans += f"<span>{_esc(note)}</span>"
    return f'<div class="upm upm-legend">{spans}</div>'


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_PAGE_CSS = """
/* A shade lighter than the summary page's dark ground: a page of plate maps
   is wells and little else, and they read better off a less deep surface. */
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-1:#252523; --surface-2:#2e2e2b; --rule:#43433f; }
}
:root[data-theme="dark"] { --surface-1:#252523; --surface-2:#2e2e2b; --rule:#43433f; }
main { max-width:46rem; }
.upm-toggle { position:fixed; top:.9rem; right:1rem; z-index:60;
  font:inherit; font-size:.75rem; padding:.25rem .6rem; cursor:pointer;
  border:1px solid var(--rule); border-radius:4px;
  background:var(--surface-2); color:var(--text-secondary); }
/* Colours cross-fade when the theme changes.  Switched on after the first
   paint, so a remembered theme does not fade in on load. */
.upm-anim body, .upm-anim main, .upm-anim .meta, .upm-anim .upm-ax,
.upm-anim .upm-cap, .upm-anim .upm-legend, .upm-anim .ramp, .upm-anim .upm-toggle {
  transition:background-color .22s ease, color .22s ease, border-color .22s ease; }
.upm-anim .upm-toggle:active { transform:scale(.96); }
@media (prefers-reduced-motion: reduce) {
  .upm-anim *, .upm-anim *::after { transition:none !important; }
}
"""

_PAGE_SCRIPT = """
(function () {
  var root = document.documentElement, btn = document.querySelector(".upm-toggle");
  function current() {
    return root.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  function label() { btn.textContent = current() === "dark" ? "Light" : "Dark"; }
  try { var saved = localStorage.getItem("usortm-theme"); if (saved) root.dataset.theme = saved; }
  catch (e) {}
  label();
  requestAnimationFrame(function () {
    requestAnimationFrame(function () { root.classList.add("upm-anim"); });
  });
  btn.addEventListener("click", function () {
    root.dataset.theme = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem("usortm-theme", root.dataset.theme); } catch (e) {}
    label();
  });
})();
"""


def plate_map_page(title: str, body: str, meta: str = "") -> str:
    """A page of its own for plate maps.

    Args:
        title: Page heading and tab title.
        body: Fragments from :func:`render_plate` and :func:`legend`.
        meta: HTML for the line under the heading; escape any user text.
    """
    css = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{css}{_PAGE_CSS}</style></head>
<body><button type="button" class="upm-toggle" aria-label="Switch light or dark">Dark</button>
<main>
  <h1>{_esc(title)}</h1>
  {meta_html}
  {body}
</main>
{plate_map_assets()}
<script>{_PAGE_SCRIPT}</script>
</body></html>
"""
