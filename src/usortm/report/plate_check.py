"""The plate map for ``usortm demux --expected``.

One map per plate as the user laid it out: a 96-well plate barcoded as a
LevSeq quadrant is drawn in its own A1-H12 coordinates, not as the scattered
quarter of a 384-well plate it occupies, and a 384-well plate as itself.
Each well is tinted by its verdict and carries a glyph for it, so the map
reads without colour; hovering gives the detail, and a table below lists
every well that is not a match.

Verdicts are states, so they wear status colours -- green for a match, amber
for a well to look at, red for one that holds the wrong thing -- rather than a
categorical palette.  The house warning orange sits too close to its red to
tell apart at a glance, so the amber here is pulled toward yellow; the
colours were checked for colour-vision separation in both themes.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from usortm.demux.expected_plate import QUADRANTS, ExpectedPlate, quadrant_to_384

_CSS_PATH = Path(__file__).with_name("summary.css")

#: verdict -> (tone, glyph, legend label).  Tone picks the status colour.
VERDICT_STYLE = {
    "match": ("good", "✓", "match"),
    "changed": ("warn", "Δ", "changed"),
    "mixed": ("warn", "≈", "mixed template"),
    "unexpected reads": ("warn", "+", "reads in a well expected empty"),
    "wrong construct": ("bad", "⇄", "wrong construct"),
    "unrecognised": ("bad", "?", "unrecognised"),
    "too few reads": ("none", "·", "too few reads"),
    "no reads": ("none", "–", "no reads"),
}

_EXTRA_CSS = """
:root { --st-good:#16996a; --st-warn:#d49a0e; --st-bad:#d1495b; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --st-good:#199e70; --st-warn:#b8860f; --st-bad:#d4506a; }
}
:root[data-theme="dark"] { --st-good:#199e70; --st-warn:#b8860f; --st-bad:#d4506a; }
main.pc { max-width:62rem; margin:0 auto; padding:1.5rem 16px 2rem; }
.pc h1 { font-size:1.35rem; margin:0 0 .2rem; }
.pc .sub { color:var(--text-muted); font-size:.82rem; margin:0 0 1.2rem;
           overflow-wrap:anywhere; }
.pc h2 { font-size:1rem; margin:1.6rem 0 .5rem; }
.tallies { display:flex; flex-wrap:wrap; gap:.4rem .9rem; margin:.2rem 0 1rem;
           font-size:.82rem; color:var(--text-secondary); }
.tallies .ls b { color:var(--text-primary); font-variant-numeric:tabular-nums; }
.maps { display:flex; flex-wrap:wrap; gap:1.4rem 2rem; }
/* The basis gives way to the screen, or a phone scrolls sideways. */
.pmap { flex:1 1 min(26rem, 100%); min-width:0; max-width:40rem; }
.pmap.big { max-width:none; flex-basis:100%; }
.pmap h3 { font-size:.85rem; font-weight:600; margin:0 0 .35rem;
           color:var(--text-secondary); }
.pgrid { display:grid; gap:2px; align-items:center; }
.pgrid .ax { font-size:.66rem; color:var(--text-muted); text-align:center;
             font-variant-numeric:tabular-nums; }
.c { position:relative; aspect-ratio:1; border-radius:3px; min-width:0;
     display:flex; align-items:center; justify-content:center;
     font-size:.72rem; line-height:1; color:var(--text-primary);
     background:var(--surface-2); border:1px solid var(--rule); }
.big .c { font-size:.55rem; border-radius:2px; }
.c.good { background:color-mix(in srgb, var(--st-good) 26%, var(--surface-1));
          border-color:var(--st-good); }
.c.warn { background:color-mix(in srgb, var(--st-warn) 30%, var(--surface-1));
          border-color:var(--st-warn); }
.c.bad  { background:color-mix(in srgb, var(--st-bad) 30%, var(--surface-1));
          border-color:var(--st-bad); box-shadow:inset 0 0 0 1px var(--st-bad); }
.c.none { border-style:dashed; color:var(--text-muted); }
.c.empty { background:repeating-linear-gradient(45deg,
             var(--surface-2) 0 3px, var(--surface-1) 3px 6px); }
.c.unlisted { background:transparent; border-color:var(--grid); }
.c[data-tip]:hover { outline:2px solid var(--series-1); outline-offset:1px; z-index:3; }
.c .g { font-weight:700; }
.c .n { display:none; }
/* Room for a name only on a 96-well map at full width. */
@media (min-width: 720px) {
  .pmap:not(.big) .c { flex-direction:column; gap:1px; }
  .pmap:not(.big) .c .n { display:block; font-size:.5rem; color:var(--text-secondary);
                          max-width:100%; overflow:hidden; white-space:nowrap;
                          text-overflow:ellipsis; padding:0 1px; }
}
.ls i.sw { width:13px; height:13px; border-radius:3px; display:inline-flex;
           align-items:center; justify-content:center; font-size:.6rem;
           font-style:normal; font-weight:700; color:var(--text-primary); }
table.wells { border-collapse:collapse; width:100%; font-size:.8rem; }
table.wells th, table.wells td { text-align:left; padding:.3rem .5rem;
  border-bottom:1px solid var(--rule); vertical-align:top; }
table.wells th { color:var(--text-muted); font-weight:600; }
table.wells td.r { text-align:right; font-variant-numeric:tabular-nums; }
table.wells code { font-family:SF Mono,Menlo,Consolas,monospace; font-size:.75rem;
                   overflow-wrap:anywhere; }
.tablewrap { overflow-x:auto; }
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


def _tip(title: str, v, expected_name: str, state: str) -> str:
    """Hover card for one well, as escaped HTML for a data attribute."""
    lines = [f'<div style="font-size:13px;"><b>{_esc(title)}</b></div>']
    if v is None:
        lines.append('<div style="margin-top:4px;">'
                     + ("expected empty; no reads" if state == "empty"
                        else "not in the expected plate; no reads") + "</div>")
        return _esc("".join(lines))
    tone, glyph, label = VERDICT_STYLE.get(v.verdict, ("none", "", v.verdict))
    lines.append(f'<div style="margin-top:4px;">{_esc(glyph)} {_esc(label)}</div>')
    small = 'style="font-size:11px;color:#555;margin-top:2px;"'
    lines.append(f'<div {small}>Expected: {_esc(expected_name or "empty")}</div>')
    if v.observed:
        lines.append(f'<div {small}>Holds: {_esc(v.observed)}</div>')
    lines.append(f'<div {small}>Reads: {v.reads:,}</div>')
    if v.differences:
        lines.append(f'<div {small}>Changes: {_esc(" ".join(v.differences[:8]))}'
                     + (" …" if len(v.differences) > 8 else "") + "</div>")
    if v.protein_changes:
        lines.append(f'<div {small}>Protein: {_esc(" ".join(v.protein_changes[:8]))}</div>')
    if v.disagreement is not None:
        lines.append(f'<div {small}>{v.disagreement:.0%} of reads disagree at the '
                     "worst insert position</div>")
    if v.note:
        lines.append(f'<div {small}>{_esc(v.note)}</div>')
    return _esc("".join(lines))


def _cell(v, expected, title: str, state: str) -> str:
    """One well: tint and glyph by verdict, a name where there is room."""
    if v is None:
        cls = "empty" if state == "empty" else "unlisted"
        return f'<i class="c {cls}" data-tip="{_tip(title, None, "", state)}"></i>'
    tone, glyph, _ = VERDICT_STYLE.get(v.verdict, ("none", "", ""))
    # The map is the plate as designed, so a cell names what it should hold
    # and the glyph says whether it does; what it holds instead is on hover.
    # A well expected empty has no design to name, so it names its contents.
    if expected is not None and not expected.empty:
        name = expected.name
    else:
        name = v.observed
    return (f'<i class="c {tone}" data-tip="{_tip(title, v, v.expected, state)}">'
            f'<span class="g">{_esc(glyph)}</span>'
            f'<span class="n">{_esc(name)}</span></i>')


def _grid(rows: int, cols: int, cell_for) -> str:
    """A plate grid with row letters and column numbers."""
    parts = [f'<div class="pgrid" style="grid-template-columns:1.1rem '
             f'repeat({cols}, minmax(0,1fr));"><span></span>']
    parts += [f'<span class="ax">{c}</span>' for c in range(1, cols + 1)]
    for r in range(rows):
        letter = chr(ord("A") + r)
        parts.append(f'<span class="ax">{letter}</span>')
        parts += [cell_for(letter, c) for c in range(1, cols + 1)]
    parts.append("</div>")
    return "".join(parts)


def plate_maps(plate: ExpectedPlate, verdicts: Sequence) -> str:
    """The maps, one per plate or quadrant as the expected plate was laid out."""
    by_key = {(v.plate, v.well): v for v in verdicts}
    maps = []
    if plate.layout == "96":
        # Quadrants the CSV uses, plus any holding reads it did not expect.
        used = {(w.plate, quadrant_of_384(w.well)[0]) for w in plate.wells.values()}
        used |= {(v.plate, quadrant_of_384(v.well)[0]) for v in verdicts}
        for p, q in sorted(used):
            rb = (p - 1) * 4 + q + 1

            def cell(letter, col, p=p, q=q):
                well = quadrant_to_384(q, ord(letter) - ord("A") + 1, col)
                exp = plate.wells.get((p, well))
                state = "empty" if exp is not None and exp.empty else (
                    "listed" if exp is not None else "unlisted")
                title = f"{QUADRANTS[q]} {letter}{col}  ·  plate {p} {well}"
                return _cell(by_key.get((p, well)), exp, title, state)

            maps.append(f'<section class="pmap"><h3>Plate {p} · {QUADRANTS[q]} '
                        f'quadrant (RB{rb:02d})</h3>{_grid(8, 12, cell)}</section>')
    else:
        plates = sorted({w.plate for w in plate.wells.values()}
                        | {v.plate for v in verdicts})
        for p in plates:
            def cell(letter, col, p=p):
                well = f"{letter}{col}"
                exp = plate.wells.get((p, well))
                state = "empty" if exp is not None and exp.empty else (
                    "listed" if exp is not None else "unlisted")
                return _cell(by_key.get((p, well)), exp, f"Plate {p} {well}", state)

            maps.append(f'<section class="pmap big"><h3>Plate {p}</h3>'
                        f'{_grid(16, 24, cell)}</section>')
    return f'<div class="maps">{"".join(maps)}</div>'


def _legend(counts: dict) -> str:
    items = []
    for verdict, (tone, glyph, label) in VERDICT_STYLE.items():
        n = counts.get(verdict, 0)
        if not n:
            continue
        items.append(f'<span class="ls"><i class="sw c {tone}">{_esc(glyph)}</i>'
                     f'{_esc(label)} <b>{n:,}</b></span>')
    items.append('<span class="ls"><i class="sw c empty"></i>expected empty, no reads</span>')
    return f'<div class="tallies">{"".join(items)}</div>'


def _table(plate: ExpectedPlate, verdicts: Sequence) -> str:
    flagged = [v for v in verdicts if v.verdict != "match"]
    if not flagged:
        return "<p>Every well holds the construct expected of it.</p>"
    rows = []
    for v in flagged:
        tone, glyph, label = VERDICT_STYLE.get(v.verdict, ("none", "", v.verdict))
        where = v.label if plate.layout == "96" else v.well
        detail = " ".join(v.differences)
        if v.protein_changes:
            detail += f" ({' '.join(v.protein_changes)})"
        rows.append(
            f"<tr><td>{v.plate}:{_esc(where)}</td><td>{_esc(v.expected or '—')}</td>"
            f'<td><i class="sw c {tone}" style="display:inline-flex;width:13px;'
            f'height:13px;margin-right:.3rem;vertical-align:-2px;">{_esc(glyph)}</i>'
            f"{_esc(label)}</td><td>{_esc(v.observed)}</td>"
            f'<td class="r">{v.reads:,}</td>'
            f"<td><code>{_esc(detail)}</code> {_esc(v.note)}</td></tr>")
    return ('<div class="tablewrap"><table class="wells"><thead><tr><th>Well</th>'
            "<th>Expected</th><th>Verdict</th><th>Holds</th><th>Reads</th>"
            f"<th>Detail</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def write_plate_check_page(
    plate: ExpectedPlate,
    verdicts: Sequence,
    path,
    counts: dict,
    fastq: Sequence = (),
    layout_lines: Optional[Sequence[str]] = None,
    stray_wells: int = 0,
    min_reads: int = 10,
) -> Path:
    """Write the plate check page: maps, legend, and the wells to look at."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    css = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""
    n_expected = sum(1 for w in plate.wells.values() if not w.empty)
    reads_from = ", ".join(Path(f).name for f in fastq) or "—"
    sub = (f"{_esc(Path(plate.source).name)} · {n_expected} wells expected · "
           f"reads from {_esc(reads_from)} · "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    layout_html = ""
    if layout_lines:
        layout_html = ("<h2>Read layout</h2><ul style=\"font-size:.82rem;"
                       "color:var(--text-secondary);margin:0;padding-left:1.1rem;\">"
                       + "".join(f"<li>{_esc(line)}</li>" for line in layout_lines)
                       + "</ul>")
    stray = ""
    if stray_wells:
        stray = (f'<p style="font-size:.8rem;color:var(--text-muted);">'
                 f"{stray_wells} well(s) expected empty or not listed carry 1–"
                 f"{min_reads - 1} reads, below the call.</p>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plate check</title>
<style>{css}{_EXTRA_CSS}</style></head>
<body><main class="pc">
<h1>Plate check</h1>
<p class="sub">{sub}</p>
{_legend(counts)}
{plate_maps(plate, verdicts)}
{stray}
<h2>Wells to look at</h2>
{_table(plate, verdicts)}
{layout_html}
<footer>Wells are called at {min_reads} reads or more. Each is checked against
the construct expected of it; see <code>verification.csv</code> for every well.</footer>
</main>
<script>{_SCRIPT}</script>
</body></html>
"""
    path.write_text(page)
    return path
