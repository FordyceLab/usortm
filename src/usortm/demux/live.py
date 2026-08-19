"""A dashboard that fills in while demultiplexing runs.

A demux run over a full flow cell takes hours, and until it finishes the only
signal is a spinner. The figures that say whether it is going well — what
fraction of reads aligned, what fraction carried a barcode, how the wells are
filling — are known long before the end.

This writes them as they are established. Two files land beside the run:

``live.html``
    Written once at the start. Re-reads the data file on a timer and redraws
    in place, so it can be left open.

``live_data.js``
    Rewritten at each stage. A ``.js`` file rather than JSON because browsers
    block ``fetch()`` from a ``file://`` page, while a script tag loads a
    sibling file without complaint.

Nothing here affects the run: a failure to write the dashboard is logged and
stepped over.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_FILE = "live_data.js"
PAGE_FILE = "live.html"
POLL_SECONDS = 4

# Stages in the order the pipeline runs them, with the label shown.
STAGES = [
    ("deps", "Checking dependencies"),
    ("config", "Generating barcode config"),
    ("hist", "Reading input"),
    ("align", "Aligning and splitting by strand"),
    ("fbc", "Forward barcode demux"),
    ("rbc", "Reverse barcode demux"),
    ("readdf", "Assembling read table"),
    ("wells", "Mapping barcodes to wells"),
    ("consensus", "Per-well consensus"),
    ("variants", "Calling variants"),
    ("streakout", "Screening streak-out candidates"),
    ("done", "Complete"),
]


class LiveReport:
    """Collects run figures and writes them out as they are established.

    Args:
        output_dir: Directory the run writes into.
        label: Segment name, shown when a run spans several FASTQs.
    """

    def __init__(self, output_dir, label: str = ""):
        self.dir = Path(output_dir)
        self.label = label
        self.started = time.time()
        self.stage = "deps"
        self.data: dict = {}
        self.enabled = True
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / PAGE_FILE).write_text(_PAGE)
            self.write()
        except OSError as exc:
            logger.warning("Live dashboard disabled: %s", exc)
            self.enabled = False

    @property
    def page(self) -> Path:
        return self.dir / PAGE_FILE

    def begin_segment(self, label: str) -> None:
        """Start reporting on another FASTQ, keeping the finished ones listed.

        A run spanning several FASTQs reports to one page; each segment's
        figures are kept as it completes so the whole run stays visible.
        """
        if self.label:
            self.data.setdefault("finished", []).append(
                {"label": self.label,
                 **{k: self.data.get(k) for k in
                    ("input_reads", "aligned", "fbc", "rbc", "wells")}}
            )
        self.label = label
        for key in ("input_reads", "aligned", "fbc", "rbc", "wells", "plates",
                    "warning"):
            self.data.pop(key, None)
        self.stage = "deps"
        self.write()

    def set_stage(self, stage: str) -> None:
        """Record which stage is running and flush."""
        self.stage = stage
        self.write()

    def update(self, **fields) -> None:
        """Record figures as they become known and flush."""
        self.data.update({k: v for k, v in fields.items() if v is not None})
        self.write()

    def write(self) -> None:
        """Write the data file. Never raises: the run matters, this does not."""
        if not self.enabled:
            return
        payload = {
            "label": self.label,
            "stage": self.stage,
            "stages": [{"key": k, "name": n} for k, n in STAGES],
            "startedEpoch": self.started,
            "writtenEpoch": time.time(),
            "updated": time.strftime("%H:%M:%S"),
            **self.data,
        }
        try:
            (self.dir / DATA_FILE).write_text(
                "window.USORTM_LIVE = " + json.dumps(payload) + ";"
            )
        except OSError as exc:
            logger.debug("Could not write live data: %s", exc)
            self.enabled = False


_PAGE = """<title>uSort-M demux — live</title>
<style>
  :root {
    color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f3f3f0; --rule:#dedcd5;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#75736c;
    --series-1:#2a78d6; --good:#1baf7a; --warn:#eb6834;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#232322; --rule:#3a3a37;
      --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96948a;
      --series-1:#3987e5; --good:#199e70; --warn:#d95926;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#232322; --rule:#3a3a37;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#96948a;
    --series-1:#3987e5; --good:#199e70; --warn:#d95926;
  }
  body { margin:0; background:var(--surface-1); color:var(--text-primary);
         font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:940px; margin:0 auto; padding:2rem; }
  h1 { font-size:1.15rem; margin:0 0 .2rem; font-weight:600; }
  .sub { color:var(--text-secondary); font-size:.9rem; margin:0 0 1.5rem; }
  .stats { display:flex; flex-wrap:wrap; gap:1.75rem; margin-bottom:1.75rem; }
  .k { font-size:.72rem; text-transform:uppercase; letter-spacing:.04em;
       color:var(--text-muted); }
  .v { font-size:1.3rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .u { font-size:.8rem; color:var(--text-secondary); font-weight:400; }
  ol { list-style:none; padding:0; margin:0 0 1.75rem; }
  li { display:flex; align-items:center; gap:.6rem; padding:.3rem 0;
       color:var(--text-muted); }
  li .dot { width:9px; height:9px; border-radius:50%; background:var(--rule);
            flex:none; }
  li.done { color:var(--text-secondary); }
  li.done .dot { background:var(--good); }
  li.now { color:var(--text-primary); font-weight:600; }
  li.now .dot { background:var(--series-1);
                animation:pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  h2 { font-size:.95rem; margin:1.75rem 0 .6rem; font-weight:600; }
  table { border-collapse:collapse; font-size:.85rem; }
  th,td { text-align:right; padding:.28rem .8rem;
          border-bottom:1px solid var(--rule);
          font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child { text-align:left; }
  .bar { height:7px; background:var(--rule); border-radius:4px; overflow:hidden;
         min-width:110px; }
  .bar span { display:block; height:100%; background:var(--series-1); }
  .foot { color:var(--text-muted); font-size:.8rem; margin-top:2rem; }
  .warnbox { border-left:3px solid var(--warn); padding:.5rem .9rem;
             background:var(--surface-2); margin:1rem 0; font-size:.87rem; }
</style>
<main>
  <h1>Demultiplexing <span id="label"></span></h1>
  <p class="sub" id="sub">waiting for the run to report…</p>
  <div class="stats" id="stats"></div>
  <div id="warn"></div>
  <ol id="stages"></ol>
  <div id="plates"></div>
  <div id="finished"></div>
  <p class="foot" id="foot"></p>
</main>
<script>
const fmt = n => (n === undefined || n === null) ? "\\u2014" : n.toLocaleString();
const pct = (a, b) => (b ? (100 * a / b).toFixed(1) + "%" : "\\u2014");

function render() {
  const D = window.USORTM_LIVE;
  if (!D) return;
  document.getElementById("label").textContent = D.label ? "\\u2014 " + D.label : "";
  tick();   // elapsed and data age tick every second, not per reload

  const cards = [["Input reads", fmt(D.input_reads), ""]];
  if (D.aligned !== undefined)
    cards.push(["Aligned", fmt(D.aligned), pct(D.aligned, D.input_reads)]);
  if (D.fbc !== undefined)
    cards.push(["Forward barcode", fmt(D.fbc), pct(D.fbc, D.aligned)]);
  if (D.rbc !== undefined)
    cards.push(["Reverse barcode", fmt(D.rbc), pct(D.rbc, D.aligned)]);
  if (D.wells !== undefined) cards.push(["Wells with data", fmt(D.wells), ""]);
  if (D.variants !== undefined) cards.push(["Variants seen", fmt(D.variants), ""]);
  document.getElementById("stats").innerHTML = cards.map(([k, v, u]) =>
    `<div><div class="k">${k}</div><div class="v">${v}<span class="u"> ${u}</span></div></div>`
  ).join("");

  document.getElementById("warn").innerHTML = D.warning
    ? `<div class="warnbox"><b>${D.warning.headline}</b><br>${D.warning.detail}</div>` : "";

  const at = D.stages.findIndex(s => s.key === D.stage);
  document.getElementById("stages").innerHTML = D.stages.map((s, i) =>
    `<li class="${i < at ? "done" : i === at ? "now" : ""}">`
    + `<span class="dot"></span>${s.name}</li>`).join("");

  if (D.plates && Object.keys(D.plates).length) {
    const rows = Object.entries(D.plates).sort((a, b) => a[0] - b[0]);
    const top = Math.max(...rows.map(r => r[1]));
    document.getElementById("plates").innerHTML =
      "<h2>Wells per plate</h2><table><tr><th>Plate</th><th>Wells</th><th></th></tr>"
      + rows.map(([p, n]) =>
          `<tr><td>${p}</td><td>${fmt(n)}</td><td><div class="bar">`
          + `<span style="width:${(100 * n / top).toFixed(1)}%"></span></div></td></tr>`
        ).join("") + "</table>";
  }
  const fin = D.finished || [];
  document.getElementById("finished").innerHTML = fin.length
    ? "<h2>Completed FASTQs</h2><table><tr><th>FASTQ</th><th>Reads</th>"
      + "<th>Aligned</th><th>Wells</th></tr>"
      + fin.map(f => `<tr><td>${f.label}</td><td>${fmt(f.input_reads)}</td>`
          + `<td>${fmt(f.aligned)}</td><td>${fmt(f.wells)}</td></tr>`).join("")
      + "</table>"
    : "";

  document.getElementById("foot").textContent =
    D.stage === "done" ? "Run complete." : `Refreshing every ${POLL}s.`;
}

function tick() {
  const D = window.USORTM_LIVE;
  if (!D) return;
  const el = Math.max(0, Date.now() / 1000 - D.startedEpoch);
  const h = Math.floor(el / 3600), m = Math.floor((el % 3600) / 60),
        s = Math.floor(el % 60);
  const run = (h ? h + "h " : "") + m + "m " + String(s).padStart(2, "0") + "s";
  const age = Math.max(0, Math.round(Date.now() / 1000 - D.writtenEpoch));
  const state = D.stage === "done" ? "finished"
              : age > POLL * 4 ? `no update for ${age}s` : `updated ${age}s ago`;
  document.getElementById("sub").textContent = `running ${run} \u00b7 ${state}`;
}
setInterval(tick, 1000);

const POLL = __POLL__;
function reload() {
  const s = document.createElement("script");
  s.src = "live_data.js?t=" + Date.now();
  s.onload = () => { render(); s.remove(); };
  s.onerror = () => s.remove();
  document.head.appendChild(s);
}
reload();
setInterval(reload, POLL * 1000);
</script>
""".replace("__POLL__", str(POLL_SECONDS))
