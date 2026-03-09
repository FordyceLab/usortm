"""Streak-out candidate detection and per-well pileup visualization.

Identifies wells containing two or more correctly-assembled subpopulations
of reads.  When the minority variant was missed during initial sampling,
users can streak out those wells to isolate it.

Algorithm
---------
For each well where the dominant fraction is below a threshold (default 0.9)
and read depth is sufficient (default ≥50):

1. Group reads by reference (strip strand prefix).
2. Drop groups with fewer than *min_group_reads* reads.
3. For each surviving group, generate a consensus (minimap2 → samtools
   consensus → re-align) and classify the CIGAR as Perfect Match, Silent
   Mutation, or other.
4. If **2+ groups** have a correct consensus (Perfect Match or Silent
   Mutation), the well is a streak-out candidate.  The minority group(s)
   are the recoverable variants.

Outputs
-------
- ``streakout_candidates.csv`` — one row per candidate well
- ``well_{plate}_{well}.html`` — per-well pileup visualization
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pysam
from Bio.Seq import Seq
from Bio import SeqIO

from usortm.demux.deps import find_minimap2, find_samtools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CIGAR classification (mirrors utils.extract_matches logic)
# ---------------------------------------------------------------------------

def _classify_cigar(cigar: Optional[str], ref_len: int,
                    ref_seq: str, cons_seq: Optional[str]) -> str:
    """Classify a consensus CIGAR string as a match status.

    Expects CIGAR produced with minimap2 ``--eqx`` so that ``=`` denotes
    sequence match and ``X`` denotes mismatch.  Falls back to sequence
    comparison when legacy ``M`` operators are present.

    Returns one of: "Perfect Match", "Silent Mutation", "Partial Match",
    "Other Error", or "Error".
    """
    import re as _re

    if cigar is None or cons_seq is None:
        return "Error"

    # Parse CIGAR into (length, op) pairs
    ops = _re.findall(r'(\d+)([A-Z=])', cigar)
    if not ops:
        return "Error"

    op_letters = set(op for _, op in ops)
    total_ref = sum(int(n) for n, op in ops if op in ('M', '=', 'X', 'D', 'N'))

    # --eqx mode: '=' and 'X' operators
    if '=' in op_letters or 'X' in op_letters:
        has_mismatch = 'X' in op_letters
        has_indel = bool(op_letters & {'I', 'D', 'N'})
        if not has_mismatch and not has_indel and total_ref == ref_len:
            return "Perfect Match"
        # Full-length alignment with only substitutions — check silent
        if not has_indel and total_ref == ref_len and len(cons_seq) == ref_len:
            try:
                if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                    return "Silent Mutation"
            except Exception:
                pass
        if total_ref == ref_len:
            return "Other Error"
        return "Partial Match"

    # Legacy M-only CIGAR: compare sequences directly
    if op_letters == {'M'} and total_ref == ref_len:
        if cons_seq.upper() == ref_seq.upper():
            return "Perfect Match"
        # Check for silent mutation
        if len(cons_seq) == ref_len:
            try:
                if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                    return "Silent Mutation"
            except Exception:
                pass
        return "Other Error"

    if total_ref < ref_len:
        return "Partial Match"
    return "Other Error"


def _is_correct(status: str) -> bool:
    return status in ("Perfect Match", "Silent Mutation")


# ---------------------------------------------------------------------------
# Per-group consensus (reuses _process_single_well pattern)
# ---------------------------------------------------------------------------

def _group_consensus(reads_fastq: str, ref_fasta: str, work_dir: str,
                     minimap2_path: str, samtools_path: str,
                     ) -> tuple[Optional[str], Optional[str]]:
    """Align reads → consensus → re-align → extract CIGAR for one group.

    Returns (cigar_str, cons_seq) or (None, None) on failure.
    """
    bam = os.path.join(work_dir, "aligned.bam")
    cons_fa = os.path.join(work_dir, "consensus.fasta")
    cons_bam = os.path.join(work_dir, "consensus.bam")

    # 1) Align reads to reference
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", ref_fasta, reads_fastq],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", bam],
            stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
        )
        mm2.wait()
    except Exception:
        return None, None

    # 2) Generate consensus
    try:
        with open(cons_fa, "w") as fh:
            subprocess.run(
                [samtools_path, "consensus", "-f", "fasta", bam],
                stdout=fh, check=False,
            )
    except Exception:
        return None, None

    # 3) Re-align consensus to reference (--eqx for =/X CIGAR ops)
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", "--eqx", ref_fasta, cons_fa],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [samtools_path, "sort", "-o", cons_bam],
            stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
        )
        mm2.wait()
    except Exception:
        return None, None

    # 4) Extract CIGAR + consensus sequence
    cigar_str, cons_seq = None, None
    try:
        with pysam.AlignmentFile(cons_bam, "rb") as bf:
            for read in bf:
                if not read.is_unmapped:
                    cigar_str = read.cigarstring
                    break
        if os.path.exists(cons_fa):
            with open(cons_fa) as fh:
                lines = fh.read().splitlines()
                cons_seq = "".join(l for l in lines if not l.startswith(">"))
    except Exception:
        pass

    return cigar_str, cons_seq


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _process_well_for_streakout(
    well_pos: str,
    well_reads: pd.DataFrame,
    well_row: pd.Series,
    reference_dir: str,
    minimap2_path: str,
    samtools_path: str,
    min_group_reads: int,
) -> Optional[dict]:
    """Analyse a single well for streak-out candidacy."""
    plate = str(int(well_row["plate"]))
    well = str(well_row["well"])
    depth = int(well_row["depth"])
    top_frac = float(well_row["major_freq"])

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")

    # Group reads by reference (strip strand prefix)
    rdf = well_reads.copy()
    rdf["_ref"] = rdf["ref_name"].str.replace(r'^(fwd|rev):', '', regex=True)
    groups = rdf.groupby("_ref")

    group_results = []
    for ref_id, grp in groups:
        if len(grp) < min_group_reads:
            continue

        ref_fasta = os.path.join(single_ref_dir, f"{ref_id}.fasta")
        if not os.path.exists(ref_fasta):
            continue

        # Read reference sequence + length
        ref_record = next(SeqIO.parse(ref_fasta, "fasta"))
        ref_seq = str(ref_record.seq)
        ref_len = len(ref_seq)

        # Write group reads to temp FASTQ and run consensus pipeline
        with tempfile.TemporaryDirectory() as tmp:
            fq_path = os.path.join(tmp, "group.fastq")
            with open(fq_path, "w") as fq:
                for _, r in grp.iterrows():
                    fq.write(f"@{r['read_name']}\n{r['read_seq']}\n+\n{r['read_qual']}\n")

            cigar, cons_seq = _group_consensus(
                fq_path, ref_fasta, tmp, minimap2_path, samtools_path,
            )

        status = _classify_cigar(cigar, ref_len, ref_seq, cons_seq)
        n_reads = len(grp)
        frac = n_reads / depth if depth else 0

        group_results.append({
            "variant": ref_id,
            "reads": n_reads,
            "frac": round(frac, 4),
            "status": status,
            "is_major": (frac >= top_frac - 0.01),
        })

    # Check: 2+ groups with correct consensus?
    correct_groups = [g for g in group_results if _is_correct(g["status"])]
    if len(correct_groups) < 2:
        return None

    # Identify recoverable (minority) variants
    recoverable = [
        g["variant"] for g in correct_groups if not g["is_major"]
    ]

    return {
        "plate": plate,
        "well": well,
        "global_well": well_pos,
        "total_reads": depth,
        "top_frac": round(top_frac, 4),
        "groups": sorted(group_results, key=lambda g: -g["frac"]),
        "recoverable_variants": recoverable,
    }


def detect_streakout_candidates(
    well_df: pd.DataFrame,
    read_df: pd.DataFrame,
    reference_dir: str,
    output_dir: str,
    minimap2_path: str = None,
    samtools_path: str = None,
    min_well_reads: int = 50,
    min_group_reads: int = 5,
    max_top_frac: float = 0.9,
    workers: int = 4,
) -> list[dict]:
    """Detect wells with multiple correctly-assembled subpopulations.

    For each well where ``major_freq < max_top_frac`` and
    ``depth >= min_well_reads``, groups reads by reference, generates a
    per-group consensus, and checks whether 2+ groups produce a correct
    consensus (Perfect Match or Silent Mutation).

    Args:
        well_df: Per-well summary DataFrame (from ``generate_well_df``).
        read_df: Per-read DataFrame (from ``format_df``).
        reference_dir: Directory containing ``single_ref_fastas/`` subdirectory.
        output_dir: Pipeline output directory (unused in detection, reserved).
        minimap2_path: Path to minimap2 binary. Auto-detected if None.
        samtools_path: Path to samtools binary. Auto-detected if None.
        min_well_reads: Minimum total reads in a well to consider.
        min_group_reads: Minimum reads in a group to attempt consensus.
        max_top_frac: Maximum dominant fraction to flag as potential mixed well.
        workers: Number of parallel workers.

    Returns:
        List of candidate dicts, one per streak-out well.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    # Filter candidate wells
    mask = (well_df["major_freq"] < max_top_frac) & (well_df["depth"] >= min_well_reads)
    candidate_wells = well_df[mask]

    if candidate_wells.empty:
        return []

    logger.info(
        "Screening %d wells for streak-out candidates (frac < %.2f, depth >= %d)",
        len(candidate_wells), max_top_frac, min_well_reads,
    )

    candidates = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for _, row in candidate_wells.iterrows():
            wp = row["global_well"]
            well_reads = read_df[read_df["well_pos"] == wp]
            if well_reads.empty:
                continue
            fut = pool.submit(
                _process_well_for_streakout,
                wp, well_reads, row, reference_dir,
                minimap2_path, samtools_path, min_group_reads,
            )
            futures[fut] = wp

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result is not None:
                    candidates.append(result)
            except Exception as exc:
                logger.warning("Streakout analysis failed for %s: %s",
                               futures[fut], exc)

    candidates.sort(key=lambda c: c["global_well"])
    return candidates


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def save_streakout_results(candidates: list[dict], output_dir: str) -> dict:
    """Write streak-out candidates to CSV and return summary dict.

    Writes ``streakout_candidates.csv`` to *output_dir*.

    Returns:
        Summary dict suitable for inclusion in ``demux_summary.json``.
    """
    csv_path = os.path.join(output_dir, "streakout_candidates.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plate", "well", "total_reads", "top_frac",
            "n_groups", "recoverable_variants",
        ])
        for c in candidates:
            writer.writerow([
                c["plate"],
                c["well"],
                c["total_reads"],
                c["top_frac"],
                len(c["groups"]),
                ";".join(c["recoverable_variants"]),
            ])

    return {
        "candidates": len(candidates),
        "recoverable_variants": list({
            v for c in candidates for v in c["recoverable_variants"]
        }),
        "csv_path": csv_path,
    }


# ---------------------------------------------------------------------------
# Per-well pileup HTML
# ---------------------------------------------------------------------------

def generate_well_pileup_html(
    well_pos: str,
    read_df: pd.DataFrame,
    reference_dir: str,
    candidate_info: dict,
    output_path: str,
    minimap2_path: str = None,
    samtools_path: str = None,
) -> None:
    """Generate an interactive pileup HTML for one streak-out candidate well.

    Aligns reads from each reference group to their respective reference,
    builds a character grid showing match/mismatch coloring, and writes
    a self-contained HTML file.

    Args:
        well_pos: Global well identifier (e.g. "1A3").
        read_df: Full per-read DataFrame.
        reference_dir: Directory containing ``single_ref_fastas/``.
        candidate_info: Candidate dict from :func:`detect_streakout_candidates`.
        output_path: Path to write the HTML file.
        minimap2_path: Path to minimap2. Auto-detected if None.
        samtools_path: Path to samtools. Auto-detected if None.
    """
    if minimap2_path is None:
        minimap2_path = find_minimap2()
    if samtools_path is None:
        samtools_path = find_samtools()

    single_ref_dir = os.path.join(reference_dir, "single_ref_fastas")

    # Filter reads for this well
    well_reads = read_df[read_df["well_pos"] == well_pos].copy()
    well_reads["_ref"] = well_reads["ref_name"].str.replace(
        r'^(fwd|rev):', '', regex=True,
    )

    # Build group info lookup from candidate_info
    group_lookup = {g["variant"]: g for g in candidate_info["groups"]}

    group_sections = []
    for ref_id, grp in well_reads.groupby("_ref"):
        ref_fasta = os.path.join(single_ref_dir, f"{ref_id}.fasta")
        if not os.path.exists(ref_fasta):
            continue

        ref_record = next(SeqIO.parse(ref_fasta, "fasta"))
        ref_seq = str(ref_record.seq)
        ref_len = len(ref_seq)

        ginfo = group_lookup.get(ref_id, {})
        n_reads = len(grp)
        frac = ginfo.get("frac", n_reads / candidate_info["total_reads"])
        status = ginfo.get("status", "")
        is_recoverable = ref_id in candidate_info["recoverable_variants"]

        # Align to reference and parse BAM for pileup
        pileup_rows = _build_pileup_grid(
            grp, ref_fasta, ref_seq, ref_len,
            minimap2_path, samtools_path,
        )

        group_sections.append({
            "ref_id": ref_id,
            "n_reads": n_reads,
            "frac": frac,
            "status": status,
            "is_recoverable": is_recoverable,
            "ref_seq": ref_seq,
            "pileup_rows": pileup_rows,
        })

    # Sort: major group first
    group_sections.sort(key=lambda s: -s["frac"])

    html = _render_pileup_html(well_pos, candidate_info, group_sections)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def _build_pileup_grid(
    group_reads: pd.DataFrame,
    ref_fasta: str,
    ref_seq: str,
    ref_len: int,
    minimap2_path: str,
    samtools_path: str,
) -> list[list[tuple[str, bool]]]:
    """Align group reads and build a character grid for pileup display.

    Returns a list of rows, where each row is a list of
    (base_char, is_match) tuples indexed by reference position.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fq_path = os.path.join(tmp, "reads.fastq")
        bam_path = os.path.join(tmp, "aligned.bam")

        # Write FASTQ
        with open(fq_path, "w") as fq:
            for _, r in group_reads.iterrows():
                fq.write(f"@{r['read_name']}\n{r['read_seq']}\n+\n{r['read_qual']}\n")

        # Align
        try:
            mm2 = subprocess.Popen(
                [minimap2_path, "-a", "--MD", ref_fasta, fq_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [samtools_path, "sort", "-o", bam_path],
                stdin=mm2.stdout, stderr=subprocess.DEVNULL, check=False,
            )
            mm2.wait()
            subprocess.run(
                [samtools_path, "index", bam_path],
                stderr=subprocess.DEVNULL, check=False,
            )
        except Exception as exc:
            logger.warning("Pileup alignment failed: %s", exc)
            return []

        # Parse BAM for pileup
        rows = []
        try:
            with pysam.AlignmentFile(bam_path, "rb") as bf:
                for read in bf:
                    if read.is_unmapped:
                        continue
                    row = [("-", True)] * ref_len  # default: gap
                    pairs = read.get_aligned_pairs(with_seq=True)
                    for qpos, rpos, rbase in pairs:
                        if rpos is None or rpos >= ref_len:
                            continue
                        if qpos is None:
                            row[rpos] = ("-", True)  # deletion
                        else:
                            qbase = read.query_sequence[qpos]
                            is_match = qbase.upper() == ref_seq[rpos].upper()
                            row[rpos] = (qbase, is_match)
                    rows.append(row)
        except Exception as exc:
            logger.warning("BAM parsing failed: %s", exc)

        if not rows:
            logger.warning(
                "Pileup produced 0 aligned rows for %d reads against %s",
                len(group_reads), ref_fasta,
            )

    return rows


def _render_pileup_html(well_pos: str, candidate: dict,
                         groups: list[dict]) -> str:
    """Render the pileup HTML page for one well.

    Uses an HTML5 canvas matrix: each read is a row of colored cells.
    Green = match, per-base color = mismatch, light gray = gap.
    """
    import html as _html
    import json as _json

    plate = candidate["plate"]
    well = candidate["well"]
    title = f"Pileup: Plate {plate} Well {well}"

    sections_html = []
    for idx, g in enumerate(groups):
        star = " &#9733;" if g["is_recoverable"] else ""
        status_class = "status-correct" if _is_correct(g["status"]) else "status-other"

        # Compute per-read identity from pileup data
        identity_str = ""
        if g["pileup_rows"]:
            total_bases = 0
            total_matches = 0
            for row in g["pileup_rows"]:
                aligned = [(b, m) for b, m in row if b != "-"]
                total_bases += len(aligned)
                total_matches += sum(1 for _, m in aligned if m)
            if total_bases > 0:
                identity = total_matches / total_bases
                identity_str = f" &middot; Read identity: {identity:.1%}"

        ref_len = len(g["ref_seq"])

        header = (
            f'<div class="group-header">'
            f'<span class="ref-name">{_html.escape(g["ref_id"])}{star}</span>'
            f'<span class="group-meta">'
            f'{g["n_reads"]} reads ({g["frac"]:.0%}) &middot; '
            f'{ref_len} bp &middot; '
            f'Consensus: <span class="{status_class}">'
            f'{_html.escape(g["status"])}</span>'
            f'{identity_str}'
            f'</span></div>'
        )

        # Encode pileup data compactly for JS:
        # '.' = match, base letter = mismatch, '-' = gap
        rows_encoded = []
        for row in g["pileup_rows"]:
            chars = []
            for base_char, is_match in row:
                if base_char == "-":
                    chars.append("-")
                elif is_match:
                    chars.append(".")
                else:
                    chars.append(base_char.upper())
            rows_encoded.append("".join(chars))

        # Build consensus from pileup: majority base at each position
        from collections import Counter
        consensus_encoded = []
        ref_seq = g["ref_seq"]
        for col_idx in range(ref_len):
            counts = Counter()
            for row in g["pileup_rows"]:
                base, _ = row[col_idx]
                if base != "-":
                    counts[base.upper()] += 1
            if counts:
                cons_base = counts.most_common(1)[0][0]
                if cons_base == ref_seq[col_idx].upper():
                    consensus_encoded.append(".")
                else:
                    consensus_encoded.append(cons_base)
            else:
                consensus_encoded.append("-")
        consensus_str = "".join(consensus_encoded)

        ref_seq_js = _json.dumps(ref_seq)
        rows_js = _json.dumps(rows_encoded)
        cons_js = _json.dumps(consensus_str)
        n_rows = len(rows_encoded)
        n_cols = ref_len

        if n_rows == 0:
            pileup_block = (
                f'<div class="pileup-empty">'
                f'No aligned reads available ({g["n_reads"]} reads unaligned)'
                f'</div>'
            )
        else:
            pileup_block = (
                f'<div class="pileup-container">'
                f'<div class="pileup-scroll" id="scroll-{idx}">'
                f'<canvas id="pileup-{idx}"></canvas>'
                f'</div>'
                f'<div class="pileup-info">{n_rows} aligned reads &times; '
                f'{n_cols} bp</div>'
                f'</div>'
                f'<script>'
                f'(function(){{'
                f'var ref={ref_seq_js};'
                f'var cons={cons_js};'
                f'var rows={rows_js};'
                f'drawPileup("pileup-{idx}",ref,cons,rows);'
                f'}})();'
                f'</script>'
            )

        sections_html.append(f'{header}\n{pileup_block}')

    body = '\n<hr class="group-sep">\n'.join(sections_html)

    recoverable_list = ", ".join(candidate["recoverable_variants"]) or "None"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<style id="usortm-theme-bridge">
:root {{
    --usortm-bg: #fafafa;
    --text: #1e293b;
    --muted: #94a3b8;
    --card-bg: #ffffff;
    --border: #e5e7eb;
}}
[data-theme="dark"] {{
    --usortm-bg: #1a1a2e;
    --text: #e0e0e0;
    --muted: #64748b;
    --card-bg: #16213e;
    --border: #334155;
}}
html, body {{
    background: var(--usortm-bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 1.5rem;
}}
h1 {{
    font-size: 1.4rem;
    margin: 0 0 0.25rem;
}}
.well-meta {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}}
.group-header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 1rem 0 0.5rem;
}}
.ref-name {{
    font-weight: 700;
    font-size: 1.05rem;
}}
.group-meta {{
    color: var(--muted);
    font-size: 0.85rem;
}}
.status-correct {{
    color: #059669;
    font-weight: 600;
}}
.status-other {{
    color: #ef4444;
}}
.pileup-container {{
    margin-bottom: 0.5rem;
}}
.pileup-scroll {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 60vh;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0;
}}
.pileup-scroll canvas {{
    display: block;
}}
.pileup-info {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.25rem;
}}
.pileup-empty {{
    font-size: 0.85rem;
    color: var(--muted);
    font-style: italic;
    padding: 1rem;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
}}
.legend {{
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.3rem;
}}
.legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid var(--border);
}}
.group-sep {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}}
</style>
<script>
function drawPileup(canvasId, refSeq, cons, rows) {{
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var nCols = refSeq.length;
  var nRows = rows.length;
  var cellW = nCols < 200 ? 4 : nCols < 500 ? 3 : 2;
  var cellH = nRows < 100 ? 3 : 2;
  var refH = Math.max(cellH, 6);
  var consH = refH;
  var gap = 1;
  canvas.width = nCols * cellW;
  canvas.height = refH + gap + consH + gap + nRows * cellH;
  var ctx = canvas.getContext('2d');
  var matchColor = '#059669';
  var gapColor = '#d1d5db';
  var refColor = '#1e293b';
  var consMatchColor = '#059669';
  var baseColors = {{'A':'#ef4444','T':'#3b82f6','C':'#f59e0b','G':'#8b5cf6'}};
  if (document.documentElement.getAttribute('data-theme') === 'dark') {{
    matchColor = '#34d399';
    gapColor = '#334155';
    refColor = '#e0e0e0';
    consMatchColor = '#34d399';
    baseColors = {{'A':'#f87171','T':'#60a5fa','C':'#fbbf24','G':'#a78bfa'}};
  }}
  // Reference row
  ctx.fillStyle = refColor;
  for (var i = 0; i < nCols; i++) {{
    ctx.fillRect(i * cellW, 0, cellW, refH);
  }}
  // Consensus row
  var consY = refH + gap;
  for (var i = 0; i < cons.length; i++) {{
    var ch = cons[i];
    if (ch === '.') {{
      ctx.fillStyle = consMatchColor;
    }} else if (ch === '-') {{
      ctx.fillStyle = gapColor;
    }} else {{
      ctx.fillStyle = baseColors[ch] || '#94a3b8';
    }}
    ctx.fillRect(i * cellW, consY, cellW, consH);
  }}
  // Read rows
  var readsY = consY + consH + gap;
  for (var r = 0; r < nRows; r++) {{
    var row = rows[r];
    var y = readsY + r * cellH;
    for (var c = 0; c < row.length; c++) {{
      var ch = row[c];
      if (ch === '.') {{
        ctx.fillStyle = matchColor;
      }} else if (ch === '-') {{
        ctx.fillStyle = gapColor;
      }} else {{
        ctx.fillStyle = baseColors[ch] || '#94a3b8';
      }}
      ctx.fillRect(c * cellW, y, cellW, cellH);
    }}
  }}
  // Tooltip
  var tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;background:#1e293b;color:#fff;padding:4px 8px;'
    + 'border-radius:4px;font-size:11px;pointer-events:none;display:none;z-index:10;'
    + 'font-family:SF Mono,Menlo,Consolas,monospace;';
  document.body.appendChild(tooltip);
  canvas.addEventListener('mousemove', function(e) {{
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var yp = e.clientY - rect.top;
    var col = Math.floor(x / cellW);
    if (col < 0 || col >= nCols) {{ tooltip.style.display = 'none'; return; }}
    if (yp < refH) {{
      tooltip.textContent = 'Ref pos ' + (col + 1) + ': ' + refSeq[col];
    }} else if (yp < consY + consH) {{
      var ch = cons[col];
      var base = ch === '.' ? refSeq[col] : ch;
      var note = ch === '.' ? ' (match)' : ch === '-' ? '' : ' (mismatch)';
      tooltip.textContent = 'Consensus pos ' + (col + 1) + ': ' + base + note;
    }} else {{
      var row_idx = Math.floor((yp - readsY) / cellH);
      if (row_idx >= 0 && row_idx < nRows) {{
        var ch = rows[row_idx][col];
        var label = ch === '.' ? refSeq[col] + ' (match)' : ch === '-' ? 'gap' : ch + ' (mismatch)';
        tooltip.textContent = 'Read ' + (row_idx + 1) + ', pos ' + (col + 1) + ': ' + label;
      }} else {{
        tooltip.style.display = 'none'; return;
      }}
    }}
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 24) + 'px';
  }});
  canvas.addEventListener('mouseleave', function() {{
    tooltip.style.display = 'none';
  }});
}}
</script>
</head>
<body>
<h1>{_html.escape(title)}</h1>
<div class="well-meta">
    {candidate["total_reads"]} total reads &middot;
    Top fraction: {candidate["top_frac"]:.0%} &middot;
    Recoverable: {_html.escape(recoverable_list)}
</div>
<div class="legend">
    <span style="font-weight:600;">Legend:</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#059669;"></span> Match</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ef4444;"></span> A</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#3b82f6;"></span> T</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#f59e0b;"></span> C</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#8b5cf6;"></span> G</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#d1d5db;"></span> Gap</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#1e293b;"></span> Reference</span>
    <span style="color:var(--muted);">|</span>
    <span style="font-size:0.75rem;color:var(--muted);">Rows: Reference &rarr; Consensus &rarr; Reads</span>
</div>
{body}
<script id="usortm-theme-sync">
(function () {{
  try {{
    var stored = localStorage.getItem('usortm-theme');
    if (stored === 'dark') {{
      document.documentElement.setAttribute('data-theme', 'dark');
    }}
  }} catch (e) {{}}
}})();
</script>
</body>
</html>"""
