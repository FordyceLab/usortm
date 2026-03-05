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

    Returns one of: "Perfect Match", "Silent Mutation", "Partial Match",
    "Other Error", or "Error".
    """
    if cigar is None or cons_seq is None:
        return "Error"

    alpha = ''.join(c for c in cigar if c.isalpha()).lower()
    if alpha == 'm':
        num = int(cigar[:-1])
        if num == int(ref_len):
            return "Perfect Match"
        return "Partial Match"

    # Non-pure-M CIGAR — check for silent mutations
    if len(cons_seq) == ref_len:
        try:
            if Seq.translate(ref_seq) == Seq.translate(cons_seq):
                return "Silent Mutation"
        except Exception:
            pass
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

    # 3) Re-align consensus to reference
    try:
        mm2 = subprocess.Popen(
            [minimap2_path, "-a", ref_fasta, cons_fa],
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
                [minimap2_path, "-a", ref_fasta, fq_path],
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
        except Exception:
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
        except Exception:
            pass

    return rows


def _render_pileup_html(well_pos: str, candidate: dict,
                         groups: list[dict]) -> str:
    """Render the pileup HTML page for one well."""
    import html as _html

    plate = candidate["plate"]
    well = candidate["well"]
    title = f"Pileup: Plate {plate} Well {well}"

    sections_html = []
    for g in groups:
        star = " &#9733;" if g["is_recoverable"] else ""
        status_class = "status-correct" if _is_correct(g["status"]) else "status-other"

        header = (
            f'<div class="group-header">'
            f'<span class="ref-name">{_html.escape(g["ref_id"])}{star}</span>'
            f'<span class="group-meta">'
            f'{g["n_reads"]} reads ({g["frac"]:.0%}) &middot; '
            f'<span class="{status_class}">{_html.escape(g["status"])}</span>'
            f'</span></div>'
        )

        # Reference line
        ref_line = ''.join(
            f'<span class="base ref-base">{b}</span>'
            for b in g["ref_seq"]
        )

        # Read lines (limit to 100 rows for performance)
        read_lines = []
        for row in g["pileup_rows"][:100]:
            chars = []
            for base_char, is_match in row:
                if base_char == "-":
                    chars.append('<span class="base gap">-</span>')
                elif is_match:
                    chars.append(f'<span class="base match">{_html.escape(base_char)}</span>')
                else:
                    chars.append(f'<span class="base mismatch">{_html.escape(base_char)}</span>')
            read_lines.append(''.join(chars))

        truncation_note = ""
        if len(g["pileup_rows"]) > 100:
            truncation_note = (
                f'<div class="truncation-note">'
                f'Showing 100 of {len(g["pileup_rows"])} reads</div>'
            )

        pileup_block = (
            f'<div class="pileup-scroll"><pre class="pileup">'
            f'<div class="ref-row">{ref_line}</div>'
            + '\n'.join(f'<div class="read-row">{rl}</div>' for rl in read_lines)
            + f'</pre></div>{truncation_note}'
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
.pileup-scroll {{
    overflow-x: auto;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem;
}}
.pileup {{
    margin: 0;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 11px;
    line-height: 1.3;
    white-space: pre;
}}
.base {{
    display: inline-block;
    width: 0.7em;
    text-align: center;
}}
.ref-base {{
    font-weight: 700;
}}
.match {{
    color: var(--muted);
}}
.mismatch {{
    color: #ef4444;
    font-weight: 700;
}}
.gap {{
    color: var(--muted);
    opacity: 0.5;
}}
.ref-row {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 2px;
    margin-bottom: 2px;
}}
.group-sep {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}}
.truncation-note {{
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}}
</style>
</head>
<body>
<h1>{_html.escape(title)}</h1>
<div class="well-meta">
    {candidate["total_reads"]} total reads &middot;
    Top fraction: {candidate["top_frac"]:.0%} &middot;
    Recoverable: {_html.escape(recoverable_list)}
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
