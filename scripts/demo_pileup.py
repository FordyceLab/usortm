#!/usr/bin/env python3
from __future__ import annotations

"""Generate a demo pileup HTML page with synthetic data.

Imports _render_pileup_html from streakout and writes a self-contained HTML
file — no minimap2, samtools, or real reads needed.

Usage:
    python scripts/demo_pileup.py [output_path]
"""

import random
import sys
import webbrowser
from pathlib import Path

from usortm.demux.streakout import _render_pileup_html

BASES = "ACGT"


def _random_base(exclude: str) -> str:
    """Return a random base that is not *exclude*."""
    return random.choice([b for b in BASES if b != exclude])


def _make_ref(length: int = 50) -> str:
    return "".join(random.choice(BASES) for _ in range(length))


def _make_pileup_rows(
    ref_seq: str,
    n_rows: int,
    *,
    mismatch_rate: float = 0.02,
    gap_rate: float = 0.01,
    mutations: dict[int, str] | None = None,
) -> list[list[tuple[str, bool]]]:
    """Synthesize pileup rows against *ref_seq*.

    *mutations* maps position → forced mismatch base (applied to every read).
    """
    mutations = mutations or {}
    rows: list[list[tuple[str, bool]]] = []
    for _ in range(n_rows):
        row: list[tuple[str, bool]] = []
        for i, ref_base in enumerate(ref_seq):
            ref_upper = ref_base.upper()
            r = random.random()
            if i in mutations:
                row.append((mutations[i], False))
            elif r < gap_rate:
                row.append(("-", False))
            elif r < gap_rate + mismatch_rate:
                row.append((_random_base(ref_upper), False))
            else:
                row.append((ref_upper, True))
        rows.append(row)
    return rows


def main() -> None:
    random.seed(42)

    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_pileup.html")

    # Build reference with 100 bp flanks + 800 bp insert = 1000 bp total
    flank_5p_len = 100
    flank_3p_len = 100
    flank_5p = _make_ref(flank_5p_len)
    insert_seq = _make_ref(800)
    flank_3p = _make_ref(flank_3p_len)
    ref_seq = flank_5p + insert_seq + flank_3p

    # --- Group 1: WT (major, ~65%) ---
    wt_rows = _make_pileup_rows(ref_seq, n_rows=10, mismatch_rate=0.02, gap_rate=0.01)

    # --- Group 2: K44A mutant (minor, ~35%) with a forced mismatch in insert region ---
    mutant_pos = flank_5p_len + 20  # position within the insert
    forced_base = _random_base(ref_seq[mutant_pos].upper())
    k44a_rows = _make_pileup_rows(
        ref_seq, n_rows=8, mismatch_rate=0.03, gap_rate=0.01,
        mutations={mutant_pos: forced_base},
    )

    groups = [
        {
            "ref_id": "pUC19-WT",
            "n_reads": 65,
            "frac": 0.65,
            "status": "Perfect Match",
            "is_recoverable": True,
            "ref_seq": ref_seq,
            "pileup_rows": wt_rows,
        },
        {
            "ref_id": "pUC19-K44A",
            "n_reads": 35,
            "frac": 0.35,
            "status": "Mismatch",
            "is_recoverable": False,
            "ref_seq": ref_seq,
            "pileup_rows": k44a_rows,
        },
    ]

    candidate = {
        "plate": "DemoPlate",
        "well": "A1",
        "recoverable_variants": ["pUC19-WT"],
        "total_reads": 100,
        "top_frac": 0.65,
    }

    html = _render_pileup_html("A1", candidate, groups,
                              flank_lengths=(flank_5p_len, flank_3p_len))
    output_path.write_text(html)
    print(f"Wrote {output_path.resolve()}")
    webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
