"""The picked plate, sequenced and judged against what was put in it.

After the merge the destination plate is built by the robot, barcoded and
sequenced.  Every well on it has an intended variant -- the one the merge
placed there -- so the question is not what each well holds but whether it
holds what it should.  That is the same question the re-order round asks of
its assembled constructs, and it is answered with the same machinery
(:mod:`usortm.verify`): an expected layout, the round's own demux, and a
verdict per well of confirmed, wrong or empty.

The expected layout is written when the round is planned, from the merged
pick, so the record of what was intended is fixed before the reads arrive.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from usortm.verify import (CONFIRMED, EMPTY, WRONG, ExpectedWell, WellVerdict,
                           failure_reason, summarise, verify)

ROUND_KIND = "pick_plate"
LAYOUT_FILE = "expected_layout.csv"
VERDICT_FILE = "pick_plate_verdicts.csv"
LAYOUT_COLUMNS = ["well", "variant", "source_round", "source_plate",
                  "source_well", "bench_plate", "bench_well", "reads"]


def expected_layout_from_pick(pick_list: Iterable[dict]) -> List[dict]:
    """One row per destination well the merge filled.

    Empty placeholders and streak-out entries are not on the plate and are
    left out; the 96-well bench position is kept where the pick carried one.
    """
    rows = []
    for e in pick_list:
        if e.get("empty") or not e.get("source_well") or not e.get("target_well"):
            continue
        if e.get("tier_override") == "Streakout":
            continue
        rows.append({
            "well": str(e["target_well"]).upper(),
            "variant": e["variant"],
            "source_round": e.get("source_round") or 1,
            "source_plate": str(e.get("source_plate") or ""),
            "source_well": str(e.get("source_well") or ""),
            "bench_plate": str(e.get("bench_plate") or ""),
            "bench_well": str(e.get("bench_well") or ""),
            "reads": e.get("reads") or 0,
        })
    rows.sort(key=lambda r: (r["well"][0], int(r["well"][1:])))
    return rows


def write_expected_layout(rows: Sequence[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LAYOUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LAYOUT_COLUMNS})
    return path


def read_expected_layout(path) -> List[dict]:
    with open(path, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def expected_wells_from_layout(rows: Iterable[dict], plate: int = 1
                               ) -> Dict[Tuple[int, str], ExpectedWell]:
    """The layout in the form :func:`usortm.verify.verify` takes.

    The plate is the one sequenced plate; the well is both where the reads
    came from and where the construct sits, so ``order_well`` is the well.
    """
    return {
        (plate, r["well"].upper()): ExpectedWell(
            plate=plate, well=r["well"].upper(), variant=r["variant"],
            replicate=1, order_well=r["well"].upper())
        for r in rows
    }


def judge(well_rows: Sequence[dict], layout: Sequence[dict], designed: set,
          plate: int = 1, min_reads: int = 20, is_clean=None
          ) -> Tuple[List[WellVerdict], dict]:
    """Verdicts and their summary for a sequenced pick plate."""
    expected = expected_wells_from_layout(layout, plate)
    verdicts = verify(well_rows, expected, designed, min_reads=min_reads,
                      is_clean=is_clean)
    return verdicts, summarise(verdicts)


def verdict_rows(verdicts: Sequence[WellVerdict], well_rows: Sequence[dict],
                 layout: Sequence[dict]) -> List[dict]:
    """The verdict table as written to disk: one row per intended well."""
    by_key = {(int(r["plate"]), str(r["well"]).upper()): r for r in well_rows}
    by_well = {r["well"].upper(): r for r in layout}
    out = []
    for v in verdicts:
        row = by_key.get((v.plate, v.well))
        lay = by_well.get(v.well, {})
        out.append({
            "well": v.well,
            "expected": v.expected,
            "observed": v.observed or "",
            "reads": v.reads,
            "status": v.status,
            "reason": "" if v.status == CONFIRMED else (failure_reason(row, v) or ""),
            "max_mismatch_frac": (row or {}).get("max_mismatch_frac", ""),
            "source_round": lay.get("source_round", ""),
            "source_plate": lay.get("source_plate", ""),
            "source_well": lay.get("source_well", ""),
            "bench_plate": lay.get("bench_plate", ""),
            "bench_well": lay.get("bench_well", ""),
        })
    return out


def write_verdicts(rows: Sequence[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["well", "expected", "observed", "reads", "status", "reason",
            "max_mismatch_frac", "source_round", "source_plate", "source_well",
            "bench_plate", "bench_well"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return path


def load_well_rows(path) -> List[dict]:
    """A round's well_assignments.csv with the fields the verdict needs typed."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {
                "plate": r["plate"], "well": r["well"], "variant": r["variant"],
                "reads": int(float(r.get("reads") or 0)),
                "consensus_fraction": float(r.get("consensus_fraction") or 0),
                "cons_check": r.get("cons_check", ""),
                "flank_check": r.get("flank_check", ""),
            }
            mmf = (r.get("max_mismatch_frac") or "").strip()
            row["max_mismatch_frac"] = float(mmf) if mmf else None
            nfp = (r.get("n_flagged_positions") or "").strip()
            if nfp:
                row["n_flagged_positions"] = int(float(nfp))
            rows.append(row)
    return rows


def designed_names(demux_output_dir) -> set:
    """The library's members, from the per-variant references demux wrote."""
    d = Path(demux_output_dir) / "reference_fasta" / "single_ref_fastas"
    names = {p.name[:-6] for p in d.glob("*.fasta")} if d.exists() else set()
    names.discard("Parent")
    return names


def pick_plate_rounds(project: dict) -> List[int]:
    """Round numbers planned as pick-plate verification rounds."""
    out = []
    for rnum, block in (project.get("rounds") or {}).items():
        if (block or {}).get("kind") == ROUND_KIND:
            out.append(int(rnum))
    return sorted(out)


__all__ = ["CONFIRMED", "EMPTY", "WRONG", "ROUND_KIND", "LAYOUT_FILE",
           "VERDICT_FILE", "expected_layout_from_pick", "write_expected_layout",
           "read_expected_layout", "expected_wells_from_layout", "judge",
           "verdict_rows", "write_verdicts", "load_well_rows", "designed_names",
           "pick_plate_rounds"]
