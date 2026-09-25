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
import re
from dataclasses import dataclass
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


def layout_from_worklists(transfers: Sequence[dict],
                          pick_list: Optional[Iterable[dict]] = None
                          ) -> Tuple[List[dict], List[str]]:
    """One row per destination well, as the robot filled it.

    The worklists are what the liquid handler ran, so they, not the pick,
    say what went into each well: the pick can be re-run after the plate is
    built, and the worklists cannot.  *transfers* are
    :func:`usortm.integra.read_integra_files` rows.  The pick, when given,
    supplies the sequenced source position and read count for each well and
    is checked against the worklists; every well where the two disagree, and
    every placed well with no transfer, is returned as a problem.  The
    layout follows the worklists either way.
    """
    by_target: Dict[str, dict] = {}
    problems: List[str] = []
    for t in transfers:
        well = t["target_well"]
        if well in by_target:
            problems.append(f"{well}: two transfers, from {by_target[well]['file']} "
                            f"and {t['file']}")
            continue
        by_target[well] = t

    placed = {}
    for e in pick_list or ():
        if e.get("empty") or not e.get("target_well") or not e.get("source_well"):
            continue
        if e.get("tier_override") == "Streakout":
            continue
        placed[str(e["target_well"]).upper()] = e

    rows = []
    for well, t in by_target.items():
        e = placed.get(well)
        bench_plate = f"R{t['round']}_{t['source_plate']}"
        row = {"well": well, "variant": t["variant"], "source_round": t["round"],
               "source_plate": bench_plate, "source_well": t["source_well"],
               "bench_plate": "", "bench_well": "", "reads": 0}
        if e is not None:
            e_plate = str(e.get("bench_plate") or e.get("source_plate") or "")
            e_well = str(e.get("bench_well") or e.get("source_well") or "").upper()
            if (e["variant"], e_plate.split("_")[-1], e_well) != (
                    t["variant"], t["source_plate"], t["source_well"]):
                problems.append(
                    f"{well}: the worklist moved {t['variant']} from plate "
                    f"{t['source_plate']} {t['source_well']}; the pick places "
                    f"{e['variant']} from {e_plate} {e_well}")
            else:
                # Agreed: keep the pick's sequenced position and depth, which
                # is how the well's source is named everywhere else.
                row.update({"source_plate": str(e.get("source_plate") or ""),
                            "source_well": str(e.get("source_well") or "").upper(),
                            "bench_plate": str(e.get("bench_plate") or ""),
                            "bench_well": str(e.get("bench_well") or ""),
                            "reads": e.get("reads") or 0})
        elif pick_list is not None:
            problems.append(f"{well}: the worklist moved {t['variant']} there; "
                            f"the pick places nothing")
        rows.append(row)
    for well in sorted(set(placed) - set(by_target)):
        problems.append(f"{well}: the pick places {placed[well]['variant']} there; "
                        f"no worklist moved anything")
    rows.sort(key=lambda r: (r["well"][0], int(r["well"][1:])))
    return rows, problems


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
    # Where the demux held a well to its expected variant, the row's variant
    # is the expectation and the free assignment sits in assigned_variant;
    # that is what a wrong well was read as, and what the verdict should say.
    from dataclasses import replace

    by_key = {(int(r["plate"]), str(r["well"]).upper()): r for r in well_rows}
    out = []
    for v in verdicts:
        row = by_key.get((v.plate, v.well))
        got = (row or {}).get("assigned_variant")
        if v.status != CONFIRMED and got:
            v = replace(v, observed=got)
        out.append(v)
    return out, summarise(out)


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
            av = (r.get("assigned_variant") or "").strip()
            if av:
                row["assigned_variant"] = av
            rows.append(row)
    return rows


def designed_names(demux_output_dir) -> set:
    """The library's members, from the per-variant references demux wrote."""
    d = Path(demux_output_dir) / "reference_fasta" / "single_ref_fastas"
    names = {p.name[:-6] for p in d.glob("*.fasta")} if d.exists() else set()
    names.discard("Parent")
    return names


# --- runs ------------------------------------------------------------------
#
# A pick-plate run sequences the plate the merge built.  It samples nothing
# new, so it is not a round: it never feeds a merge and never counts toward
# recovery.  Runs sit in their own numbered step directory, named by the
# sequencing run that produced them, and are recorded under their own key in
# the project state.  See docs/project-layout.md.

#: The step directory runs live in, numbered for its place in the workflow.
RUNS_DIR = "7_pick_plate"
#: Where the project state records runs.
STATE_KEY = "pick_plate_runs"
#: Run names are sequencing IDs such as G3Y8KW, used as directory names.
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RunPaths:
    """Every path one pick-plate run uses.

    ``demux/`` holds what the FASTQs can rebuild; ``results/`` holds what is
    kept, so deleting ``demux/`` loses nothing the report needs.
    """

    project: Path
    name: str

    @property
    def root(self) -> Path:
        return self.project / RUNS_DIR / self.name

    @property
    def plate_map(self) -> Path:
        return self.root / "plate_map.toml"

    @property
    def layout(self) -> Path:
        return self.root / LAYOUT_FILE

    @property
    def demux(self) -> Path:
        return self.root / "demux"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def wells_csv(self) -> Path:
        """The per-well table, copied out of ``demux/`` when the demux ends."""
        return self.results / "well_assignments.csv"

    @property
    def verdicts(self) -> Path:
        return self.results / VERDICT_FILE

    @property
    def pileups(self) -> Path:
        return self.results / "pileups"


def run_paths(project_dir, name: str) -> RunPaths:
    """Paths for run *name*, which must be usable as a directory name."""
    if not _RUN_NAME.match(str(name or "")):
        raise ValueError(
            f"{name!r} is not a usable run name; use the sequencing run's "
            f"ID, such as G3Y8KW (letters, digits, '.', '_' and '-')")
    return RunPaths(Path(project_dir), str(name))


def pick_for_layout(project_dir) -> Optional[Path]:
    """The pick the plate was built from: the merge, else the single pick."""
    for rel in ("merged/pick_list.json", "pick/pick_list.json"):
        path = Path(project_dir) / rel
        if path.exists():
            return path
    return None


def list_runs(project: dict) -> List[Tuple[str, dict]]:
    """Recorded runs, oldest first by the time each was started."""
    runs = (project.get(STATE_KEY) or {}).items()
    return sorted(((n, b or {}) for n, b in runs),
                  key=lambda nb: (nb[1].get("created") or "", nb[0]))


def newest_run(project: dict) -> Optional[str]:
    runs = list_runs(project)
    return runs[-1][0] if runs else None


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
           "pick_plate_rounds", "RUNS_DIR", "STATE_KEY", "RunPaths", "run_paths",
           "pick_for_layout", "list_runs", "newest_run", "layout_from_worklists"]
