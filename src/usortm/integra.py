"""The pick, written the way the Integra ASSIST PLUS is loaded.

The robot works one source plate at a time: a plate goes on the deck and its
file is run.  So the pick is written as one file per source plate,
``integra_assist_plate{N}.csv``, each holding only the transfers out of that
plate -- the form the lab's earlier picks were run from.  A source plate with
nothing to pick still gets its file, header only, so a plate that is missing
from the set is a plate that was not in the run rather than one that was
overlooked.

Columns are the robot's: ``SampleID;SourcePlateID;SourceWell;TargetPlateID;
TargetWell;TransferVolume``, semicolon-delimited.  The volume is written as
the robot reads it, ``5`` rather than ``5.0``.

Picks were previously written one file per *target* plate, with every
source plate mixed together; those files are removed when the new ones are
written so a stale layout cannot be loaded beside the current one.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional

HEADER = ["SampleID", "SourcePlateID", "SourceWell",
          "TargetPlateID", "TargetWell", "TransferVolume"]

FILE_GLOB = "integra_assist_*plate*.csv"
LEGACY_GLOB = "hitlist_plate_*.csv"

_ROUND_PREFIXED = re.compile(r"^R(\d+)_(.+)$")


def integra_filename(source_plate) -> str:
    """The file a source plate's transfers are written to.

    ``3`` gives ``integra_assist_plate3.csv``.  A merge names plates by round,
    ``R2_3``, which gives ``integra_assist_R2_plate3.csv``: the round stays in
    the name because two rounds can both have a plate 3.
    """
    s = str(source_plate)
    m = _ROUND_PREFIXED.match(s)
    if m:
        return f"integra_assist_R{m.group(1)}_plate{m.group(2)}.csv"
    return f"integra_assist_plate{s}.csv"


def format_volume(volume: float) -> str:
    """``5.0`` as ``5``, ``2.5`` as ``2.5``: the robot's own spelling."""
    return f"{float(volume):g}"


def _plate_sort_key(source_plate) -> tuple:
    s = str(source_plate)
    m = _ROUND_PREFIXED.match(s)
    rnd, plate = (int(m.group(1)), m.group(2)) if m else (0, s)
    return (rnd, 0, int(plate)) if plate.isdigit() else (rnd, 1, plate)


def _default_skip(hit: dict) -> bool:
    return bool(hit.get("empty")) or hit.get("tier_override") == "Streakout"


def write_integra_files(
    pick_list: Iterable[dict],
    out_dir: Path,
    volume: float,
    source_plates: Optional[Iterable] = None,
    skip: Callable[[dict], bool] = _default_skip,
) -> List[Path]:
    """Write one file per source plate and return them, in plate order.

    *source_plates* names every plate that should get a file, whether or not
    anything is picked from it; plates that appear in the pick are always
    included.  *skip* says which entries are not transfers -- empty
    placeholders, and streak-out entries, which are instructions for a person
    rather than the robot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A hit may carry a bench position -- the 96-well colony plate and well a
    # re-ordered construct was assembled in -- beside the position it was
    # sequenced at.  The robot picks from the bench, so that is what is
    # written when it is there.
    def plate_of(hit):
        return str(hit.get("bench_plate") or hit["source_plate"])

    def well_of(hit):
        return hit.get("bench_well") or hit["source_well"]

    by_plate: dict = {}
    for plate in (source_plates or ()):
        by_plate.setdefault(str(plate), [])
    for hit in pick_list:
        if skip(hit):
            continue
        by_plate.setdefault(plate_of(hit), []).append(hit)

    for stale in list(out_dir.glob(LEGACY_GLOB)) + list(out_dir.glob(FILE_GLOB)):
        stale.unlink()

    written = []
    for plate in sorted(by_plate, key=_plate_sort_key):
        path = out_dir / integra_filename(plate)
        with open(path, "w", newline="") as fh:
            # "\n" line endings, as in the files the lab has run from; the
            # csv module's default is "\r\n".
            writer = csv.writer(fh, delimiter=";", lineterminator="\n")
            writer.writerow(HEADER)
            for hit in by_plate[plate]:
                writer.writerow([
                    str(hit["variant"]).replace(";", "."),
                    plate_of(hit),
                    well_of(hit),
                    hit["target_plate"],
                    hit["target_well"],
                    format_volume(volume),
                ])
        written.append(path)
    return written
