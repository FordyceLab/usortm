"""What each well of a barcoded plate is expected to hold.

Outside a uSort-M sort, demultiplexing is a check: a plate of known
constructs is barcoded, pooled and sequenced, and the question is whether
each well holds what it should.  The expectation comes as a CSV, one row per
well, addressed the way the LevSeq barcodes address it.

A LevSeq plate is 384 wells: the forward barcode (FB01-FB96) gives a
position in a 96 grid, and the reverse barcode gives the plate and the
quadrant, four reverse barcodes to a plate.  The quadrants interleave rather
than tile -- TL takes odd rows and odd columns, TR odd and even, BL even and
odd, BR even and even -- which is what :func:`usortm.demux.utils.barcode_to_well`
decodes.  So a CSV can say where a well is in either of two ways:

``plate, well, name, sequence``
    A 384-well position, A1-P24, taken as it is.

``plate, quadrant, well, name, sequence`` (or ``rbc, well, ...``)
    A 96-well plate barcoded as one quadrant of a LevSeq plate: its A1-H12
    position, and the quadrant (TL/TR/BL/BR or 1-4) or the reverse barcode
    (RB01-RB32) it was given.

Sequences follow the library CSV convention: uppercase is the variable
region, and lowercase flanks are dropped.  A row with no sequence is a well
expected to be empty, so reads found there are reported rather than ignored.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAX_PLATES = 8  # RB01-RB32; barcode_to_well decodes no further
QUADRANTS = ("TL", "TR", "BL", "BR")
_WELL = re.compile(r"^\s*([A-Pa-p])\s*0*(\d{1,2})\s*$")
_RBC = re.compile(r"^\s*(?:RB)?0*(\d{1,2})\s*$", re.IGNORECASE)
_FASTA_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


class ExpectedPlateError(ValueError):
    """Raised when an expected-plate CSV cannot be read."""


@dataclass(frozen=True)
class ExpectedWell:
    """One well's expectation.

    Attributes:
        plate: LevSeq plate, 1-8.
        well: 384-well position, e.g. ``"B12"``, as the pipeline reports it.
        name: Construct name, empty for a well expected to be empty.
        sequence: Expected variable region, uppercase; empty when empty.
        label: The well as the CSV gave it, e.g. ``"A6"`` or ``"TR:A6"``.
        row: CSV line the well came from, for error messages.
    """

    plate: int
    well: str
    name: str
    sequence: str
    label: str
    row: int

    @property
    def key(self) -> tuple:
        return (self.plate, self.well)

    @property
    def empty(self) -> bool:
        return not self.sequence


@dataclass
class ExpectedPlate:
    """Every expected well, keyed by ``(plate, 384-well position)``.

    Attributes:
        wells: ``{(plate, well): ExpectedWell}``.
        layout: ``"384"`` or ``"96"``, whichever the CSV used.
        source: The CSV path.
        notes: Things worth telling the user that are not errors.
    """

    wells: dict
    layout: str
    source: Path
    notes: list = field(default_factory=list)

    @property
    def plates(self) -> list:
        return sorted({p for p, _ in self.wells})

    @property
    def n_plates(self) -> int:
        """Plates the run needs reverse barcodes for: up to the highest used."""
        return max(self.plates) if self.plates else 1

    def constructs(self) -> dict:
        """``{sequence: reference name}``, one entry per distinct sequence.

        Wells that share a sequence share a reference, named after the first
        construct to use it; a reference per well would make identical
        references compete for the same reads.
        """
        out: dict = {}
        used: set = set()
        for w in sorted(self.wells.values(), key=lambda w: w.row):
            if w.empty or w.sequence in out:
                continue
            base = _FASTA_UNSAFE.sub("_", w.name).strip("_") or f"construct_{w.row}"
            name, n = base, 2
            while name in used:
                name, n = f"{base}_{n}", n + 1
            used.add(name)
            out[w.sequence] = name
        return out

    def reference_name(self, well: ExpectedWell) -> Optional[str]:
        """The reference a well's expected sequence is written under."""
        return None if well.empty else self.constructs()[well.sequence]

    def write_reference_fasta(self, path) -> Path:
        """Write one reference per distinct expected sequence."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for seq, name in self.constructs().items():
                fh.write(f">{name}\n{seq}\n")
        return path


def quadrant_to_384(quadrant: int, row96: int, col96: int) -> str:
    """384-well position of a 96-well position in one quadrant.

    Args:
        quadrant: 0-3 for TL, TR, BL, BR.
        row96: 1-based 96-well row, 1-8.
        col96: 1-based 96-well column, 1-12.

    Returns:
        The 384-well name, e.g. ``"B2"`` for A1 in BR.
    """
    row_off = 1 if quadrant in (0, 1) else 2
    col_off = 1 if quadrant in (0, 2) else 2
    row384 = (row96 - 1) * 2 + row_off
    col384 = (col96 - 1) * 2 + col_off
    return f"{chr(ord('A') + row384 - 1)}{col384}"


def _insert(seq: str) -> str:
    """The variable region: uppercase only, unless nothing is uppercase."""
    seq = re.sub(r"\s+", "", seq or "")
    upper = "".join(c for c in seq if c.isupper())
    return upper if upper else seq.upper()


def _parse_quadrant(value: str) -> Optional[int]:
    v = value.strip().upper()
    if v in QUADRANTS:
        return QUADRANTS.index(v)
    if v in {"1", "2", "3", "4"}:
        return int(v) - 1
    return None


def read_expected_plate(path) -> ExpectedPlate:
    """Read an expected-plate CSV.

    Args:
        path: CSV with ``plate``, ``well``, ``name`` and ``sequence`` columns,
            plus ``quadrant`` or ``rbc`` for a 96-well plate.  Headers are
            matched ignoring case and surrounding space.

    Returns:
        ExpectedPlate.

    Raises:
        ExpectedPlateError: On a missing column, a well or plate out of
            range, a sequence with anything but bases in it, or two rows for
            the same well.
    """
    path = Path(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ExpectedPlateError(f"{path}: no header row.")
        cols = {h.strip().lower(): h for h in reader.fieldnames if h}
        rows = list(reader)

    missing = [c for c in ("well", "sequence") if c not in cols]
    if missing:
        raise ExpectedPlateError(
            f"{path}: missing column(s) {', '.join(missing)}. Found: "
            f"{', '.join(reader.fieldnames)}."
        )
    by_rbc = "rbc" in cols
    by_quadrant = "quadrant" in cols
    if by_rbc and by_quadrant:
        raise ExpectedPlateError(
            f"{path}: give either a quadrant column or an rbc column, not "
            "both; each says which reverse barcode a 96-well plate was given."
        )
    layout = "96" if (by_rbc or by_quadrant) else "384"
    if not by_rbc and "plate" not in cols:
        if layout == "96":
            raise ExpectedPlateError(
                f"{path}: a quadrant column needs a plate column beside it; "
                "the quadrant says where on the LevSeq plate, not which plate."
            )
    max_row, max_col = ("H", 12) if layout == "96" else ("P", 24)

    def _get(row, col):
        return (row.get(cols[col]) or "").strip() if col in cols else ""

    wells: dict = {}
    for i, row in enumerate(rows, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue
        where = f"{path}, line {i}"

        m = _WELL.match(_get(row, "well"))
        if not m:
            raise ExpectedPlateError(f"{where}: well {_get(row, 'well')!r} is not a well name.")
        r_letter, col = m.group(1).upper(), int(m.group(2))
        if r_letter > max_row or not 1 <= col <= max_col:
            hint = (" A 384-well position needs no quadrant or rbc column."
                    if layout == "96" else "")
            raise ExpectedPlateError(
                f"{where}: well {r_letter}{col} is outside A1-{max_row}{max_col} "
                f"for a {layout}-well plate.{hint}"
            )

        if by_rbc:
            mr = _RBC.match(_get(row, "rbc"))
            rb = int(mr.group(1)) if mr else 0
            if not 1 <= rb <= 4 * MAX_PLATES:
                raise ExpectedPlateError(
                    f"{where}: reverse barcode {_get(row, 'rbc')!r} is not "
                    f"RB01-RB{4 * MAX_PLATES}."
                )
            plate, quadrant = (rb - 1) // 4 + 1, (rb - 1) % 4
            if "plate" in cols and _get(row, "plate") and _get(row, "plate") != str(plate):
                raise ExpectedPlateError(
                    f"{where}: RB{rb:02d} is on plate {plate}, but the plate "
                    f"column says {_get(row, 'plate')}."
                )
        else:
            try:
                plate = int(_get(row, "plate") or "1") if "plate" in cols else 1
            except ValueError:
                raise ExpectedPlateError(f"{where}: plate {_get(row, 'plate')!r} is not a number.")
            quadrant = None
            if by_quadrant:
                quadrant = _parse_quadrant(_get(row, "quadrant"))
                if quadrant is None:
                    raise ExpectedPlateError(
                        f"{where}: quadrant {_get(row, 'quadrant')!r} is not "
                        "TL, TR, BL, BR or 1-4."
                    )
        if not 1 <= plate <= MAX_PLATES:
            raise ExpectedPlateError(
                f"{where}: plate {plate} is outside 1-{MAX_PLATES}, the plates "
                "the LevSeq reverse barcodes cover."
            )

        if quadrant is None:
            well = f"{r_letter}{col}"
            label = well
        else:
            well = quadrant_to_384(quadrant, ord(r_letter) - ord("A") + 1, col)
            label = f"{QUADRANTS[quadrant]}:{r_letter}{col}"

        seq = _insert(_get(row, "sequence"))
        bad = sorted(set(seq) - set("ACGT"))
        if bad:
            raise ExpectedPlateError(
                f"{where}: the sequence contains {''.join(bad)}; expected "
                "A, C, G and T only."
            )
        name = _get(row, "name") if "name" in cols else ""
        if seq and not name:
            name = f"{plate}_{label}"

        entry = ExpectedWell(plate=plate, well=well, name=name if seq else "",
                             sequence=seq, label=label, row=i)
        if entry.key in wells:
            first = wells[entry.key]
            raise ExpectedPlateError(
                f"{where}: plate {plate} {label} is already given on line "
                f"{first.row}."
            )
        wells[entry.key] = entry

    if not any(not w.empty for w in wells.values()):
        raise ExpectedPlateError(f"{path}: no well has an expected sequence.")

    notes = []
    if layout == "384":
        rows_used = {w.well[0] for w in wells.values()}
        cols_used = {int(w.well[1:]) for w in wells.values()}
        if max(rows_used) <= "H" and max(cols_used) <= 12 and len(wells) > 12:
            notes.append(
                "Every well is within A1-H12 and there is no quadrant or rbc "
                "column, so the wells were read as 384-well positions. If "
                "this is a 96-well plate barcoded as one quadrant, add a "
                "quadrant column."
            )
    return ExpectedPlate(wells=wells, layout=layout, source=path, notes=notes)
