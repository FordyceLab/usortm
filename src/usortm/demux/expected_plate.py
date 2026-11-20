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
decodes.

Two wells matter for each sample, and the CSV names them apart: the barcode
well, the LevSeq address it was barcoded with, which is what demultiplexing
reads; and the sample well, where it sits on the plate it came from, which
is only reported.  A bare ``well`` or ``plate`` column is refused.  The
barcode well can be given in either of two ways:

``barcode_plate, barcode_well, name, sequence``
    A 384-well position, A1-P24, taken as it is.  ``barcode_plate``
    defaults to 1.

``barcode_plate, quadrant, barcode_well, ...`` (or ``rbc, barcode_well, ...``)
    A 96-well plate barcoded as one quadrant of a LevSeq plate: its A1-H12
    position, and the quadrant (TL/TR/BL/BR or 1-4) or the reverse barcode
    (RB01-RB32) it was given.

``sample_plate`` and ``sample_well`` may be added to either.

Sequences follow the library CSV convention: uppercase is the variable
region, and lowercase flanks are dropped.  A row with no sequence is a well
expected to be empty, so reads found there are reported rather than ignored.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, replace
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
        plate: LevSeq barcode plate, 1-8.
        well: Barcode well as a 384-well position, e.g. ``"B12"``, as the
            pipeline reports it.
        name: Construct name, empty for a well expected to be empty.
        sequence: Expected variable region, uppercase; empty when empty.
        label: The barcode well as the CSV gave it, e.g. ``"A6"`` or
            ``"TR:A6"``.
        row: CSV line the well came from, for error messages.
        source: Where the sample sits, when the CSV says
            (``sample_plate``/``sample_well``), e.g. ``"1:A3"``.
    """

    plate: int
    well: str
    name: str
    sequence: str
    label: str
    row: int
    source: str = ""

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
        flank_5p: Sequence every construct shares before its variable
            region, when the CSV gave whole amplicons; empty otherwise.
        flank_3p: The same after it.
    """

    wells: dict
    layout: str
    source: Path
    notes: list = field(default_factory=list)
    flank_5p: str = ""
    flank_3p: str = ""

    @property
    def amplicons(self) -> bool:
        """Whether the flanks came from the CSV's own whole amplicons."""
        return bool(self.flank_5p and self.flank_3p)

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


#: Other names a column goes by, first match taken.  A LevSeq mapping sheet,
#: for one, gives ``bc_plate``, ``bc_well``, ``clone_plate``, ``clone_well``
#: and ``amplicon_seq``.
_ALIASES = {
    "barcode_plate": ("bc_plate", "levseq_plate"),
    "barcode_well": ("bc_well", "levseq_well"),
    "sample_plate": ("clone_plate", "source_plate"),
    "sample_well": ("clone_well", "source_well"),
    "sequence": ("amplicon_seq", "amplicon", "seq", "insert"),
    "name": ("id", "construct", "variant"),
}

#: Headers that do not say whether they mean the barcode or the sample, and
#: so are refused rather than guessed at.
_AMBIGUOUS = {"well": "barcode_well", "plate": "barcode_plate"}

# Shared ends shorter than this are not taken as flanks: a handful of bases
# common to every construct is as likely a start codon as a backbone.
MIN_SHARED_FLANK = 50


def _split_amplicons(seqs) -> Optional[tuple]:
    """``(flank_5p, flank_3p)`` shared by every sequence, or None.

    Whole amplicons drawn from one backbone share everything outside their
    variable region, so what every one of them begins and ends with is the
    flanks.  Needs at least two distinct sequences to tell flank from insert.
    """
    import os

    distinct = sorted(set(seqs))
    if len(distinct) < 2:
        return None
    pre = os.path.commonprefix(distinct)
    suf = os.path.commonprefix([s[::-1] for s in distinct])[::-1]
    # The two ends may not overlap in the shortest sequence.
    room = min(len(s) for s in distinct) - len(pre)
    suf = suf[len(suf) - min(len(suf), max(room, 0)):] if suf else ""
    if len(pre) < MIN_SHARED_FLANK or len(suf) < MIN_SHARED_FLANK:
        return None
    return pre, suf


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


#: The columns a CSV can supply, by the names :func:`read_expected_plate` uses.
FIELDS = ("barcode_plate", "barcode_well", "name", "sequence", "quadrant", "rbc",
          "sample_plate", "sample_well")


def parse_columns(text: str) -> dict:
    """``"barcode_well=Well,sequence=Seq"`` as ``{"barcode_well": "Well", ...}``.

    Raises:
        ExpectedPlateError: On an entry that is not ``field=column``, or a
            field that is not one of :data:`FIELDS`.
    """
    out = {}
    for entry in filter(None, (e.strip() for e in (text or "").split(","))):
        field_name, sep, column = (x.strip() for x in entry.partition("="))
        if not sep or not field_name or not column:
            raise ExpectedPlateError(
                f"--columns entry {entry!r} is not field=column, e.g. "
                "barcode_well=Well.")
        if field_name.lower() not in FIELDS:
            raise ExpectedPlateError(
                f"--columns: {field_name!r} is not a field; use one of "
                f"{', '.join(FIELDS)}.")
        out[field_name.lower()] = column
    return out


def read_expected_plate(path, columns: Optional[dict] = None) -> ExpectedPlate:
    """Read an expected-plate CSV.

    Args:
        path: CSV with ``barcode_well`` and ``sequence`` columns, and
            optionally ``barcode_plate`` (default 1), ``name``, ``quadrant``
            or ``rbc`` for a 96-well plate, and ``sample_plate`` /
            ``sample_well`` for where each sample sits.  The barcode columns
            are the LevSeq address the sample was barcoded with, which is
            what demultiplexing reads.  Headers are matched ignoring case and
            surrounding space; a bare ``well`` or ``plate`` is refused, since
            it does not say which of the two it means.
        columns: ``{field: CSV header}`` for a CSV that names its columns
            otherwise, e.g. ``{"barcode_well": "Well"}``.  Overrides the
            names the reader would otherwise recognise.

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
    mapped = set()
    for field_name, header in (columns or {}).items():
        match = next((h for h in reader.fieldnames if h and h.strip().lower()
                      == header.strip().lower()), None)
        if match is None:
            raise ExpectedPlateError(
                f"{path}: no column {header!r} (given for {field_name}). Found: "
                f"{', '.join(reader.fieldnames)}.")
        cols[field_name] = match
        mapped.add(match)
    for canonical, aliases in _ALIASES.items():
        if canonical not in cols:
            hit = next((a for a in aliases if a in cols), None)
            if hit:
                cols[canonical] = cols[hit]

    # A bare "well" could be where the sample was barcoded or where it sits,
    # and reading one as the other assigns every read to the wrong sample.
    for bare, explicit in _AMBIGUOUS.items():
        if bare in cols and cols[bare] not in mapped and explicit not in cols:
            other = explicit.replace("barcode_", "sample_")
            raise ExpectedPlateError(
                f"{path}: the column {cols[bare]!r} does not say which "
                f"{bare} it is. Rename it {explicit} if it is the LevSeq "
                f"{bare} the sample was barcoded in, or {other} if it is "
                f"where the sample sits (or pass --columns "
                f"{explicit}={cols[bare]})."
            )

    missing = [c for c in ("barcode_well", "sequence") if c not in cols]
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
    if not by_rbc and "barcode_plate" not in cols:
        if layout == "96":
            raise ExpectedPlateError(
                f"{path}: a quadrant column needs a barcode_plate column beside "
                "it; the quadrant says where on the LevSeq plate, not which plate."
            )
    max_row, max_col = ("H", 12) if layout == "96" else ("P", 24)

    def _get(row, col):
        return (row.get(cols[col]) or "").strip() if col in cols else ""

    wells: dict = {}
    marked = False
    for i, row in enumerate(rows, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue
        where = f"{path}, line {i}"

        m = _WELL.match(_get(row, "barcode_well"))
        if not m:
            raise ExpectedPlateError(
                f"{where}: barcode well {_get(row, 'barcode_well')!r} is not a well name.")
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
            if "barcode_plate" in cols and _get(row, "barcode_plate") and _get(row, "barcode_plate") != str(plate):
                raise ExpectedPlateError(
                    f"{where}: RB{rb:02d} is on plate {plate}, but the plate "
                    f"column says {_get(row, 'barcode_plate')}."
                )
        else:
            try:
                plate = int(_get(row, "barcode_plate") or "1") if "barcode_plate" in cols else 1
            except ValueError:
                raise ExpectedPlateError(f"{where}: plate {_get(row, 'barcode_plate')!r} is not a number.")
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
        marked = marked or any(c.isupper() for c in _get(row, "sequence"))
        src = ""
        if "sample_well" in cols and _get(row, "sample_well"):
            cp = _get(row, "sample_plate") if "sample_plate" in cols else ""
            src = f"{cp}:{_get(row, 'sample_well')}" if cp else _get(row, "sample_well")

        entry = ExpectedWell(plate=plate, well=well, name=name if seq else "",
                             sequence=seq, label=label, row=i, source=src)
        if entry.key in wells:
            first = wells[entry.key]
            raise ExpectedPlateError(
                f"{where}: barcode plate {plate} well {label} is already given "
                f"on line {first.row}; each sample needs its own barcode well."
            )
        wells[entry.key] = entry

    if not any(not w.empty for w in wells.values()):
        raise ExpectedPlateError(f"{path}: no well has an expected sequence.")

    notes = []
    flank_5p = flank_3p = ""
    # Whole amplicons, with nothing marked as the insert: what they all share
    # at either end is the backbone, and only the middle is the construct.
    if not marked:
        split = _split_amplicons([w.sequence for w in wells.values() if not w.empty])
        if split:
            flank_5p, flank_3p = split
            for key, w in list(wells.items()):
                if not w.empty:
                    wells[key] = replace(
                        w, sequence=w.sequence[len(flank_5p):len(w.sequence) - len(flank_3p)])
            mids = [len(w.sequence) for w in wells.values() if not w.empty]
            notes.append(
                f"Sequences are whole amplicons sharing a {len(flank_5p):,} bp start "
                f"and a {len(flank_3p):,} bp end; those are taken as the flanks, and "
                f"the {min(mids):,}-{max(mids):,} bp between as the variable region."
            )
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
    return ExpectedPlate(wells=wells, layout=layout, source=path, notes=notes,
                         flank_5p=flank_5p, flank_3p=flank_3p)
