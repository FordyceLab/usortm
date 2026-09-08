"""Check a re-ordered plate against what was ordered for it.

A first round discovers what each well holds: cells are sorted from a library,
so a well's variant is whatever its reads say.  A re-order round is the other
way round.  The dropouts are synthesised as separate constructs, assembled in
parallel and picked into known wells, so every well has an intended variant
before it is sequenced and the question is whether the assembly produced it.

That makes the same reads answer a different question, and the outcomes it
separates are not the ones a first round has: a well can hold what was ordered
for it, hold something else, or hold nothing that grew.  Only the first adds a
variant to the library, and the other two fail for different reasons -- a wrong
sequence is an assembly or cloning problem, an empty well a growth or picking
one -- so they are counted apart rather than together as "not recovered".

The ordered layout comes from the synthesis order itself, which is written as a
plate: :func:`read_order_layout` reads it back.  Where a 96-well order plate
lands in the 384-well coordinates a LevSeq run reports is a property of how the
cultures were arrayed rather than of the order, so it is stated separately and
can be read off the data with :func:`infer_layout`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Rows and columns of the 96-well plate a synthesis order is written on.
ORDER_ROWS = "ABCDEFGH"
ORDER_COLS = 12

#: Rows and columns of the 384-well plate a LevSeq run addresses.  Four
#: 96-well plates interleave into one of these, which is why a quadrant needs
#: naming.
SEQ_ROWS = "ABCDEFGHIJKLMNOP"
SEQ_COLS = 24

#: How a 96-well order plate was arrayed into the sequenced plate.
#:
#: ``block`` is a straight copy into the top-left 96 wells, which is what
#: transferring a 96-well culture plate one-for-one gives.  ``q1`` to ``q4``
#: are the interleaved quadrants of a 384-well plate, which is what a 96-to-384
#: head produces; the quadrant is set by the offset the head was placed at.
LAYOUTS = ("block", "q1", "q2", "q3", "q4")

_QUADRANT_OFFSETS = {"q1": (0, 0), "q2": (0, 1), "q3": (1, 0), "q4": (1, 1)}


def _split_well(well: str) -> Tuple[int, int]:
    """Row and column indices, zero-based, of a well label like ``B7``."""
    text = str(well).strip().upper()
    if len(text) < 2 or not text[0].isalpha() or not text[1:].isdigit():
        raise ValueError(f"not a well label: {well!r}")
    if text[0] not in SEQ_ROWS:
        raise ValueError(f"row {text[0]!r} is outside a 384-well plate")
    return SEQ_ROWS.index(text[0]), int(text[1:]) - 1


def to_seq_well(well: str, layout: str = "block") -> str:
    """Where an order plate's *well* lands in 384-well coordinates.

    Args:
        well: A 96-well position, ``A1`` to ``H12``.
        layout: One of :data:`LAYOUTS`.

    Returns:
        The 384-well position, ``A1`` to ``P24``.

    Raises:
        ValueError: If *well* is outside a 96-well plate, or *layout* is not
            one of :data:`LAYOUTS`.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    row, col = _split_well(well)
    if row >= len(ORDER_ROWS) or col >= ORDER_COLS:
        raise ValueError(f"{well} is outside a 96-well plate")
    if layout == "block":
        return f"{SEQ_ROWS[row]}{col + 1}"
    d_row, d_col = _QUADRANT_OFFSETS[layout]
    return f"{SEQ_ROWS[2 * row + d_row]}{2 * col + d_col + 1}"


@dataclass(frozen=True)
class OrderedWell:
    """One construct as it was ordered: a plate, a well on it, and a name."""

    plate: int
    well: str
    variant: str


def read_order_layout(path) -> List[OrderedWell]:
    """Read the plate a synthesis order was written on.

    The order file is the upload the vendor takes, so it is also the record of
    which well each construct was ordered into.  Plates are separated by a
    blank line and a ``# Plate N`` marker, and each carries its own header row.

    Args:
        path: The order CSV, as written by ``usortm reorder``.

    Returns:
        One :class:`OrderedWell` per construct, in file order.

    Raises:
        ValueError: If no constructs are found, which means the file is not an
            order plate and reading on would invent a layout.
    """
    out: List[OrderedWell] = []
    plate = 1
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            first = row[0].strip()
            if first.startswith("#"):
                # "# Plate 2" and the like; the number is the plate's own.
                digits = "".join(c for c in first if c.isdigit())
                plate = int(digits) if digits else plate + 1
                continue
            if first.lower().startswith("well"):
                continue
            if len(row) < 2 or not row[1].strip():
                continue
            out.append(OrderedWell(plate=plate, well=first,
                                   variant=row[1].strip()))
    if not out:
        raise ValueError(f"no ordered constructs found in {path}")
    return out


def expected_wells(order: Iterable[OrderedWell], layout: str = "block",
                   plate_of: Optional[Dict[int, int]] = None
                   ) -> Dict[Tuple[int, str], str]:
    """The variant intended for each sequenced well.

    Args:
        order: The ordered layout, from :func:`read_order_layout`.
        layout: How the order plate was arrayed; one of :data:`LAYOUTS`.
        plate_of: Maps an order plate number to the sequenced plate number it
            became.  Defaults to the identity, which holds when the order
            plates were sequenced in the order they were made.

    Returns:
        ``{(plate, well): variant}`` in the coordinates the demux reports.
    """
    out: Dict[Tuple[int, str], str] = {}
    for item in order:
        plate = (plate_of or {}).get(item.plate, item.plate)
        out[(plate, to_seq_well(item.well, layout))] = item.variant
    return out


def infer_layout(order: Iterable[OrderedWell], well_data: Sequence[dict],
                 min_reads: int = 20,
                 plate_of: Optional[Dict[int, int]] = None) -> dict:
    """Read the arraying off the data rather than being told it.

    Each candidate layout puts the ordered constructs in different wells, and
    only one lines up with where the reads landed.  Scoring is the share of
    intended wells that grew: the right layout puts nearly every construct on a
    well with reads, a wrong one scatters them over wells that were never
    filled.

    A run where most of the plate grew cannot separate the layouts, since every
    candidate then scores well.  The margin over the runner-up says whether the
    answer means anything and is returned rather than resolved here.

    Args:
        order: The ordered layout.
        well_data: Per-well rows from the demux, needing ``plate``, ``well``
            and ``reads``.
        min_reads: Reads a well needs before it counts as grown.
        plate_of: As for :func:`expected_wells`.

    Returns:
        ``layout`` (the best candidate), ``score`` (its share of intended wells
        that grew), ``margin`` (its score less the runner-up's), and ``scores``
        for every candidate.
    """
    grew = set()
    for w in well_data:
        if (w.get("reads") or 0) >= min_reads:
            grew.add((int(w["plate"]), str(w["well"]).strip().upper()))

    order = list(order)
    scores = {}
    for candidate in LAYOUTS:
        wanted = expected_wells(order, candidate, plate_of)
        hit = sum(1 for key in wanted if key in grew)
        scores[candidate] = hit / len(wanted) if wanted else 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "layout": ranked[0][0],
        "score": ranked[0][1],
        "margin": ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
        "scores": scores,
    }


#: A well holds what was ordered for it.
CONFIRMED = "confirmed"
#: A well grew and was read, but not as the construct ordered for it.
WRONG = "wrong"
#: A well returned too few reads to call.
EMPTY = "empty"


@dataclass(frozen=True)
class WellVerdict:
    """One intended well, and what the reads made of it."""

    plate: int
    well: str
    expected: str
    observed: Optional[str]
    reads: int
    status: str


def verify(well_data: Sequence[dict], expected: Dict[Tuple[int, str], str],
           designed: set, min_reads: int = 20,
           is_clean=None) -> List[WellVerdict]:
    """Judge each intended well against what was sequenced in it.

    Args:
        well_data: Per-well rows from the demux.
        expected: ``{(plate, well): variant}``, from :func:`expected_wells`.
        designed: The library's members, used to judge whether a well was read
            cleanly enough for its call to stand.
        min_reads: Reads a well needs before its call is used at all.
        is_clean: Predicate taking a well row and *designed*, deciding whether
            the well holds its assigned member cleanly.  Defaults to the rule
            the plate maps flag on, so a well flagged there is never counted
            confirmed here.

    Returns:
        One :class:`WellVerdict` per intended well, ordered by plate and well.
        Wells that were sequenced but never ordered are left out; they are not
        part of the question this asks.
    """
    if is_clean is None:
        from usortm.report.plates import carries_designed_sequence
        is_clean = carries_designed_sequence

    seen = {(int(w["plate"]), str(w["well"]).strip().upper()): w
            for w in well_data}
    out: List[WellVerdict] = []
    for (plate, well), want in expected.items():
        row = seen.get((plate, well))
        reads = int((row or {}).get("reads") or 0)
        if row is None or reads < min_reads:
            out.append(WellVerdict(plate, well, want, None, reads, EMPTY))
            continue
        got = row.get("variant") or ""
        status = (CONFIRMED if got == want and is_clean(row, designed)
                  else WRONG)
        out.append(WellVerdict(plate, well, want, got, reads, status))
    return sorted(out, key=lambda v: (v.plate, _split_well(v.well)))


def summarise(verdicts: Sequence[WellVerdict]) -> dict:
    """Counts by outcome, and the variants each outcome accounts for.

    A variant ordered into more than one well is confirmed when any of its
    wells holds it, so the variant counts are not the well counts.

    Returns:
        ``n_wells`` and ``wells`` by status; ``n_variants`` ordered, and the
        sets ``confirmed`` and ``not_confirmed``.
    """
    wells = {CONFIRMED: 0, WRONG: 0, EMPTY: 0}
    for v in verdicts:
        wells[v.status] = wells.get(v.status, 0) + 1

    ordered = {v.expected for v in verdicts}
    confirmed = {v.expected for v in verdicts if v.status == CONFIRMED}
    return {
        "n_wells": len(verdicts),
        "wells": wells,
        "n_variants": len(ordered),
        "confirmed": confirmed,
        "not_confirmed": ordered - confirmed,
    }
