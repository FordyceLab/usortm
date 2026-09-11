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

Two records fix which well holds what.  The synthesis order is already written
as a plate, so it says which construct was ordered into which well of a 96-well
plate; :func:`read_order_layout` reads it back.  The replicate map says where
each picked colony of that plate was arrayed into the 384-well coordinates a
LevSeq run reports -- a quadrant per replicate, since four 96-well plates
interleave into one 384.  The two are separate because the order is fixed when
the constructs are bought and the arraying is decided at the bench afterwards.

A provided map is still checked against the reads: :func:`infer_layout` asks
which arraying puts the constructs on wells that grew, which catches a map that
describes a different plate than the one sequenced.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Rows and columns of the 96-well plate a synthesis order is written on.
ORDER_ROWS = "ABCDEFGH"
ORDER_COLS = 12

#: Rows and columns of the 384-well plate a LevSeq run addresses.
SEQ_ROWS = "ABCDEFGHIJKLMNOP"
SEQ_COLS = 24

#: Where a 96-well plate sits inside the 384-well plate it was arrayed into.
#:
#: Four 96-well plates interleave into one 384, and a quadrant is named by the
#: 384-well its own A1 lands on -- the vocabulary a bench protocol uses.
#: ``block`` is the other case: a straight copy into the top-left 96 wells,
#: which is what moving a 96-well plate one-for-one gives.
QUADRANTS = ("A1", "A2", "B1", "B2")
LAYOUTS = ("block",) + QUADRANTS

_QUADRANT_OFFSETS = {"A1": (0, 0), "A2": (0, 1), "B1": (1, 0), "B2": (1, 1)}


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


@dataclass(frozen=True)
class Replicate:
    """One picked colony of every construct, and where it was arrayed."""

    n: int
    plate: int
    quadrant: str


@dataclass(frozen=True)
class ExpectedWell:
    """The construct and replicate intended for one sequenced well.

    Both coordinates are kept.  ``well`` is where the read came from, in the
    384-well plate the colonies were consolidated into for sequencing;
    ``order_well`` is the 96-well position the construct was ordered and
    assembled in, which is the plate the bench actually worked in and the one
    worth drawing.
    """

    plate: int
    well: str
    variant: str
    replicate: int
    order_well: str = ""


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


class ReplicateMapError(ValueError):
    """A replicate map that cannot describe a plate."""


def parse_replicate_map(doc: dict) -> List[Replicate]:
    """Build the replicate map from an already-parsed TOML document.

    Args:
        doc: Parsed TOML mapping with ``[[replicate]]`` tables, each naming
            the replicate number ``n``, the sequenced ``plate`` it went to,
            and the ``quadrant`` of that plate it occupies.

    Returns:
        One :class:`Replicate` per entry, ordered by replicate number.

    Raises:
        ReplicateMapError: If an entry is malformed, a replicate number
            repeats, or two replicates claim one quadrant of one plate --
            which would put two colonies in the same well.
    """
    entries = doc.get("replicate")
    if not entries:
        raise ReplicateMapError(
            "No [[replicate]] entries found. Each replicate needs 'n', "
            "'plate' and 'quadrant'."
        )
    if not isinstance(entries, list):
        raise ReplicateMapError("'replicate' must be a list of tables.")

    out: List[Replicate] = []
    seen_n: Dict[int, int] = {}
    seen_slot: Dict[Tuple[int, str], int] = {}
    for i, entry in enumerate(entries):
        where = f"[[replicate]] #{i + 1}"
        if not isinstance(entry, dict):
            raise ReplicateMapError(f"{where} is not a table.")
        try:
            n = int(entry["n"])
            plate = int(entry["plate"])
        except KeyError as exc:
            raise ReplicateMapError(f"{where} is missing {exc.args[0]!r}.")
        except (TypeError, ValueError):
            raise ReplicateMapError(f"{where}: 'n' and 'plate' must be whole "
                                    f"numbers.")
        quadrant = str(entry.get("quadrant", "")).strip().upper()
        if quadrant not in QUADRANTS:
            raise ReplicateMapError(
                f"{where}: quadrant {entry.get('quadrant')!r} is not one of "
                f"{', '.join(QUADRANTS)}."
            )
        if n in seen_n:
            raise ReplicateMapError(
                f"{where}: replicate {n} is already defined by "
                f"[[replicate]] #{seen_n[n]}."
            )
        slot = (plate, quadrant)
        if slot in seen_slot:
            raise ReplicateMapError(
                f"{where}: plate {plate} quadrant {quadrant} is already taken "
                f"by [[replicate]] #{seen_slot[slot]}; two replicates there "
                f"would share every well."
            )
        seen_n[n] = i + 1
        seen_slot[slot] = i + 1
        out.append(Replicate(n=n, plate=plate, quadrant=quadrant))
    return sorted(out, key=lambda r: r.n)


def read_replicate_map(path) -> List[Replicate]:
    """Read the replicate map from a TOML file.

    Args:
        path: The map file, as passed to ``usortm demux --replicate-map``.

    Returns:
        One :class:`Replicate` per entry.

    Raises:
        ReplicateMapError: If the file is not valid TOML, or does not describe
            a plate.
    """
    try:
        import tomllib
    except ModuleNotFoundError:            # pragma: no cover - Python < 3.11
        import tomli as tomllib
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ReplicateMapError(f"{path} is not valid TOML: {exc}")
    return parse_replicate_map(doc)


def single_replicate(plate: int = 1, quadrant: str = "block") -> List[Replicate]:
    """The map for a plate picked once per construct rather than in replicate.

    Args:
        plate: The sequenced plate the constructs went to.
        quadrant: Where the 96-well plate sits in it, or ``block`` for a
            one-for-one copy.
    """
    return [Replicate(n=1, plate=plate, quadrant=quadrant)]


def expected_wells(order: Iterable[OrderedWell],
                   replicates: Sequence[Replicate],
                   plate_of: Optional[Dict[int, int]] = None
                   ) -> Dict[Tuple[int, str], ExpectedWell]:
    """The construct and replicate intended for each sequenced well.

    Args:
        order: The ordered layout, from :func:`read_order_layout`.
        replicates: Where each picked colony was arrayed.
        plate_of: Maps an order plate number to the sequenced plate it became,
            for an order spanning several plates.  The replicate's own plate
            is used when this is not given.

    Returns:
        ``{(plate, well): ExpectedWell}`` in the coordinates the demux reports.

    Raises:
        ReplicateMapError: If two constructs would land in one well, which
            means the order and the map disagree about the plate.
    """
    out: Dict[Tuple[int, str], ExpectedWell] = {}
    for rep in replicates:
        for item in order:
            plate = (plate_of or {}).get(item.plate, rep.plate)
            layout = rep.quadrant if rep.quadrant in LAYOUTS else "block"
            key = (plate, to_seq_well(item.well, layout))
            if key in out:
                clash = out[key]
                raise ReplicateMapError(
                    f"plate {plate} well {key[1]} is claimed by both "
                    f"{clash.variant} (replicate {clash.replicate}) and "
                    f"{item.variant} (replicate {rep.n})."
                )
            out[key] = ExpectedWell(plate=plate, well=key[1],
                                    variant=item.variant, replicate=rep.n,
                                    order_well=item.well)
    return out


def bench_positions(order: Iterable[OrderedWell],
                    replicates: Sequence[Replicate],
                    plate_of: Optional[Dict[int, int]] = None
                    ) -> Dict[Tuple[int, str], Tuple[int, str]]:
    """Where each sequenced well was handled: the colony plate and its well.

    The inverse of the arraying.  A re-ordered construct is assembled in a
    96-well plate and its picked colonies are consolidated into a 384-well
    plate for sequencing; a pick from that round is made from the 96-well
    colony plates, so the robot needs the plate a colony was worked in and
    the well it sits in there, not the position it was sequenced at.

    Returns ``{(sequenced plate, sequenced well): (replicate, order well)}``
    -- the replicate number is the colony plate.
    """
    return {key: (e.replicate, e.order_well)
            for key, e in expected_wells(order, replicates, plate_of).items()}


def bench_positions_for_round(project_dir, round_num: int
                              ) -> Dict[Tuple[int, str], Tuple[int, str]]:
    """:func:`bench_positions` for a round of a project, from its files.

    Empty when the round does not describe an arraying: a first round has
    none, and a re-order round without a replicate map cannot be mapped
    back.  Reads ``rounds/<n>/replicate_map.toml`` and the order file under
    ``reorder/``.
    """
    from pathlib import Path

    root = Path(project_dir)
    round_dir = root if round_num == 1 else root / "rounds" / str(round_num)
    rep_file = round_dir / "replicate_map.toml"
    orders = sorted(root.glob("reorder/reorder_*.csv"))
    if not (rep_file.exists() and orders):
        return {}
    return bench_positions(read_order_layout(orders[0]),
                           read_replicate_map(rep_file))


def infer_layout(order: Iterable[OrderedWell], well_data: Sequence[dict],
                 min_reads: int = 20, plate: int = 1) -> dict:
    """Read one replicate's arraying off the data rather than being told it.

    Each candidate quadrant puts the ordered constructs in different wells, and
    only the right one lines up with where the reads landed.  Scoring is the
    share of intended wells that grew: the right quadrant puts nearly every
    construct on a well with reads, a wrong one scatters them over wells that
    were never filled.

    A plate where every quadrant grew cannot separate them, since each
    candidate then scores well.  The margin over the runner-up says whether the
    answer means anything and is returned rather than resolved here.  This is
    the check on a provided map rather than a replacement for it: with three
    replicates every quadrant is occupied, and the margin correctly collapses.

    Args:
        order: The ordered layout.
        well_data: Per-well rows from the demux, needing ``plate``, ``well``
            and ``reads``.
        min_reads: Reads a well needs before it counts as grown.
        plate: The sequenced plate to score against.

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
        wanted = expected_wells(order, [Replicate(1, plate, candidate)])
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
    """One intended well, and what the reads made of it.

    ``well`` is the sequenced 384-well position; ``order_well`` the 96-well
    one the construct was assembled in.
    """

    plate: int
    well: str
    expected: str
    observed: Optional[str]
    reads: int
    status: str
    replicate: int = 1
    order_well: str = ""


def verify(well_data: Sequence[dict],
           expected: Dict[Tuple[int, str], ExpectedWell],
           designed: set, min_reads: int = 20,
           is_clean=None) -> List[WellVerdict]:
    """Judge each intended well against what was sequenced in it.

    Args:
        well_data: Per-well rows from the demux.
        expected: ``{(plate, well): ExpectedWell}``, from
            :func:`expected_wells`.
        designed: The library's members, used to judge whether a well was read
            cleanly enough for its call to stand.
        min_reads: Reads a well needs before its call is used at all.
        is_clean: Predicate taking a well row and *designed*, deciding whether
            the well holds its assigned member cleanly.  Defaults to the rule
            the plate maps flag on, so a well flagged there is never counted
            confirmed here.

    Returns:
        One :class:`WellVerdict` per intended well, ordered by plate, replicate
        and well.  Wells that were sequenced but never ordered are left out;
        they are not part of the question this asks.
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
            out.append(WellVerdict(plate, well, want.variant, None, reads,
                                   EMPTY, want.replicate, want.order_well))
            continue
        got = row.get("variant") or ""
        status = (CONFIRMED if got == want.variant and is_clean(row, designed)
                  else WRONG)
        out.append(WellVerdict(plate, well, want.variant, got, reads, status,
                               want.replicate, want.order_well))
    return sorted(out, key=lambda v: (v.plate, v.replicate,
                                      _split_well(v.well)))


#: Why a well did not confirm its order, most decisive first.  A well can
#: fail several tests at once -- an empty vector also reads as a different
#: variant -- so the reason is the first that applies rather than a list, and
#: the order is by what a bench response would be: nothing grew, nothing
#: aligned, the assembly gave the backbone, two templates are present, the
#: junction is wrong, and only then the sequence itself.
FAILURE_REASONS = ("no reads", "no alignment", "parent", "mixed template",
                   "flank mismatch", "another variant", "sequence differs")


def failure_reason(row: Optional[dict], verdict: "WellVerdict",
                   mixed_threshold: float = 0.25) -> Optional[str]:
    """Why a well did not hold the construct ordered for it.

    Returns None for a confirmed well.  The reasons are separated because they
    send you to different places: a plate of parents is an assembly problem, a
    plate of flank mismatches a cloning one, and a plate of empty wells a
    growth or picking one.

    Args:
        row: The well's demux record, or None if it was never sequenced.
        verdict: The well's :class:`WellVerdict`.
        mixed_threshold: Worst-column disagreement past which a well holds
            more than one template.

    Returns:
        One of :data:`FAILURE_REASONS`, or None.
    """
    if verdict.status == CONFIRMED:
        return None
    if row is None or verdict.status == EMPTY:
        return "no reads"
    if (row.get("flank_check") or "") == "No alignment" or (
            row.get("consensus_fraction") or 0) <= 0:
        return "no alignment"
    observed = row.get("variant") or ""
    if observed == "Parent":
        return "parent"
    worst = row.get("max_mismatch_frac")
    if worst is not None and worst == worst and float(worst) > mixed_threshold:
        return "mixed template"
    if (row.get("flank_check") or "OK") != "OK":
        return "flank mismatch"
    if observed != verdict.expected:
        return "another variant"
    return "sequence differs"


def summarise(verdicts: Sequence[WellVerdict]) -> dict:
    """Counts by outcome, by replicate, and the variants each accounts for.

    A construct picked in triplicate is recovered when any one of its colonies
    holds it, so the variant counts are not the well counts.  The per-replicate
    counts are the reason to pick in triplicate at all: a replicate that fails
    far more often than the others points at the picking or the plate rather
    than at the constructs.

    Returns:
        ``n_wells`` and ``wells`` by status; ``by_replicate`` giving the same
        counts per replicate; ``n_variants`` ordered, and the sets
        ``confirmed`` and ``not_confirmed``.
    """
    def _tally():
        return {CONFIRMED: 0, WRONG: 0, EMPTY: 0}

    wells = _tally()
    by_replicate: Dict[int, dict] = {}
    for v in verdicts:
        wells[v.status] = wells.get(v.status, 0) + 1
        rep = by_replicate.setdefault(v.replicate, _tally())
        rep[v.status] = rep.get(v.status, 0) + 1

    ordered = {v.expected for v in verdicts}
    confirmed = {v.expected for v in verdicts if v.status == CONFIRMED}
    return {
        "n_wells": len(verdicts),
        "wells": wells,
        "by_replicate": dict(sorted(by_replicate.items())),
        "n_variants": len(ordered),
        "confirmed": confirmed,
        "not_confirmed": ordered - confirmed,
    }
