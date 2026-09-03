"""Mapping between LevSeq barcode plates and sort plates.

A sequencing run normally uses one barcode plate per sort plate, so the two
numbers agree and nothing here is needed.  They diverge when a run has more
sort plates than the kit has barcode plates: barcode plates get reused across
separate FASTQs, and the FASTQ a read came from is the only thing that says
which sort plate it belongs to.

For example, ten sort plates across two FASTQs::

    fastq 1: barcode plates 1-6  ->  sort plates 1-6
    fastq 2: barcode plates 7,8  ->  sort plates 7,8
             barcode plates 1,2  ->  sort plates 9,10

Because barcode plates 1 and 2 appear in both files, the FASTQs cannot be
concatenated before demultiplexing — each is demuxed on its own and the
results are combined afterwards on a single sort-plate numbering.

This module defines that configuration, read from a TOML file::

    [[fastq]]
    path = "run1/fastq_pass"
    plates = { 1 = 1, 2 = 2, 3 = 3, 4 = 4, 5 = 5, 6 = 6 }

    [[fastq]]
    path = "run2/fastq_pass"
    plates = { 7 = 7, 8 = 8, 1 = 9, 2 = 10 }
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import tomllib

# barcode_to_well() encodes plate and quadrant in a single reverse barcode
# index and accepts RB01-RB32, i.e. four quadrants across eight plates.
MAX_BARCODE_PLATES = 8
RBC_PER_PLATE = 4


FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


def segment_name_for(path) -> str:
    """Name a segment after its FASTQ, without the extension.

    Path.stem strips one suffix, so a ".fastq.gz" keeps its ".fastq" and the
    segment's output directory ends up named like a file.

    Args:
        path: FASTQ file or directory of FASTQs.

    Returns:
        The name with any FASTQ extension removed.
    """
    name = Path(path).name
    lowered = name.lower()
    for suffix in FASTQ_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem or name


class PlateMapError(ValueError):
    """Raised when a plate-map configuration is invalid."""


@dataclass
class Segment:
    """One FASTQ and the sort plates its barcode plates correspond to.

    Attributes:
        name: Short identifier, used for the segment's output directory.
        path: Path to the FASTQ file or directory of FASTQs.
        plates: Mapping of barcode plate number to sort plate number, both
            1-based.
    """

    name: str
    path: Path
    plates: dict[int, int] = field(default_factory=dict)
    #: Further FASTQs read together with *path*, when more than one covers the
    #: same sort plate.  Empty for a segment that stands alone.
    extra_paths: list[Path] = field(default_factory=list)

    @property
    def all_paths(self) -> list[Path]:
        """Every FASTQ this segment reads."""
        return [self.path, *self.extra_paths]

    @property
    def barcode_plates(self) -> list[int]:
        """Barcode plates this segment uses, ascending."""
        return sorted(self.plates)

    @property
    def sort_plates(self) -> list[int]:
        """Sort plates this segment produces, ascending."""
        return sorted(self.plates.values())

    @property
    def n_rbc(self) -> int:
        """Reverse barcodes needed to cover this segment's barcode plates.

        Barcodes are generated contiguously from RB01, so a segment using
        barcode plates 1, 2, 7 and 8 needs all 32 — reads landing on the
        plates it does not declare are dropped by :func:`barcode_to_well`.
        """
        return max(self.barcode_plates) * RBC_PER_PLATE

    def describe(self) -> str:
        """One-line human summary, e.g. ``barcode 1,2 -> sort 9,10``."""
        pairs = sorted(self.plates.items(), key=lambda kv: kv[1])
        bc = ",".join(str(b) for b, _ in pairs)
        sp = ",".join(str(s) for _, s in pairs)
        return f"barcode {bc} -> sort {sp}"


def identity_segment(fastq: Path, n_plates: int, name: str = "all") -> Segment:
    """Build the ordinary one-FASTQ mapping where plate numbers agree.

    Args:
        fastq: Path to the FASTQ file or directory.
        n_plates: Number of sort plates.
        name: Segment name.

    Returns:
        A single :class:`Segment` mapping each barcode plate to itself.
    """
    n = max(1, min(n_plates, MAX_BARCODE_PLATES))
    return Segment(name=name, path=Path(fastq), plates={i: i for i in range(1, n + 1)})


def _coerce_plates(raw: dict, where: str) -> dict[int, int]:
    """Convert a TOML ``plates`` table to ``{int: int}``, with validation."""
    if not isinstance(raw, dict) or not raw:
        raise PlateMapError(f"{where}: 'plates' must be a non-empty table.")

    plates: dict[int, int] = {}
    for bc_key, sort_val in raw.items():
        try:
            bc = int(str(bc_key))
        except ValueError:
            raise PlateMapError(
                f"{where}: barcode plate '{bc_key}' is not a number."
            ) from None
        if not isinstance(sort_val, int) or isinstance(sort_val, bool):
            raise PlateMapError(
                f"{where}: sort plate for barcode plate {bc} must be a number, "
                f"got {sort_val!r}."
            )
        if not 1 <= bc <= MAX_BARCODE_PLATES:
            raise PlateMapError(
                f"{where}: barcode plate {bc} is out of range — the LevSeq kit "
                f"provides {MAX_BARCODE_PLATES} barcode plates "
                f"(RB01-RB{MAX_BARCODE_PLATES * RBC_PER_PLATE:02d})."
            )
        if sort_val < 1:
            raise PlateMapError(
                f"{where}: sort plate {sort_val} must be 1 or greater."
            )
        if bc in plates:
            raise PlateMapError(
                f"{where}: barcode plate {bc} is listed twice."
            )
        plates[bc] = sort_val

    dupes = {s for s in plates.values() if list(plates.values()).count(s) > 1}
    if dupes:
        raise PlateMapError(
            f"{where}: sort plate(s) {sorted(dupes)} assigned to more than one "
            "barcode plate in the same FASTQ."
        )
    return plates


def parse_plate_map(doc: dict, base_dir: Optional[Path] = None) -> list[Segment]:
    """Build segments from an already-parsed TOML document.

    Args:
        doc: Parsed TOML mapping.
        base_dir: Directory that relative ``path`` values resolve against.

    Returns:
        List of :class:`Segment`.

    Raises:
        PlateMapError: If the configuration is malformed or inconsistent.
    """
    entries = doc.get("fastq")
    if not entries:
        raise PlateMapError(
            "No [[fastq]] entries found. Each FASTQ needs a 'path' and a "
            "'plates' table mapping barcode plate to sort plate."
        )
    if not isinstance(entries, list):
        raise PlateMapError("'fastq' must be a list of [[fastq]] tables.")

    segments: list[Segment] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"[[fastq]] #{i + 1}"
        if not isinstance(entry, dict):
            raise PlateMapError(f"{where}: expected a table.")
        raw_path = entry.get("path")
        if not raw_path:
            raise PlateMapError(f"{where}: missing 'path'.")

        path = Path(raw_path)
        if base_dir is not None and not path.is_absolute():
            # A relative path here is ambiguous: it may have been written
            # relative to the directory the run was launched from rather than
            # to this file. Prefer whichever one exists, and fall back to the
            # file-relative reading so the error names a definite path.
            from_config = Path(base_dir) / path
            path = from_config if from_config.exists() or not path.exists() \
                else path.resolve()

        name = str(entry.get("name") or segment_name_for(path)
                   or f"segment{i + 1}")
        if name in seen_names:
            raise PlateMapError(
                f"{where}: duplicate segment name '{name}' — give one of them "
                "an explicit 'name'."
            )
        seen_names.add(name)

        segments.append(
            Segment(name=name, path=path, plates=_coerce_plates(entry.get("plates"), where))
        )

    segments, pooling_notes = _pool_shared_plates(segments)
    # Carried on the list so the caller can say what was pooled: the run has
    # fewer segments than the file has FASTQ entries, and that should not be
    # something a reader has to notice for themselves.
    segments = _WithNotes(segments)
    segments.notes = pooling_notes
    return segments


def check_segment_paths(segments: list) -> list:
    """Return the segments whose FASTQ path does not exist."""
    return [seg for seg in segments if not Path(seg.path).exists()]


class _WithNotes(list):
    """A list of segments that remembers what was said while building it."""

    notes: list = []


def _pool_shared_plates(segments: list[Segment]) -> tuple[list[Segment],
                                                          list[str]]:
    """Fold segments covering the same sort plate into one.

    Two FASTQs can legitimately carry the same plate -- a re-PCR, a second run
    for depth -- and their reads belong together.  Left as separate segments
    they would each write per-well files named by sort plate and well, and the
    merged view replaces on collision, so one segment's reads would vanish
    without a word.

    Returns the folded segments and a note for each pooling performed.

    Raises:
        PlateMapError: If two segments route different barcode plates to one
            sort plate, which no pooling can resolve.
    """
    # Segments are joined when they share a sort plate, and joining is
    # transitive: a third FASTQ sharing a plate with either belongs with both.
    groups: list[list[Segment]] = []
    for seg in segments:
        touching = [g for g in groups
                    if set(g_seg_plates(g)) & set(seg.sort_plates)]
        if not touching:
            groups.append([seg])
            continue
        merged = [seg]
        for g in touching:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)

    # Groups come back in the order their first segment appeared.
    groups.sort(key=lambda g: min(segments.index(s) for s in g))

    folded: list[Segment] = []
    notes: list[str] = []
    for group in groups:
        group = sorted(group, key=segments.index)
        if len(group) == 1:
            folded.append(group[0])
            continue

        plates: dict[int, int] = {}
        for seg in group:
            for barcode, sort_plate in seg.plates.items():
                if plates.get(barcode, sort_plate) != sort_plate:
                    raise PlateMapError(
                        f"Barcode plate {barcode} maps to sort plate "
                        f"{plates[barcode]} in one FASTQ and {sort_plate} in "
                        f"'{seg.name}'. The FASTQs share a sort plate, so "
                        "their reads must be read together, and that needs "
                        "one mapping."
                    )
                plates[barcode] = sort_plate

        by_sort: dict[int, int] = {}
        for barcode, sort_plate in plates.items():
            if sort_plate in by_sort:
                raise PlateMapError(
                    f"Sort plate {sort_plate} is reached from barcode plates "
                    f"{by_sort[sort_plate]} and {barcode} across FASTQs that "
                    "share a plate. Which barcode a read carries would decide "
                    "which well it lands in, so this cannot be pooled."
                )
            by_sort[sort_plate] = barcode

        head, rest = group[0], group[1:]
        shared = sorted(set(head.sort_plates).intersection(
            *(set(s.sort_plates) for s in rest)))
        notes.append(
            f"{', '.join(s.name for s in group)} cover sort plate"
            f"{'s' if len(shared) > 1 else ''} "
            f"{', '.join(str(p) for p in shared)} between them; their reads "
            f"are pooled."
        )
        folded.append(replace(
            head,
            plates=plates,
            extra_paths=[*head.extra_paths, *(s.path for s in rest),
                         *(p for s in rest for p in s.extra_paths)],
        ))
    return folded, notes


def g_seg_plates(group: list[Segment]) -> list[int]:
    """Every sort plate a group of segments covers."""
    return [p for seg in group for p in seg.sort_plates]


def load_plate_map(path: Path) -> list[Segment]:
    """Read and validate a plate-map TOML file.

    Relative FASTQ paths resolve against the config file's directory.

    Args:
        path: Path to the TOML file.

    Returns:
        List of :class:`Segment`.

    Raises:
        PlateMapError: If the file cannot be parsed or is inconsistent.
    """
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise PlateMapError(f"{path}: invalid TOML — {exc}") from exc
    except OSError as exc:
        raise PlateMapError(f"{path}: cannot be read — {exc}") from exc

    return parse_plate_map(doc, base_dir=path.parent)


def format_plate_map_toml(segments: list[Segment]) -> str:
    """Render segments as a TOML document.

    Used to save an interactively-built mapping so the next run can pass it
    with ``--plate-map`` instead of answering the prompts again.
    """
    lines = [
        "# uSort-M plate map: which sort plate each barcode plate corresponds",
        "# to, per FASTQ. Pass with: usortm demux <project> --plate-map <file>",
        "",
    ]
    for seg in segments:
        pairs = ", ".join(
            f"{bc} = {seg.plates[bc]}" for bc in seg.barcode_plates
        )
        lines.append("[[fastq]]")
        lines.append(f'name = "{seg.name}"')
        # Absolute, because a relative path in this file resolves against the
        # file's own directory on reload -- not the directory the run was
        # launched from, which is where the path was typed.
        lines.append(f'path = "{Path(seg.path).resolve()}"')
        lines.append(f"plates = {{ {pairs} }}")
        lines.append("")
    return "\n".join(lines)


def write_plate_map(segments: list[Segment], path: Path) -> Path:
    """Write segments to *path* as TOML and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_plate_map_toml(segments))
    return path


def total_sort_plates(segments: list[Segment]) -> int:
    """Highest sort plate number across all segments."""
    return max((s for seg in segments for s in seg.sort_plates), default=0)
