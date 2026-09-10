"""Positions where a change is the chemistry's, not the construct's.

A position can disagree in every well of a run at a fraction that does not
vary with what the well holds.  A change belonging to a construct appears in
that construct's wells and nowhere else; one that appears everywhere,
whatever the well holds, is a property of the sequencing of that library.

Left in, such a position is counted as reads disagreeing with the reference,
and one sitting just above the 10% mark at which a position is flagged marks
a large share of the plate as worth checking and buries the wells that are.

Which positions these are is a property of a particular library and its
preparation, not of the method, so this module holds no positions of its
own: a project states its own in a ``qc_mask.toml`` beside its other
configuration, and a round may state different ones.  A mask carried from one
library to another would forgive changes that library never had.

Masking one is a claim about the chemistry, so it is written down rather than
buried in a threshold: the file names the position, the base, and the evidence
it rests on.

The claim is deliberately narrow.  Only the named base at the named position
is forgiven, since an artefact explains one substitution and not the position,
and only where it is at most half the reads, since an artefact is a minority
signal.  A well whose reads carry the masked base throughout carries it in its
construct, and forgiving that would report a real substitution as a clean
position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

#: What the file is called when it sits in a project's config directory.
QC_MASK_FILE = "qc_mask.toml"

_BASES = ("A", "C", "G", "T")


class QCMaskError(ValueError):
    """A mask file that cannot be applied as written."""


@dataclass(frozen=True)
class MaskedChange:
    """One substitution forgiven at one position.

    ``position`` is 1-based, as a sequence is read and as the file states it.
    """

    position: int
    base: str
    note: str = ""


def parse_qc_mask(doc: dict) -> List[MaskedChange]:
    """Build the mask from an already-parsed TOML document.

    Args:
        doc: Parsed TOML with ``[[mask]]`` tables, each naming a 1-based
            ``position``, the ``base`` read there, and optionally a ``note``
            saying on what evidence.

    Returns:
        One :class:`MaskedChange` per entry, ordered by position.

    Raises:
        QCMaskError: If an entry is malformed, or the same substitution is
            named twice -- which means the file disagrees with itself about
            what is being claimed.
    """
    entries = doc.get("mask")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise QCMaskError("'mask' must be a list of [[mask]] tables.")

    out: List[MaskedChange] = []
    seen: Dict[tuple, int] = {}
    for i, entry in enumerate(entries):
        where = f"[[mask]] #{i + 1}"
        if not isinstance(entry, dict):
            raise QCMaskError(f"{where} is not a table.")
        try:
            position = int(entry["position"])
        except KeyError:
            raise QCMaskError(f"{where} is missing 'position'.")
        except (TypeError, ValueError):
            raise QCMaskError(f"{where}: position must be a whole number.")
        if position < 1:
            raise QCMaskError(
                f"{where}: position {position} is not 1-based; the first base "
                f"of the reference is 1."
            )
        base = str(entry.get("base", "")).strip().upper()
        if base not in _BASES:
            raise QCMaskError(
                f"{where}: base {entry.get('base')!r} is not one of "
                f"{', '.join(_BASES)}."
            )
        key = (position, base)
        if key in seen:
            raise QCMaskError(
                f"{where}: {base} at position {position} is already masked by "
                f"[[mask]] #{seen[key]}."
            )
        seen[key] = i + 1
        out.append(MaskedChange(position=position, base=base,
                                note=str(entry.get("note", "")).strip()))
    return sorted(out, key=lambda m: (m.position, m.base))


def read_qc_mask(path) -> List[MaskedChange]:
    """Read the mask from a TOML file.

    Raises:
        QCMaskError: If the file is not valid TOML or cannot be applied.
    """
    try:
        import tomllib
    except ModuleNotFoundError:            # pragma: no cover - Python < 3.11
        import tomli as tomllib
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise QCMaskError(f"{path} is not valid TOML: {exc}")
    return parse_qc_mask(doc)


def find_qc_mask(project_dir, round_num: int = 1) -> Optional[str]:
    """The mask a project keeps, if it keeps one.

    A round may carry its own, since a re-order round is a different
    preparation and need not share the first round's artefacts; without one it
    falls back to the project's.  Both are looked up the way every other
    configuration file is, so either layout of the project works.
    """
    from pathlib import Path

    from usortm.paths import config_file

    root = Path(project_dir)
    candidates = []
    if round_num and round_num > 1:
        candidates.append(config_file(root / "rounds" / str(round_num),
                                      QC_MASK_FILE))
    candidates.append(config_file(root, QC_MASK_FILE))
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def as_lookup(mask: List[MaskedChange]) -> Dict[int, Set[str]]:
    """The mask as ``{0-based position: {bases}}``, for the column scan.

    Zero-based because that is what a BAM's reference positions are; the file
    is 1-based because that is how a sequence is read.  Converting once here
    keeps the two apart.
    """
    out: Dict[int, Set[str]] = {}
    for item in mask:
        out.setdefault(item.position - 1, set()).add(item.base)
    return out


def describe(mask: List[MaskedChange]) -> str:
    """One line naming what is masked, for a console line or a report."""
    if not mask:
        return ""
    return ", ".join(f"{m.base} at {m.position}" for m in mask)
