"""One file describing a whole read, instead of masks and a vector separately.

Demultiplexing needs two things that both come from the construct: the
sequences flanking each barcode, so Dorado can find them, and the sequences
flanking the variable region, so consensus is called against a full-length
reference.  Supplied separately they can disagree, and masks belonging to
another backbone classify nothing while alignment still succeeds.

A read template supplies both at once — an annotated read with the three
variable spans masked out::

    >Reference_read
    ...20 bp...NNNN(24, forward barcode)...637 bp...
    NNNN(294, variable region)...1011 bp...NNNN(24, reverse barcode)...

Everything else is derived: the barcode masks are the sequence either side
of the barcode runs, and the vector flanks are the constant sequence between
the barcodes and the variable region.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Barcode spans this far outside a plausible barcode length are almost
# certainly a mis-drawn template rather than an unusual kit.
MIN_BARCODE_SPAN = 6
MAX_BARCODE_SPAN = 60
DEFAULT_MASK_LENGTH = 22

_MASKED = re.compile(r"[NnXx]+")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


class ReadTemplateError(ValueError):
    """Raised when a read template cannot be interpreted."""


def _rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass
class ReadTemplate:
    """A parsed read template.

    Attributes:
        sequence: The full template sequence, uppercased.
        flank_5p: Constant sequence between the forward barcode and the
            variable region.
        flank_3p: Constant sequence between the variable region and the
            reverse barcode.
        variable_length: Length of the masked variable span.
        masks: ``mask1_front``/``mask1_rear``/``mask2_front``/``mask2_rear``
            for the Dorado barcode arrangement.
        spans: ``(start, end)`` of the forward barcode, variable region and
            reverse barcode.
    """

    sequence: str
    flank_5p: str
    flank_3p: str
    variable_length: int
    masks: dict
    spans: tuple

    def vector_sequence(self) -> str:
        """The equivalent ``--vector-fasta`` record: flanks around the insert."""
        return f"{self.flank_5p}{'N' * self.variable_length}{self.flank_3p}"

    def describe(self) -> str:
        (f0, f1), (v0, v1), (r0, r1) = self.spans
        return (
            f"forward barcode {f1 - f0} bp at {f0}, "
            f"variable region {v1 - v0} bp at {v0}, "
            f"reverse barcode {r1 - r0} bp at {r0}"
        )


def _read_single_record(path: Path) -> str:
    """Return the sequence of a one-record FASTA."""
    seqs, current = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line.strip())
    if current:
        seqs.append("".join(current))

    if not seqs:
        raise ReadTemplateError(f"{path}: no sequence found.")
    if len(seqs) > 1:
        raise ReadTemplateError(
            f"{path}: expected one record, found {len(seqs)}. A read template "
            "describes a single read layout."
        )
    return seqs[0].upper()


def parse_read_template(
    path,
    mask_length: int = DEFAULT_MASK_LENGTH,
) -> ReadTemplate:
    """Parse a read template into barcode masks and vector flanks.

    Args:
        path: FASTA holding one record with three masked spans, in read
            order: forward barcode, variable region, reverse barcode.
        mask_length: Bases either side of a barcode to use as its mask.

    Returns:
        ReadTemplate.

    Raises:
        ReadTemplateError: If the record cannot be read, does not have
            exactly three masked spans, or those spans are implausible.
    """
    path = Path(path)
    seq = _read_single_record(path)
    runs = [(m.start(), m.end()) for m in _MASKED.finditer(seq)]

    if len(runs) != 3:
        found = ", ".join(f"{e - s} bp at {s}" for s, e in runs) or "none"
        raise ReadTemplateError(
            f"{path}: expected three masked spans (forward barcode, variable "
            f"region, reverse barcode) written as runs of N or X; found "
            f"{len(runs)} ({found}).\n"
            "A template with only the variable region masked is a "
            "--vector-fasta, not a read template."
        )

    (f0, f1), (v0, v1), (r0, r1) = runs
    for label, (s, e) in (("forward barcode", runs[0]), ("reverse barcode", runs[2])):
        span = e - s
        if not MIN_BARCODE_SPAN <= span <= MAX_BARCODE_SPAN:
            raise ReadTemplateError(
                f"{path}: the {label} span is {span} bp, outside the plausible "
                f"{MIN_BARCODE_SPAN}-{MAX_BARCODE_SPAN} bp. The three masked "
                "spans must be in read order: forward barcode, variable "
                "region, reverse barcode."
            )

    if f0 == 0:
        raise ReadTemplateError(
            f"{path}: the forward barcode starts at the first base, leaving no "
            "sequence in front of it to use as a mask."
        )
    if r1 == len(seq):
        raise ReadTemplateError(
            f"{path}: the reverse barcode ends at the last base, leaving no "
            "sequence after it to use as a mask."
        )

    masks = {
        "mask1_front": seq[max(0, f0 - mask_length):f0],
        "mask1_rear": seq[f1:f1 + mask_length],
        "mask2_front": seq[max(0, r0 - mask_length):r0],
        "mask2_rear": seq[r1:r1 + mask_length],
    }

    return ReadTemplate(
        sequence=seq,
        flank_5p=seq[f1:v0],
        flank_3p=seq[v1:r0],
        variable_length=v1 - v0,
        masks=masks,
        spans=(runs[0], runs[1], runs[2]),
    )


def write_vector_fasta(template: ReadTemplate, path) -> Path:
    """Write the template's flanks as a ``--vector-fasta`` record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">read_template_vector\n{template.vector_sequence()}\n")
    return path


def write_mask_config(template: ReadTemplate, path, source: Optional[str] = None) -> Path:
    """Write the template's barcode masks as a mask config TOML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = template.masks
    origin = f" from {source}" if source else ""
    path.write_text(
        "# Barcode mask (flanking) sequences for Dorado demultiplexing.\n"
        f"# Derived{origin} — the sequence either side of the barcode spans.\n\n"
        "[meta]\n"
        f'description = "Derived from a read template{origin}"\n\n'
        "[fbc]\n"
        f'mask1_front = "{m["mask1_front"]}"\n'
        f'mask1_rear  = "{m["mask1_rear"]}"\n'
        f'mask2_front = "{m["mask2_front"]}"\n'
        f'mask2_rear  = "{m["mask2_rear"]}"\n'
    )
    return path
