"""Put a construct's annotations onto the reference a pileup is drawn against.

The tags either side of the variable region -- the SNAP-tag, the fluorescent
protein, the terminator -- are annotated in the SnapGene or GenBank file the
construct was designed in. The reference a pileup aligns to is not that file:
it is one library variant's insert with the vector flanks around it, built by
the pipeline, and it starts at some offset into the designed construct and
usually differs in length.

So the annotations have to be moved before they can be drawn. The offset is
recovered by probing rather than configured: several stretches of the pileup
reference are located in the annotated sequence, and the shift they agree on is
the one applied. Probing outside the variable region matters, since that is the
one span that genuinely differs between a designed construct and any particular
variant.

A feature that falls off either end after the shift is dropped rather than
clipped: half a tag drawn at the edge of a pileup reads as a tag that is
genuinely truncated in the construct, which would be a lie.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["load_annotations", "transfer_features", "features_for_reference"]

#: Suffixes that carry annotations.  A plain FASTA has none.
ANNOTATED_SUFFIXES = (".dna", ".gb", ".gbk", ".genbank", ".ape")


def load_annotations(path):
    """Read an annotated construct file, or return None if it has none.

    Args:
        path: A SnapGene ``.dna``, GenBank, or ApE file.

    Returns:
        A ``seqviewer.Reference``, or None if the file is missing, is not an
        annotated format, or could not be read.
    """
    import os

    if not path or not os.path.exists(path):
        return None
    if not str(path).lower().endswith(ANNOTATED_SUFFIXES):
        return None
    try:
        from seqviewer.genbank import load_reference

        return load_reference(str(path))
    except Exception as exc:
        logger.warning("Could not read annotations from %s: %s", path, exc)
        return None


def _find_offset(source_seq: str, target_seq: str, probe_len: int = 60,
                 n_probes: int = 6) -> Optional[int]:
    """How far into *source_seq* does *target_seq* start?

    Probes are taken across the whole of *target_seq* and each is looked up in
    *source_seq*.  A probe landing inside the variable region will disagree
    with the rest, or not be found at all, so the offset returned is the one
    the majority of probes agree on.  None when fewer than two agree, which is
    the signal that the two sequences are not the same construct.
    """
    if not source_seq or not target_seq:
        return None
    source = source_seq.upper()
    target = target_seq.upper()

    span = max(1, (len(target) - probe_len) // max(1, n_probes - 1))
    votes: dict = {}
    for k in range(n_probes):
        start = min(k * span, max(0, len(target) - probe_len))
        probe = target[start:start + probe_len]
        if len(probe) < probe_len:
            continue
        found = source.find(probe)
        if found < 0:
            continue
        votes[found - start] = votes.get(found - start, 0) + 1

    if not votes:
        return None
    offset, count = max(votes.items(), key=lambda kv: kv[1])
    if count < 2:
        return None
    return offset


def transfer_features(reference, target_seq: str) -> List:
    """Shift *reference*'s features onto *target_seq*'s coordinates.

    Args:
        reference: A ``seqviewer.Reference`` carrying the annotations.
        target_seq: The sequence a pileup will be drawn against.

    Returns:
        Features positioned on *target_seq*, in its coordinate frame.  Empty
        when the two sequences do not line up, when the reference has no
        features, or when nothing survives the shift.
    """
    if reference is None or not getattr(reference, "features", None):
        return []

    offset = _find_offset(reference.seq, target_seq)
    if offset is None:
        logger.info(
            "Annotated construct does not line up with the pileup reference; "
            "drawing the pileup without annotations",
        )
        return []

    out = []
    for feature in reference.features:
        if getattr(feature, "wraps_origin", False):
            continue          # a span across the origin has no place on a linear reference
        start = feature.start - offset
        end = feature.end - offset
        if start < 0 or end > len(target_seq) or end <= start:
            continue
        out.append(replace(feature, start=start, end=end))
    return out


def features_for_reference(annotation_path, target_seq: str) -> List:
    """Load *annotation_path* and place its features on *target_seq*.

    Returns an empty list for anything that does not work out, so a caller can
    treat annotations as decoration that is present when it can be.
    """
    return transfer_features(load_annotations(annotation_path), target_seq)
