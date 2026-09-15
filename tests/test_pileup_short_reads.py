"""A well whose reads all stop short of the reference midpoint still gets a
pileup that shows them.

The grid keeps only reads crossing the midpoint, which suits concatemer
split-reads.  On the sequenced pick plate a well held a construct with the
right 5' junction and a different insert: 478 reads aligned over the first
627 of 1,942 bases, none crossed the midpoint, and the page said "No aligned
reads available (0 reads unaligned)" -- the one well worth opening was the
one with nothing on it.
"""
import random
import shutil

import pandas as pd
import pytest

from usortm.demux import utils
from usortm.demux.streakout import _generate_one_pick_pileup


def _tool(name):
    finder = getattr(utils, f"find_{name}", None)
    try:
        return finder() if finder else shutil.which(name)
    except Exception:                                  # noqa: BLE001
        return shutil.which(name)


@pytest.mark.skipif(not (_tool("minimap2") and _tool("samtools")),
                    reason="minimap2 and samtools are needed")
def test_reads_that_stop_before_the_midpoint_are_still_drawn(tmp_path):
    rng = random.Random(3)
    ref = "".join(rng.choice("ACGT") for _ in range(1942))
    single = tmp_path / "single_ref_fastas"
    single.mkdir()
    (single / "G53X.fasta").write_text(f">G53X\n{ref}\n")
    # reads: the first 600 bases of the reference, then 350 bases of something
    # else -- the shape of a wrong insert behind a correct 5' junction
    tail = "".join(rng.choice("ACGT") for _ in range(350))
    reads = pd.DataFrame({
        "read_name": [f"r{i}" for i in range(12)],
        "read_seq": [ref[:600] + tail] * 12,
        "read_qual": ["I" * 950] * 12,
    })
    out = tmp_path / "well_1_J6.html"
    result = _generate_one_pick_pileup(
        well_pos="1J6", source_plate="1", source_well="J6", variant="G53X",
        reads=12, consensus_fraction=0.0, cons_check="Error", well_reads=reads,
        single_ref_dir=str(single), output_path=str(out),
        minimap2_path=_tool("minimap2"), samtools_path=_tool("samtools"),
        ref_index=None, flank_5p_len=637, flank_3p_len=1011,
    )
    assert result is not None and out.exists()
    page = out.read_text()
    assert "No aligned reads available" not in page
    assert "no read crosses the reference midpoint" in page
