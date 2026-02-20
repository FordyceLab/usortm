"""LevSeq barcode sequences and Dorado configuration file generation.

Contains the 96 forward (NB) and 96 reverse (RB) barcode sequences from
the LevSeq protocol (Oxford Nanopore native barcoding kit), along with
functions to generate the Dorado TOML and FASTA config files needed for
demultiplexing.

Barcode sequences sourced from:
    https://github.com/fhalab/LevSeq  (minion_barcodes.fasta)
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# LevSeq forward barcodes (NB01-NB96), 24 nucleotides each
# ---------------------------------------------------------------------------
LEVSEQ_FBC = [
    "CACAAAGACACCGACAACTTTCTT",  # NB01
    "ACAGACGACTACAAACGGAATCGA",  # NB02
    "CCTGGTAACTGGGACACAAGACTC",  # NB03
    "TAGGGAAACACGATAGAATCCGAA",  # NB04
    "AAGGTTACACAAACCCTGGACAAG",  # NB05
    "GACTACTTTCTGCCTTTGCGAGAA",  # NB06
    "AAGGATTCATTCCCACGGTAACAC",  # NB07
    "ACGTAACTTGGTTTGTTCCCTGAA",  # NB08
    "AACCAAGACTCGCTGTGCCTAGTT",  # NB09
    "GAGAGGACAAAGGTTTCAACGCTT",  # NB10
    "TCCATTCCCTCCGATAGATGAAAC",  # NB11
    "TCCGATTCTGCTTCTTTCTACCTG",  # NB12
    "AGAACGACTTCCATACTCGTGTGA",  # NB13
    "AACGAGTCTCTTGGGACCCATAGA",  # NB14
    "AGGTCTACCTCGCTAACACCACTG",  # NB15
    "CGTCAACTGACAGTGGTTCGTACT",  # NB16
    "ACCCTCCAGGAAAGTACCTCTGAT",  # NB17
    "CCAAACCCAACAACCTAGATAGGC",  # NB18
    "GTTCCTCGTGCAGTGTCAAGAGAT",  # NB19
    "TTGCGTCCTGTTACGAGAACTCAT",  # NB20
    "GAGCCTCTCATTGTCCGTTCTCTA",  # NB21
    "ACCACTGCCATGTATCAAAGTACG",  # NB22
    "CTTACTACCCAGTGAACCTCCTCG",  # NB23
    "GCATAGTTCTGCATGATGGGTTAG",  # NB24
    "GTAAGTTGGGTATGCAACGCAATG",  # NB25
    "CATACAGCGACTACGCATTCTCAT",  # NB26
    "CGACGGTTAGATTCACCTCTTACA",  # NB27
    "TGAAACCTAAGAAGGCACCGTATC",  # NB28
    "CTAGACACCTTGGGTTGACAGACC",  # NB29
    "TCAGTGAGGATCTACTTCGACCCA",  # NB30
    "TGCGTACAGCAATCAGTTACATTG",  # NB31
    "CCAGTAGAAGTCCGACAACGTCAT",  # NB32
    "CAGACTTGGTACGGTTGGGTAACT",  # NB33
    "GGACGAAGAACTCAAGTCAAAGGC",  # NB34
    "CTACTTACGAAGCTGAGGGACTGC",  # NB35
    "ATGTCCCAGTTAGAGGAGGAAACA",  # NB36
    "GCTTGCGATTGATGCTTAGTATCA",  # NB37
    "ACCACAGGAGGACGATACAGAGAA",  # NB38
    "CCACAGTGTCAACTAGAGCCTCTC",  # NB39
    "TAGTTTGGATGACCAAGGATAGCC",  # NB40
    "GGAGTTCGTCCAGAGAAGTACACG",  # NB41
    "CTACGTGTAAGGCATACCTGCCAG",  # NB42
    "CTTTCGTTGTTGACTCGACGGTAG",  # NB43
    "AGTAGAAAGGGTTCCTTCCCACTC",  # NB44
    "GATCCAACAGAGATGCCTTCAGTG",  # NB45
    "GCTGTGTTCCACTTCATTCTCCTG",  # NB46
    "GTGCAACTTTCCCACAGGTAGTTC",  # NB47
    "CATCTGGAACGTGGTACACCTGTA",  # NB48
    "ACTGGTGCAGCTTTGAACATCTAG",  # NB49
    "ATGGACTTTGGTAACTTCCTGCGT",  # NB50
    "GTTGAATGAGCCTACTGGGTCCTC",  # NB51
    "TGAGAGACAAGATTGTTCGTGGAC",  # NB52
    "AGATTCAGACCGTCTCATGCAAAG",  # NB53
    "CAAGAGCTTTGACTAAGGAGCATG",  # NB54
    "TGGAAGATGAGACCCTGATCTACG",  # NB55
    "TCACTACTCAACAGGTGGCATGAA",  # NB56
    "GCTAGGTCAATCTCCTTCGGAAGT",  # NB57
    "CAGGTTACTCCTCCGTGAGTCTGA",  # NB58
    "TCAATCAAGAAGGGAAAGCAAGGT",  # NB59
    "CATGTTCAACCAAGGCTTCTATGG",  # NB60
    "AGAGGGTACTATGTGCCTCAGCAC",  # NB61
    "CACCCACACTTACTTCAGGACGTA",  # NB62
    "TTCTGAAGTTCCTGGGTCTTGAAC",  # NB63
    "GACAGACACCGTTCATCGACTTTC",  # NB64
    "TTCTCAGTCTTCCTCCAGACAAGG",  # NB65
    "CCGATCCTTGTGGCTTCTAACTTC",  # NB66
    "GTTTGTCATACTCGTGTGCTCACC",  # NB67
    "GAATCTAAGCAAACACGAAGGTGG",  # NB68
    "TACAGTCCGAGCCTCATGTGATCT",  # NB69
    "ACCGAGATCCTACGAATGGAGTGT",  # NB70
    "CCTGGGAGCATCAGGTAGTAACAG",  # NB71
    "TAGCTGACTGTCTTCCATACCGAC",  # NB72
    "AAGAAACAGGATGACAGAACCCTC",  # NB73
    "TACAAGCATCCCAACACTTCCACT",  # NB74
    "GACCATTGTGATGAACCCTGTTGT",  # NB75
    "ATGCTTGTTACATCAACCCTGGAC",  # NB76
    "CGACCTGTTTCTCAGGGATACAAC",  # NB77
    "AACAACCGAACCTTTGAATCAGAA",  # NB78
    "TCTCGGAGATAGTTCTCACTGCTG",  # NB79
    "CGGATGAACATAGGATAGCGATTC",  # NB80
    "CCTCATCTTGTGAAGTTGTTTCGG",  # NB81
    "ACGGTATGTCGAGTTCCAGGACTA",  # NB82
    "TGGCTTGATCTAGGTAAGGTCGAA",  # NB83
    "GTAGTGGACCTAGAACCTGTGCCA",  # NB84
    "AACGGAGGAGTTAGTTGGATGATC",  # NB85
    "AGGTGATCCCAACAAGCGTAAGTA",  # NB86
    "TACATGCTCCTGTTGTTAGGGAGG",  # NB87
    "TCTTCTACTACCGATCCGAAGCAG",  # NB88
    "ACAGCATCAATGTTTGGCTAGTTG",  # NB89
    "GATGTAGAGGGTACGGTTTGAGGC",  # NB90
    "GGCTCCATAGGAACTCACGCTACT",  # NB91
    "TTGTGAGTGGAAAGATACAGGACC",  # NB92
    "AGTTTCCATCACTTCAGACTTGGG",  # NB93
    "GATTGTCCTCAAACTGCCACCTAC",  # NB94
    "CCTGTCTGGAAGAAGAATGGACTT",  # NB95
    "CTGAACGGTCATAGAGTCCACCAT",  # NB96
]

# ---------------------------------------------------------------------------
# LevSeq reverse barcodes (RB01-RB96), 24 nucleotides each
# RB01-RB12 are unique sequences; RB13-RB96 mirror NB13-NB96.
# ---------------------------------------------------------------------------
LEVSEQ_RBC = [
    "AAGAAAGTTGTCGGTGTCTTTGTG",  # RB01
    "TCGATTCCGTTTGTAGTCGTCTGT",  # RB02
    "GAGTCTTGTGTCCCAGTTACCAGG",  # RB03
    "TTCGGATTCTATCGTGTTTCCCTA",  # RB04
    "CTTGTCCAGGGTTTGTGTAACCTT",  # RB05
    "TTCTCGCAAAGGCAGAAAGTAGTC",  # RB06
    "GTGTTACCGTGGGAATGAATCCTT",  # RB07
    "TTCAGGGAACAAACCAAGTTACGT",  # RB08
    "AACTAGGCACAGCGAGTCTTGGTT",  # RB09
    "AAGCGTTGAAACCTTTGTCCTCTC",  # RB10
    "GTTTCATCTATCGGAGGGAATGGA",  # RB11
    "CAGGTAGAAAGAAGCAGAATCGGA",  # RB12
    "AGAACGACTTCCATACTCGTGTGA",  # RB13
    "AACGAGTCTCTTGGGACCCATAGA",  # RB14
    "AGGTCTACCTCGCTAACACCACTG",  # RB15
    "CGTCAACTGACAGTGGTTCGTACT",  # RB16
    "ACCCTCCAGGAAAGTACCTCTGAT",  # RB17
    "CCAAACCCAACAACCTAGATAGGC",  # RB18
    "GTTCCTCGTGCAGTGTCAAGAGAT",  # RB19
    "TTGCGTCCTGTTACGAGAACTCAT",  # RB20
    "GAGCCTCTCATTGTCCGTTCTCTA",  # RB21
    "ACCACTGCCATGTATCAAAGTACG",  # RB22
    "CTTACTACCCAGTGAACCTCCTCG",  # RB23
    "GCATAGTTCTGCATGATGGGTTAG",  # RB24
    "GTAAGTTGGGTATGCAACGCAATG",  # RB25
    "CATACAGCGACTACGCATTCTCAT",  # RB26
    "CGACGGTTAGATTCACCTCTTACA",  # RB27
    "TGAAACCTAAGAAGGCACCGTATC",  # RB28
    "CTAGACACCTTGGGTTGACAGACC",  # RB29
    "TCAGTGAGGATCTACTTCGACCCA",  # RB30
    "TGCGTACAGCAATCAGTTACATTG",  # RB31
    "CCAGTAGAAGTCCGACAACGTCAT",  # RB32
    "CAGACTTGGTACGGTTGGGTAACT",  # RB33
    "GGACGAAGAACTCAAGTCAAAGGC",  # RB34
    "CTACTTACGAAGCTGAGGGACTGC",  # RB35
    "ATGTCCCAGTTAGAGGAGGAAACA",  # RB36
    "GCTTGCGATTGATGCTTAGTATCA",  # RB37
    "ACCACAGGAGGACGATACAGAGAA",  # RB38
    "CCACAGTGTCAACTAGAGCCTCTC",  # RB39
    "TAGTTTGGATGACCAAGGATAGCC",  # RB40
    "GGAGTTCGTCCAGAGAAGTACACG",  # RB41
    "CTACGTGTAAGGCATACCTGCCAG",  # RB42
    "CTTTCGTTGTTGACTCGACGGTAG",  # RB43
    "AGTAGAAAGGGTTCCTTCCCACTC",  # RB44
    "GATCCAACAGAGATGCCTTCAGTG",  # RB45
    "GCTGTGTTCCACTTCATTCTCCTG",  # RB46
    "GTGCAACTTTCCCACAGGTAGTTC",  # RB47
    "CATCTGGAACGTGGTACACCTGTA",  # RB48
    "ACTGGTGCAGCTTTGAACATCTAG",  # RB49
    "ATGGACTTTGGTAACTTCCTGCGT",  # RB50
    "GTTGAATGAGCCTACTGGGTCCTC",  # RB51
    "TGAGAGACAAGATTGTTCGTGGAC",  # RB52
    "AGATTCAGACCGTCTCATGCAAAG",  # RB53
    "CAAGAGCTTTGACTAAGGAGCATG",  # RB54
    "TGGAAGATGAGACCCTGATCTACG",  # RB55
    "TCACTACTCAACAGGTGGCATGAA",  # RB56
    "GCTAGGTCAATCTCCTTCGGAAGT",  # RB57
    "CAGGTTACTCCTCCGTGAGTCTGA",  # RB58
    "TCAATCAAGAAGGGAAAGCAAGGT",  # RB59
    "CATGTTCAACCAAGGCTTCTATGG",  # RB60
    "AGAGGGTACTATGTGCCTCAGCAC",  # RB61
    "CACCCACACTTACTTCAGGACGTA",  # RB62
    "TTCTGAAGTTCCTGGGTCTTGAAC",  # RB63
    "GACAGACACCGTTCATCGACTTTC",  # RB64
    "TTCTCAGTCTTCCTCCAGACAAGG",  # RB65
    "CCGATCCTTGTGGCTTCTAACTTC",  # RB66
    "GTTTGTCATACTCGTGTGCTCACC",  # RB67
    "GAATCTAAGCAAACACGAAGGTGG",  # RB68
    "TACAGTCCGAGCCTCATGTGATCT",  # RB69
    "ACCGAGATCCTACGAATGGAGTGT",  # RB70
    "CCTGGGAGCATCAGGTAGTAACAG",  # RB71
    "TAGCTGACTGTCTTCCATACCGAC",  # RB72
    "AAGAAACAGGATGACAGAACCCTC",  # RB73
    "TACAAGCATCCCAACACTTCCACT",  # RB74
    "GACCATTGTGATGAACCCTGTTGT",  # RB75
    "ATGCTTGTTACATCAACCCTGGAC",  # RB76
    "CGACCTGTTTCTCAGGGATACAAC",  # RB77
    "AACAACCGAACCTTTGAATCAGAA",  # RB78
    "TCTCGGAGATAGTTCTCACTGCTG",  # RB79
    "CGGATGAACATAGGATAGCGATTC",  # RB80
    "CCTCATCTTGTGAAGTTGTTTCGG",  # RB81
    "ACGGTATGTCGAGTTCCAGGACTA",  # RB82
    "TGGCTTGATCTAGGTAAGGTCGAA",  # RB83
    "GTAGTGGACCTAGAACCTGTGCCA",  # RB84
    "AACGGAGGAGTTAGTTGGATGATC",  # RB85
    "AGGTGATCCCAACAAGCGTAAGTA",  # RB86
    "TACATGCTCCTGTTGTTAGGGAGG",  # RB87
    "TCTTCTACTACCGATCCGAAGCAG",  # RB88
    "ACAGCATCAATGTTTGGCTAGTTG",  # RB89
    "GATGTAGAGGGTACGGTTTGAGGC",  # RB90
    "GGCTCCATAGGAACTCACGCTACT",  # RB91
    "TTGTGAGTGGAAAGATACAGGACC",  # RB92
    "AGTTTCCATCACTTCAGACTTGGG",  # RB93
    "GATTGTCCTCAAACTGCCACCTAC",  # RB94
    "CCTGTCTGGAAGAAGAATGGACTT",  # RB95
    "CTGAACGGTCATAGAGTCCACCAT",  # RB96
]


def get_rbc_count_for_plates(n_plates: int) -> int:
    """Calculate the number of reverse barcodes needed for a given plate count.

    Each 384-well plate uses 4 RBCs (one per quadrant: TL, TR, BL, BR).
    Maximum of 96 RBCs available (24 plates).

    Args:
        n_plates: Number of 384-well plates.

    Returns:
        Number of reverse barcodes to use.
    """
    return min(n_plates * 4, 96)


# ---------------------------------------------------------------------------
# Default scoring parameters for Dorado barcode classification
# ---------------------------------------------------------------------------

DEFAULT_SCORING: dict = {
    "max_barcode_penalty": 12,
    "min_barcode_penalty_dist": 3,
    "flank_right_pad": 5,
    "flank_left_pad": 5,
    "min_separation_only_dist": 6,
    "min_flank_score": 0.9,
    "barcode_end_proximity": 150,
}

# ---------------------------------------------------------------------------
# Default mask (flanking) sequences — cutinase plasmid backbone
# ---------------------------------------------------------------------------

DEFAULT_MASKS: dict = {
    "fbc": {
        "mask1_front": "AATATAAATT",
        "mask1_rear": "CTGAGATACCTACAGCGTGAGC",
        "mask2_front": "CAAGTGAGAAATCACCATGAGTGACG",
        "mask2_rear": "ATAATTTATA",
    },
    "rbc": {
        "mask1_front": "TATAAATTAT",
        "mask1_rear": "CGTCACTCATGGTGATTTCTCACTTG",
        "mask2_front": "GCTCACGCTGTAGGTATCTCAG",
        "mask2_rear": "AATTTATATT",
    },
}


# ---------------------------------------------------------------------------
# Dorado TOML / FASTA config generation
# ---------------------------------------------------------------------------

def write_levseq_fbc_toml(
    output_dir: Path,
    kit_name: str = "levSeq_bcs_map",
    masks: dict = None,
    scoring: dict = None,
) -> Path:
    """Generate a Dorado TOML barcode arrangement file for forward barcodes.

    The mask (flanking) sequences are derived from the plasmid backbone
    surrounding each barcode insertion site and are required by Dorado
    (at least one mask per barcode end must be non-empty).

    Args:
        output_dir: Directory to write the TOML file.
        kit_name: Kit name identifier for Dorado.
        masks: Optional dict with keys ``mask1_front``, ``mask1_rear``,
            ``mask2_front``, ``mask2_rear``.  Falls back to DEFAULT_MASKS.
        scoring: Optional dict overriding DEFAULT_SCORING parameters.

    Returns:
        Path to the generated TOML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m = {**DEFAULT_MASKS["fbc"], **(masks or {})}
    s = {**DEFAULT_SCORING, **(scoring or {})}

    toml_path = output_dir / "levseq_fbc.toml"
    content = f"""[arrangement]
name = "{kit_name}"
kit = "Jewett_levSeq"

# Forward masks (flanking sequences around forward barcodes)
mask1_front = "{m['mask1_front']}"
mask1_rear  = "{m['mask1_rear']}"

# Reverse masks (context for double-end scoring)
mask2_front = "{m['mask2_front']}"
mask2_rear  = "{m['mask2_rear']}"

# Barcode patterns (both set to same FBC pattern)
barcode1_pattern = "LevSeq-fbc-%02i"
barcode2_pattern = "LevSeq-fbc-%02i"
first_index = 1
last_index = 96

[scoring]
max_barcode_penalty = {s['max_barcode_penalty']}
min_barcode_penalty_dist = {s['min_barcode_penalty_dist']}
flank_right_pad = {s['flank_right_pad']}
flank_left_pad = {s['flank_left_pad']}
min_separation_only_dist = {s['min_separation_only_dist']}
min_flank_score = {s['min_flank_score']}
barcode_end_proximity = {s['barcode_end_proximity']}
"""
    toml_path.write_text(content)
    return toml_path


def write_levseq_rbc_toml(
    output_dir: Path,
    n_barcodes: int = 4,
    kit_name: str = "levSeq_bcs_map",
    masks: dict = None,
    scoring: dict = None,
) -> Path:
    """Generate a Dorado TOML barcode arrangement file for reverse barcodes.

    The mask (flanking) sequences are the reverse complements of the FBC
    masks, since reverse-strand reads see the backbone in the opposite
    orientation.  At least one mask per barcode end must be non-empty
    for Dorado to accept the arrangement.

    Args:
        output_dir: Directory to write the TOML file.
        n_barcodes: Number of reverse barcodes to include (default 4 = 1 plate).
        kit_name: Kit name identifier for Dorado.
        masks: Optional dict with keys ``mask1_front``, ``mask1_rear``,
            ``mask2_front``, ``mask2_rear``.  Falls back to DEFAULT_MASKS.
        scoring: Optional dict overriding DEFAULT_SCORING parameters.

    Returns:
        Path to the generated TOML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m = {**DEFAULT_MASKS["rbc"], **(masks or {})}
    s = {**DEFAULT_SCORING, **(scoring or {})}

    last_index = min(n_barcodes, 96)
    toml_path = output_dir / "levseq_rbc.toml"
    content = f"""[arrangement]
name = "{kit_name}"
kit = "Jewett_levSeq"

# Forward masks (RC of FBC reverse masks — context for double-end scoring)
mask1_front = "{m['mask1_front']}"
mask1_rear  = "{m['mask1_rear']}"

# Reverse masks (flanking sequences around reverse barcodes)
mask2_front = "{m['mask2_front']}"
mask2_rear  = "{m['mask2_rear']}"

# Barcode patterns (both set to same RBC pattern)
barcode1_pattern = "LevSeq-rbc-%02i"
barcode2_pattern = "LevSeq-rbc-%02i"
first_index = 1
last_index = {last_index}

[scoring]
max_barcode_penalty = {s['max_barcode_penalty']}
min_barcode_penalty_dist = {s['min_barcode_penalty_dist']}
flank_right_pad = {s['flank_right_pad']}
flank_left_pad = {s['flank_left_pad']}
min_separation_only_dist = {s['min_separation_only_dist']}
min_flank_score = {s['min_flank_score']}
barcode_end_proximity = {s['barcode_end_proximity']}
"""
    toml_path.write_text(content)
    return toml_path


def write_levseq_fbc_fasta(output_dir: Path) -> Path:
    """Write all 96 forward barcode sequences to a FASTA file.

    Args:
        output_dir: Directory to write the FASTA file.

    Returns:
        Path to the generated FASTA file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / "levseq_fbc.fasta"
    lines = []
    for i, seq in enumerate(LEVSEQ_FBC):
        lines.append(f">LevSeq-fbc-{i + 1:02d}")
        lines.append(seq)
    fasta_path.write_text("\n".join(lines) + "\n")
    return fasta_path


def write_levseq_rbc_fasta(
    output_dir: Path,
    n_barcodes: Optional[int] = None,
) -> Path:
    """Write reverse barcode sequences to a FASTA file.

    Args:
        output_dir: Directory to write the FASTA file.
        n_barcodes: Number of RBCs to include. Defaults to all 96.

    Returns:
        Path to the generated FASTA file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(n_barcodes, 96) if n_barcodes is not None else 96
    fasta_path = output_dir / "levseq_rbc.fasta"
    lines = []
    for i in range(count):
        lines.append(f">LevSeq-rbc-{i + 1:02d}")
        lines.append(LEVSEQ_RBC[i])
    fasta_path.write_text("\n".join(lines) + "\n")
    return fasta_path
