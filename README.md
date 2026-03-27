# uSort-M

**Rapid and low-cost parsed variant library generation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

uSort-M converts pooled DNA libraries into large collections of individually-isolated, sequence-verified variants at a fraction of traditional gene synthesis costs.

## Overview

Traditional approaches to generating parsed variant libraries require expensive per-gene synthesis and individual cloning. uSort-M uses FACS to isolate single cells from a pooled transformation, then identifies variants by amplicon sequencing with well-specific barcodes.

**Key advantages:**
- Significant cost savings compared to traditional gene synthesis
- 10-day working time from oligo pool to verified clones
- Scalable from tens to thousands of variants
- Compatible with diverse library inputs

## Installation

```bash
# Basic installation
pip install -e .

# Full installation with all dependencies
pip install -e ".[all]"
```

### External Tools (for demultiplexing)

The `usortm demux` command requires these tools installed separately:

| Tool | Min Version | Purpose | macOS | Linux |
|------|-------------|---------|-------|-------|
| [dorado](https://github.com/nanoporetech/dorado) | 1.3+ | Barcode demultiplexing | [GitHub releases](https://github.com/nanoporetech/dorado/releases) (`.zip`) | [GitHub releases](https://github.com/nanoporetech/dorado/releases) (`.tar.gz`) |
| [minimap2](https://github.com/lh3/minimap2) | 2.20+ | Reference alignment | `brew install minimap2` | `apt/dnf install minimap2` |
| [samtools](https://github.com/samtools/samtools) | 1.16+ | BAM processing & consensus | `brew install samtools` | `apt/dnf install samtools` |

Windows users should run inside WSL2 (Ubuntu) and use the Linux instructions. `usortm` auto-discovers dorado in `~/Downloads/dorado-*/bin/` and `~/.dorado/bin/`; set `DORADO_PATH`, `MINIMAP2_PATH`, or `SAMTOOLS_PATH` to override.

## Quick Start

### Estimate costs

```bash
usortm estimate --library-size 500 --seq-length 300
```

### Plan and execute a full workflow

```bash
# 1. Initialize project from variant list
usortm plan variants.csv --output my_project/

# 2. [Perform wet lab: assembly, sorting, barcoding, sequencing]

# 3. Process sequencing data (with library CSV for variant calling)
usortm demux my_project/ --fastq sequencing-data.fastq --library-csv variants.csv

# 4. Generate hit-picking list
usortm pick my_project/

# 5. Create final report
#    Generates HTML summary, CSVs, and a shareable zip file
usortm report my_project/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `estimate` | Quick cost and effort estimation |
| `plan` | Initialize project from variant list |
| `demux` | Demultiplex sequencing data (LevSeq barcodes via dorado, reference alignment, consensus, variant calling) |
| `pick` | Generate Integra ASSIST hit-picking list (ordered by input library) |
| `reorder` | Export synthesis order for dropout variants (unrecovered after round 1) |
| `merge` | Merge hit-picking lists from multiple rounds into a single final pick list |
| `report` | Generate final plate maps, coverage stats, HTML summary, and shareable zip |

### Multi-round workflow (recovering dropouts)

After round 1 pick, variants that were not recovered (dropouts) can be re-synthesized and run through a second round of uSort-M to maximize library coverage.

```bash
# After round 1 pick completes, export a synthesis order for dropout variants
usortm reorder my_project/

# [Re-synthesize dropouts and perform wet lab for round 2]

# Plan round 2 against the existing project
usortm plan dropouts.csv --output my_project/ --round 2

# Demultiplex round 2 sequencing data
usortm demux my_project/ --fastq round2-data.fastq --round 2

# Pick round 2 hits
usortm pick my_project/ --round 2

# Merge round 1 and round 2 picks into a single final list
usortm merge my_project/

# Generate the merged report (covers both rounds)
usortm report my_project/ --round merged
```

After `usortm merge`, the combined Integra ASSIST pick list is written to `my_project/merged/pick/Integra ASSIST Input/`. Variants are placed at their library-ordered positions across both rounds, with round 2 hits filling in wherever round 1 did not recover.

### Example: Cost Estimate

```bash
usortm estimate -n 500 -l 300

# Output:
# ╭────────────────────────────────╮
# │ uSort-M Cost Estimate          │
# │ Library: 500 variants × 300 bp │
# ╰────────────────────────────────╯
#
#                   Cost Breakdown
# ╭────────────────────────┬─────────┬─────────────╮
# │ Step                   │ uSort-M │ Traditional │
# ├────────────────────────┼─────────┼─────────────┤
# │ Synthesis              │  $1,373 │     $17,500 │
# │ Cloning                │     $54 │      $6,048 │
# │ Sorting                │    $104 │         N/A │
# │ Barcoding + Sequencing │  $1,477 │        $500 │
# │ Hit-picking            │     $80 │         N/A │
# │ Total                  │  $3,088 │     $24,048 │
# ╰────────────────────────┴─────────┴─────────────╯
#
#   7.8-fold savings with uSort-M
```

## Workflow Timeline

| Day | Step | Duration |
|-----|------|----------|
| 1 | Pooled assembly + transformation | 4-6 hours |
| 2+ | FACS sorting | ~8 min/plate |
| 2+ | Outgrowth | Overnight |
| 3+ | PCR barcoding | ~50 min/plate |
| 4-6 | Sequencing | 1-3 days |
| 6+ | Analysis + hit-picking | 1-2 hours |

## Python API

```python
from usortm.costs import cost_functions as cf

# Calculate costs
costs = cf.usortm_total_cost(
    library_sizes=[500, 1000],
    seq_lengths=[300]
)

# Run coverage simulations
from usortm.simulate import sortm

results = sortm.sortm(
    n_sims=1000,
    lib_size=500,
    skew=4,
    fold_sampling=8,
)
```

## Documentation

Full documentation available at [fordycelab.github.io/usortm](https://fordycelab.github.io/usortm)

## Citation

If you use uSort-M in your research, please cite:

```
Olivas MB, Almhjell PJ, Shanahan JD, Fordyce PM. uSort-M: Scalable isolation 
of user-defined sequences from diverse pooled libraries. bioRxiv (2026). DOI: 10.64898/2026.01.12.699065
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/FordyceLab/usortm)
- [Documentation](https://fordycelab.github.io/usortm)
- [Fordyce Lab](https://fordycelab.stanford.edu)
