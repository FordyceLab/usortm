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

### Optional dependencies

```bash
pip install -e ".[viz]"    # Visualization (matplotlib, bokeh)
pip install -e ".[demux]"  # Demultiplexing (biopython, pysam)
pip install -e ".[dev]"    # Development (pytest)
```

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

# 3. Process sequencing data
usortm demux my_project/ --fastq data.fastq

# 4. Generate hit-picking list
usortm pick my_project/

# 5. Create final report
usortm report my_project/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `estimate` | Quick cost and effort estimation |
| `plan` | Initialize project from variant list |
| `demux` | Demultiplex sequencing data |
| `pick` | Generate Integra ASSIST hit-picking list |
| `report` | Generate final plate maps and summary |
| `integra` | Standalone hit-list generation (without project) |

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

- [Getting Started](https://fordycelab.github.io/usortm/getting-started.html)
- [Workflow Guide](https://fordycelab.github.io/usortm/workflow/)
- [Cost Calculator](https://fordycelab.github.io/usortm/tools/cost-calculator.html)
- [CLI Reference](https://fordycelab.github.io/usortm/cli-reference.html)

## Citation

If you use uSort-M in your research, please cite:

```
Olivas MB, Almhjell PJ, Shanahan JD, Fordyce PM. uSort-M: Scalable isolation 
of user-defined sequences from diverse pooled libraries. bioRxiv (2026).
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/FordyceLab/usortm)
- [Documentation](https://fordycelab.github.io/usortm)
- [Fordyce Lab](https://fordycelab.stanford.edu)
