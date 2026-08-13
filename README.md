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

# 2. [Wet lab: order and amplify the library]

# 3. Measure the real library skew before committing to a sort
#    (sequence a little of the amplified library, e.g. Plasmidsaurus premium PCR)
usortm skew library.fastq --project my_project/

# 4. [Wet lab: sorting, barcoding, sequencing]

# 5. Process sequencing data (with library CSV for variant calling)
usortm demux my_project/ --fastq sequencing-data.fastq --library-csv variants.csv

# 6. Generate hit-picking list
usortm pick my_project/

# 7. Create final report
#    Generates HTML summary, CSVs, and a shareable zip file
usortm report my_project/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `estimate` | Quick cost and effort estimation |
| `plan` | Initialize project from variant list |
| `skew` | Measure library skew from sequencing reads and recommend a sorting depth |
| `demux` | Demultiplex sequencing data (LevSeq barcodes via dorado, reference alignment, consensus, variant calling) |
| `pick` | Generate Integra ASSIST hit-picking list (ordered by input library) |
| `reorder` | Export synthesis order for dropout variants (unrecovered after round 1) |
| `merge` | Merge hit-picking lists from multiple rounds into a single final pick list |
| `report` | Generate final plate maps, coverage stats, HTML summary, and shareable zip |

### Measuring library skew before sorting

How deeply you need to sort is set by how unevenly the library is distributed.
`usortm plan` has to assume that skew from the synthesis method, because the
library does not exist yet. Once it does, a shallow sequencing run of the
amplified library — 12–20k reads is plenty — measures it directly.

```bash
usortm skew library.fastq --project my_project/
```

The command aligns the reads against the starting variant list, counts reads
per variant, and recommends a fold-sampling depth. It needs only **minimap2**,
not the full demux toolchain. The measurement costs one extra sequencing
turnaround before sorting, which is usually cheaper than sorting the wrong
number of plates.

**Why the raw ratio is not the answer.** At 12–20k reads over a few hundred to
a few thousand variants, each variant is seen only ~8–30 times. Poisson
counting noise alone makes a *perfectly even* library measure as roughly
1.6× skewed, and a genuinely 4×-skewed library of 2000 variants measures as
~6.9×. Sorting on the raw number wastes plates. `usortm skew` fits a
zero-inflated Poisson–log-normal model and reports both:

```
                Measured Library Skew
╭──────────────────────────┬────────────────────────────────╮
│ Depth                    │            7.5 reads/variant   │
│ Q90/Q10, raw             │                         6.9×   │
│ Q90/Q10, noise-corrected │      4.0× (95% CI 3.6–4.3)     │
│ Effective library size   │                        1,204   │
│ Undetected variants      │                           31   │
│ Estimated dropout        │                         1.6%   │
╰──────────────────────────┴────────────────────────────────╯
```

Dropouts are estimated separately from skew, so variants missing from the tube
are not mistaken for unevenness. That distinction is actionable: sorting
recovers rare variants, but nothing recovers a variant that was never
synthesized, so the report gives a **coverage ceiling** and flags a target
above it.

The command prints a log-abundance histogram — a uniform library is a tight
bell, and skew is width. Two curves are drawn over it: the fit *including*
counting noise, which should track the bars, and the underlying abundance with
that noise removed. The gap between them is the spread that reading the
histogram at face value would mistake for skew.

Results are written to `<project>/skew/` (`variant_counts.csv`,
`skew_report.json`, and an HTML summary with the histogram plus rank-abundance
and cumulative plots), and recorded in `usortm_project.json` under
`measured_skew`. The
planning-time `skew` and `fold_sampling` are left untouched so the assumption
and the measurement sit side by side.

**Libraries this cannot measure.** If variants differ by a single codon, ONT
reads cannot be attributed to individual variants and per-variant counts would
be meaningless. `usortm skew` checks separability first and refuses rather than
reporting confident nonsense; `--force` overrides.

**Checking it against a known answer.** `scripts/make_synthetic_library.py`
generates a library CSV and FASTQ whose abundances, dropouts, and per-variant
read counts are all recorded, so the whole chain can be measured against the
truth rather than against another estimate:

```bash
python scripts/make_synthetic_library.py /tmp/lib --library-size 400 --skew 4 --dropout 0.05
usortm skew /tmp/lib/library.fastq --variants /tmp/lib/variants.csv --output /tmp/lib/skew
```

Recover the `realized skew` it prints, not the requested one — a finite draw
differs from the distribution it came from. `--mode codon_scan` builds the
single-codon shape instead, which the separability check refuses.

The corrected estimate is unbiased to a few percent up to about 8× Q90/Q10
across 300–2000 variants and 7–50 reads per variant. Above ~10× it reads low
(≈0.85× of truth at 16×), because too much of the library falls below one
expected read; that case is flagged in the output and should be read as a lower
bound on both skew and sorting depth.

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

Coverage is simulated from the library size, skew, and fold-sampling, and reported with the costs:

```bash
usortm estimate -n 376 -s 2 -f 3.72
#   Simulation: 3.72× fold-sampling → 79.3% expected coverage (77%–81% across 100 sims)
```

Omit `-f` and the fold-sampling is searched for instead, returning the shallowest sort that reaches `--target-coverage` (90% by default).

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

# Or predict coverage for a planned sort
prediction = sortm.expected_coverage(
    lib_size=376,
    skew=2,
    fold_sampling=3.72,
)
prediction["coverage"]  # mean fraction of the library recovered

# Measure skew from a sequenced library and recommend a sorting depth
from usortm.qc import profile_library

profile = profile_library("library.fastq", "variants.csv", "skew_out/")
profile.stats.q90_q10_corrected      # noise-corrected Q90/Q10
profile.stats.coverage_ceiling       # limit imposed by synthesis dropouts
profile.recommendation.fold_sampling # wells to sort per library member

# Simulate against measured abundances instead of a fitted log-normal
results = sortm.sortm(
    n_sims=1000,
    fold_sampling=8,
    pool=profile.stats.shrunk_abundance,
)
```

## Documentation

Full documentation available at [fordycelab.github.io/usortm](https://fordycelab.github.io/usortm)

## Citation

If you use uSort-M in your research, please cite:

```
Olivas MB, Almhjell PJ, Brixi LK, Shanahan JD, Fordyce PM. uSort-M: Scalable isolation 
of user-defined sequences from diverse pooled libraries. bioRxiv (2026). DOI: 10.64898/2026.01.12.699065
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/FordyceLab/usortm)
- [Documentation](https://fordycelab.github.io/usortm)
- [Fordyce Lab](https://fordycelab.stanford.edu)
