#!/usr/bin/env python3
"""Generate a synthetic library CSV + FASTQ with known ground truth.

Useful for exercising `usortm skew` end to end, or for checking what the
measurement does to a library shape before committing real reads to it.
The abundance distribution, dropouts, and per-variant read counts are all
recorded in truth.json next to the data.

Usage:
    python scripts/make_synthetic_library.py OUT_DIR [options]

Examples:
    # A diverse 400-variant library, 4x skewed, 5% of it never synthesized
    python scripts/make_synthetic_library.py /tmp/lib1 \\
        --library-size 400 --skew 4 --dropout 0.05

    # An amber-scan shape, which `usortm skew` will refuse to measure
    python scripts/make_synthetic_library.py /tmp/lib2 --mode codon_scan

Then:
    usortm skew /tmp/lib1/library.fastq --variants /tmp/lib1/variants.csv \\
        --output /tmp/lib1/skew
"""

import argparse
import sys
from pathlib import Path

from usortm.qc.synthetic import make_synthetic_library


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out_dir", type=Path, help="Directory to write into.")
    parser.add_argument("--library-size", type=int, default=400,
                        help="Number of variants (default: 400).")
    parser.add_argument("--seq-length", type=int, default=300,
                        help="Variable-region length in bp (default: 300).")
    parser.add_argument("--skew", type=float, default=4.0,
                        help="Requested Q90/Q10 abundance ratio (default: 4).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Fraction of variants never synthesized (default: 0).")
    parser.add_argument("--n-reads", type=int, default=15000,
                        help="Reads to generate (default: 15000).")
    parser.add_argument("--error-rate", type=float, default=0.03,
                        help="Per-base read error rate (default: 0.03).")
    parser.add_argument("--mode", choices=("diverse", "codon_scan"),
                        default="diverse",
                        help="Library shape (default: diverse).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args(argv)

    try:
        lib = make_synthetic_library(
            args.out_dir,
            library_size=args.library_size,
            seq_length=args.seq_length,
            skew=args.skew,
            dropout=args.dropout,
            n_reads=args.n_reads,
            error_rate=args.error_rate,
            mode=args.mode,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"variants  {lib.variants_csv}")
    print(f"reads     {lib.fastq}")
    print(f"truth     {lib.truth_json}")
    print()
    print(f"  mode              {lib.params['mode']}")
    print(f"  variants          {lib.library_size}")
    print(f"  reads             {lib.n_reads:,} ({lib.n_junk:,} junk)")
    print(f"  mean depth        {lib.params['mean_depth']} reads/variant")
    print(f"  requested skew    {lib.params['requested_skew']}x")
    print(f"  realized skew     {lib.realized_skew:.2f}x   <- recover this")
    print(f"  absent variants   {lib.n_absent}")
    print()
    print("Measure it with:")
    print(f"  usortm skew {lib.fastq} \\")
    print(f"    --variants {lib.variants_csv} \\")
    print(f"    --output {args.out_dir}/skew")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
