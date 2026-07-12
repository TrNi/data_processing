#!/usr/bin/env python3
"""
sample_urls.py

Randomly samples K rows per user from urls.csv and writes N independent CSVs.

Output filenames:
    urls_0.csv  ...  urls_9.csv      (N <= 10)
    urls_00.csv ... urls_99.csv      (N >  10)

Each output CSV preserves the original header and contains K rows sampled
without replacement from the pool. Samples across users are independent
(the same row can appear in multiple users' CSVs).

Usage:
    python sample_urls.py [--input <urls.csv>] [--outdir <dir>] [--N <int>] [--K <int>] [--seed <int>]
"""

import csv
import random
import argparse
from pathlib import Path

INPUT_CSV  = r"G:\Development\data_processing\urls.csv"
OUTPUT_DIR = r"G:\Development\data_processing\visuals\mor_scoring\samples"
N          = 5
K_PER_USER = 60
SEED       = None  # set an int for reproducibility


def load_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)
    return header, rows


def main():
    parser = argparse.ArgumentParser(description="Sample K rows per user from urls.csv, N times.")
    parser.add_argument("--input",  default=INPUT_CSV,  help="Source urls.csv path.")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Directory for output CSVs.")
    parser.add_argument("--N",      type=int, default=N,          help="Number of users / output files.")
    parser.add_argument("--K",      type=int, default=K_PER_USER, help="Rows per user.")
    parser.add_argument("--seed",   type=int, default=SEED,       help="RNG seed for reproducibility.")
    args = parser.parse_args()

    src   = Path(args.input)
    outd  = Path(args.outdir)
    n     = args.N
    k     = args.K
    seed  = args.seed

    header, rows = load_rows(src)
    pool_size = len(rows)

    if k > pool_size:
        raise ValueError(f"K={k} exceeds pool size {pool_size}; reduce K or add more rows.")

    outd.mkdir(parents=True, exist_ok=True)
    width = 2 if n > 10 else 1
    rng   = random.Random(seed)

    print(f"Pool: {pool_size} rows  |  N={n} users  |  K={k} rows each  |  seed={seed}")

    for i in range(n):
        sample   = rng.sample(rows, k)
        out_name = f"urls_{i:0{width}d}.csv"
        out_path = outd / out_name

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sample)

        print(f"  Written: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
