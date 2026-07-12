#!/usr/bin/env python3
"""
generate_url_csv.py

Traverses a local MOR directory and produces a CSV where each row contains
5 Backblaze public URLs from the same image position across 5 subdirectories
(dir_00 ... dir_04) within each Scene/fl group.

Expected local layout:
    MOR_ROOT / <Scene> / <fl> / dir_00_<name> / <image files>

URL = BASE_URL / <Scene> / <fl> / dir_0X_<name> / <filename>

Usage:
    python generate_url_csv.py [--root <MOR_ROOT>] [--output <csv_path>]
"""

import csv
import sys
import argparse
from pathlib import Path

MOR_ROOT = r"I:\My Drive\DOF_benchmarking\MOR"
BASE_URL = "https://f004.backblazeb2.com/file/dof-mos"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUTPUT_CSV = r"G:\Development\data_processing\urls.csv"


def get_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def build_tree(root: Path) -> dict:
    """
    Returns { (scene, fl) : { dir_name : [Path, ...] } }
    All keys and file lists are sorted lexicographically.
    """
    tree = {}
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for fl_dir in sorted(scene_dir.iterdir()):
            if not fl_dir.is_dir():
                continue
            dirs = {}
            for sub in sorted(fl_dir.iterdir()):
                if sub.is_dir() and sub.name.startswith("dir_"):
                    dirs[sub.name] = get_images(sub)
            if dirs:
                tree[(scene_dir.name, fl_dir.name)] = dirs
    return tree


def build_rows(tree: dict, root: Path) -> list[list[str]]:
    rows = []
    for (scene, fl), dirs in sorted(tree.items()):
        sorted_dirs = sorted(dirs.keys())

        if len(sorted_dirs) != 5:
            print(
                f"WARNING: {scene}/{fl} has {len(sorted_dirs)} dir(s) (expected 5) — skipping.",
                file=sys.stderr,
            )
            continue

        file_lists = [dirs[d] for d in sorted_dirs]
        counts = [len(fl_list) for fl_list in file_lists]
        min_count = min(counts)

        if min_count == 0:
            print(f"WARNING: {scene}/{fl} has empty directory — skipping.", file=sys.stderr)
            continue

        if len(set(counts)) != 1:
            print(
                f"WARNING: {scene}/{fl} dir file counts differ {counts} — using min={min_count} rows.",
                file=sys.stderr,
            )

        for i in range(min_count):
            row = [
                f"{BASE_URL}/{img.relative_to(root).as_posix()}"
                for img in (fl_list[i] for fl_list in file_lists)
            ]
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 5-column URL CSV from the local MOR directory."
    )
    parser.add_argument(
        "--root",
        default=MOR_ROOT,
        help=f"Root MOR directory to traverse (default: {MOR_ROOT}).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help=f"Output CSV file path (default: {OUTPUT_CSV}).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Traversing: {root}")
    tree = build_tree(root)
    groups = sorted(tree.keys())
    print(f"  {len(groups)} Scene/fl groups found: {groups}")
    for (scene, fl), dirs in sorted(tree.items()):
        counts = {d: len(imgs) for d, imgs in sorted(dirs.items())}
        print(f"  {scene}/{fl}: {counts}")

    rows = build_rows(tree, root)
    print(f"  {len(rows)} rows to write.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dir_00", "dir_01", "dir_02", "dir_03", "dir_04"])
        writer.writerows(rows)

    print(f"CSV written to: {out_path}")


if __name__ == "__main__":
    main()
