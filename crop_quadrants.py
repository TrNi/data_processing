import os
import csv
from pathlib import Path
from PIL import Image

# ─── HARDCODED INPUTS ────────────────────────────────────────────────────────
# One integer (1-4) per image in DIRS[0], in lexicographic order.
# 1=top-left  2=top-right  3=bottom-right  4=bottom-left
abc = [2, 1, 3, 2, 2, 4, 2, 2, 3, 1, 2, 2, 4, 1, 3, 3, 4, 3, 4, 2, 1, 1,
       4, 4, 3, 3, 3, 1, 3, 1, 2, 3, 2, 3, 4, 3, 1, 4, 2, 3, 1, 3, 3, 4,
       2, 1, 3, 4, 2, 3, 2, 1, 2, 4, 2, 2, 3, 3, 1, 4, 4, 2, 3, 4, 4, 2,
       1, 1, 4, 2, 1, 2, 4, 2, 1, 1, 4, 1, 2, 3]

# DIRS = [
#     r"I:\My Drive\DOF_benchmarking\Scene8\EOS6D_A_Left\fl_70mm\F2.8_align_whitebal",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\bokehme_dfs20_K40_dispfocus0.15",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\drbokeh_K15_fp0.3",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\bokehdiff_K15_fp0.3\demo",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\bokehlicious_fl_70_A_10_intp",
# ]
# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene8\fl_70"
# EXISTING_CSV = SAVEDIR+r"\mapping.csv"

# DIRS = [
#     r"I:\My Drive\DOF_benchmarking\Scene8\EOS6D_A_Left\fl_45mm\F2.8_align_whitebal",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\bokehme_dfs20_K40_dispfocus0.15",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\drbokeh_K15_fp0.3",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\bokehdiff_K15_fp0.3\demo",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\bokehlicious_fl_45_A_13_intp",
# ]
# EXISTING_CSV = r"I:\My Drive\DOF_benchmarking\MOR\Scene8\fl_45\mapping.csv"
# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene8\fl_45"

# DIRS = [
#     r"I:\My Drive\DOF_benchmarking\Scene5\EOS6D_A_Left\fl_36mm\F2.8_align_whitebal",
#     r"I:\My Drive\DOF_benchmarking\Scene5\fl_36\bokehme_dfs20_K40_dispfocus0.15",
#     r"I:\My Drive\DOF_benchmarking\Scene5\fl_36\drbokeh_K15_fp0.3",
#     r"I:\My Drive\DOF_benchmarking\Scene5\fl_36\bokehdiff_K15_fp0.3\demo",
#     r"I:\My Drive\DOF_benchmarking\Scene5\fl_36\bokehlicious_fl_36_A_20_intp",
#  ]
# EXISTING_CSV = r"I:\My Drive\DOF_benchmarking\MOR\Scene5\fl_36\mapping.csv"
# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene5\fl_36"


# DIRS = [
#     r"I:\My Drive\DOF_benchmarking\Scene3\EOS6D_A_Right\fl_60mm\F2.8_align_whitebal",
#     r"I:\My Drive\DOF_benchmarking\Scene3\fl_60\bokehme_dfs20_K40_dispfocus0.15",
#     r"I:\My Drive\DOF_benchmarking\Scene3\fl_60\drbokeh_K15_fp0.3",
#     r"I:\My Drive\DOF_benchmarking\Scene3\fl_60\bokehdiff_K15_fp0.3\demo",
#     r"I:\My Drive\DOF_benchmarking\Scene3\fl_60\bokehlicious_fl_60_A_7_intp",
# ]
# EXISTING_CSV = r"I:\My Drive\DOF_benchmarking\MOR\Scene3\fl_60\mapping.csv"
# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene3\fl_60"

# DIRS = [
# #     r"I:\My Drive\DOF_benchmarking\Scene3\EOS6D_A_Right\fl_45mm\F2.8_align_whitebal",
# #     r"I:\My Drive\DOF_benchmarking\Scene3\fl_45\bokehme_dfs20_K40_dispfocus0.15",
# #     r"I:\My Drive\DOF_benchmarking\Scene3\fl_45\drbokeh_K15_fp0.3",
# #     r"I:\My Drive\DOF_benchmarking\Scene3\fl_45\bokehdiff_K15_fp0.3\demo",
#     r"I:\My Drive\DOF_benchmarking\Scene3\fl_45\bokehlicious_fl_45_A_13_intp",
# ]
# EXISTING_CSV = r"I:\My Drive\DOF_benchmarking\MOR\Scene3\fl_45\mapping.csv"
# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene3\fl_45"

DIRS = [
    r"I:\My Drive\DOF_benchmarking\inference\fl_70\F2.8_align_whitebal",
    r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehme_dfs20_K40_dispfocus0.15",
    r"I:\My Drive\DOF_benchmarking\inference\fl_70\drbokeh_K15_fp0.3_ls71",
    r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehdiffnew_K15\demo",
    r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F10_intp", # remaining
]
EXISTING_CSV = r"I:\My Drive\DOF_benchmarking\MOR\Scene1\fl_70\mapping.csv"
SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene1\fl_70"

# DIRS = [
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\F2.8_align_whitebal",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehme_dfs20_K40_dispfocus0.15",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\drbokeh_K15_fp0.3_ls71",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehdiffnew_K20\demo",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F7_intp",
# ]

# SAVEDIR = r"I:\My Drive\DOF_benchmarking\MOR\Scene1\fl_60"


CROP_W = 1368
CROP_H = 912

# Set to a CSV path to skip build_csv and jump straight to apply_crops.
# Set to None to run both steps.
# e.g. r"I:\My Drive\DOF_benchmarking\MOR\Scene1\fl_70\mapping.csv"
# ─────────────────────────────────────────────────────────────────────────────

KIND_MAP = {1: "top_left", 2: "top_right", 3: "bottom_right", 4: "bottom_left"}
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def get_images(folder: str):
    return sorted(
        [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )


def crop_box(kind: str, img_w: int, img_h: int):
    """Return PIL crop box (x1, y1, x2, y2) aligned to actual image edges."""
    if kind == "top_left":
        return 0, 0, CROP_W, CROP_H
    if kind == "top_right":
        return img_w - CROP_W, 0, img_w, CROP_H
    if kind == "bottom_right":
        return img_w - CROP_W, img_h - CROP_H, img_w, img_h
    if kind == "bottom_left":
        return 0, img_h - CROP_H, CROP_W, img_h
    if kind == "center":
        x = (img_w - CROP_W) // 2
        y = (img_h - CROP_H) // 2
        return x, y, x + CROP_W, y + CROP_H
    raise ValueError(f"Unknown crop kind: {kind!r}")


# ─── PART 1: build mapping CSV ───────────────────────────────────────────────

def build_csv() -> str:
    os.makedirs(SAVEDIR, exist_ok=True)
    images = get_images(DIRS[0])

    # if len(images) != len(abc):
    #     raise ValueError(
    #         f"abc has {len(abc)} entries but '{DIRS[0]}' contains {len(images)} images"
    #     )

    rows = []

    # Block 1: corner crops driven by abc (indices 0 .. N-1)
    for i, img_path in enumerate(images):
        with Image.open(img_path) as im:
            w, h = im.size
        kind = KIND_MAP[abc[i+35]]
        x1, y1, x2, y2 = crop_box(kind, w, h)
        rows.append([i, str(img_path), abc[i], kind, x1, y1, x2, y2])

    # Block 2: center crops, image index resets to 0 (indices 0 .. N-1 again)
    for i, img_path in enumerate(images):
        with Image.open(img_path) as im:
            w, h = im.size
        x1, y1, x2, y2 = crop_box("center", w, h)
        rows.append([i, str(img_path), 0, "center", x1, y1, x2, y2])

    csv_path = os.path.join(SAVEDIR, "mapping.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_index", "full_path", "integer", "crop_kind",
             "crop_tl_x", "crop_tl_y", "crop_br_x", "crop_br_y"]
        )
        writer.writerows(rows)

    n = len(images)
    print(f"CSV saved → {csv_path}  ({n} images × 2 blocks = {len(rows)} rows)")
    return csv_path


# ─── PART 2: apply crops to all 5 directories ────────────────────────────────

def apply_crops(csv_path: str):
    with open(csv_path, newline="") as f:
        mapping = list(csv.DictReader(f))

    for dir_idx, folder in enumerate(DIRS):
        images = get_images(folder)
        # subfolder name: zero-padded index + original folder name (avoids collisions)
        out_dir = Path(SAVEDIR) / f"dir_{dir_idx:02d}_{Path(folder).name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for row_idx, row in enumerate(mapping):
            img_idx = int(row["image_index"])
            kind    = row["crop_kind"]

            if img_idx >= len(images):
                print(f"  [SKIP] dir {dir_idx} row {row_idx}: index {img_idx} >= {len(images)} images")
                continue

            img_path = images[img_idx]
            with Image.open(img_path) as im:
                w, h = im.size
                box     = crop_box(kind, w, h)
                cropped = im.crop(box)

            out_name = f"{row_idx:04d}_{kind}_{img_path.name}"
            cropped.save(out_dir / out_name)
            saved += 1

        print(f"Crops saved → {out_dir}  ({saved} files)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if EXISTING_CSV:
        print(f"Using existing CSV → {EXISTING_CSV}")
        apply_crops(EXISTING_CSV)
    else:
        csv_path = build_csv()
        apply_crops(csv_path)
