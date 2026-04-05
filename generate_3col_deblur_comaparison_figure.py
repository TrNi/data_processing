"""
generate_deblur_comparison_figure.py  (memory-efficient version)
================================================================
Processes ONE cell (row x column) at a time and frees all arrays
immediately after imshow() so RAM stays low on constrained hardware.

Usage:
    python generate_deblur_comparison_figure.py \
        /path/scene1/roi_coords_deblur.txt \
        /path/scene2/roi_coords_deblur.txt \
        /path/scene3/roi_coords_deblur.txt \
        --output comparison_deblur.png
"""

import os
import sys
import gc
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Times New Roman']

BBOX_COLORS = [
    (0.20, 1.00, 0.20),   # green
    (1.00, 0.20, 0.60),   # pink
    (0.10, 0.45, 1.00),   # blue
]
BBOX_COLORS_PIL = [
    ( 51, 255,  51),
    (255,  51, 153),
    ( 26, 115, 255),
]

SCALE_FACTOR = 1.5
BORDER_WIDTH = 20


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_text_file(filepath):
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    bboxes = []
    for i in range(0, 6, 2):
        y1, x1 = map(int, lines[i].split(','))
        y2, x2 = map(int, lines[i + 1].split(','))
        bboxes.append((x1, y1, x2, y2))          # PIL: (x1,y1,x2,y2)

    index = int(lines[6].split(',')[1].strip())

    entries = []
    for line in lines[7:]:
        parts = line.split(',', 1)
        if len(parts) == 2:
            entries.append((parts[0].strip(), parts[1].strip()))

    return bboxes, index, entries


# ---------------------------------------------------------------------------
# Loaders  — each returns exactly one PIL Image, nothing cached
# ---------------------------------------------------------------------------

def load_png(path, target_size=None):
    img = Image.open(path).convert('RGB')
    if target_size and img.size != target_size:
        tmp = img.resize(target_size, Image.LANCZOS)
        img.close(); img = tmp
    return img


def load_h5(h5_path, key, target_size=None):
    with h5py.File(h5_path, 'r') as hf:
        if key not in hf:
            avail = list(hf.keys())[:5]
            raise KeyError(f"Key '{key}' not in {h5_path}. First 5: {avail}")
        arr = hf[key][:]

    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    img = Image.fromarray(arr, 'RGB')
    del arr

    if target_size and img.size != target_size:
        tmp = img.resize(target_size, Image.LANCZOS)
        img.close(); img = tmp
    return img


def load_one(label, path, input_key, target_size):
    ll = label.lower()
    if ll.startswith('ip') or ll.startswith('gt'):
        return load_png(path, target_size)
    return load_h5(path, input_key, target_size)


# ---------------------------------------------------------------------------
# Render one cell, free everything before returning
# ---------------------------------------------------------------------------

def render_cell(fig, gs, row_idx, col_idx,
                label, path, input_key, gt_size,
                bboxes, is_first_col):

    # 1. Load + upscale
    img = load_one(label, path, input_key, gt_size)
    new_w = int(img.width  * SCALE_FACTOR)
    new_h = int(img.height * SCALE_FACTOR)
    img_big = img.resize((new_w, new_h), Image.LANCZOS)
    img.close(); del img

    scaled = [
        (int(x1*SCALE_FACTOR), int(y1*SCALE_FACTOR),
         int(x2*SCALE_FACTOR), int(y2*SCALE_FACTOR))
        for (x1, y1, x2, y2) in bboxes
    ]

    # 2. Original-image panel
    ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
    arr_big = np.array(img_big)
    ax_orig.imshow(arr_big, aspect='auto', interpolation='nearest')
    del arr_big                   # mpl has its own copy now

    ax_orig.axis('off'); ax_orig.margins(0)
    plt.setp(ax_orig.spines.values(), visible=False)
    ax_orig.set_xlim([0, new_w]); ax_orig.set_ylim([new_h, 0])

    for bi, (x1, y1, x2, y2) in enumerate(scaled):
        ax_orig.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.2, edgecolor=BBOX_COLORS[bi], facecolor='none'
        ))

    if is_first_col:
        ax_orig.text(-0.015, 0.5, label,
                     transform=ax_orig.transAxes,
                     fontsize=7, va='center', ha='right', rotation=90,
                     fontfamily='serif', fontweight='normal')

    # 3. Stacked crops panel
    ax_crops = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
    ax_crops.axis('off'); ax_crops.margins(0)
    plt.setp(ax_crops.spines.values(), visible=False)

    max_cw = max(sx2 - sx1 for (sx1, _, sx2, _) in scaled)

    total_h = 0
    tmp_rows = []
    for bi, (x1, y1, x2, y2) in enumerate(scaled):
        crop = img_big.crop((x1, y1, x2, y2))
        if crop.width < max_cw:
            sc = max_cw / crop.width
            tmp = crop.resize((max_cw, int(crop.height * sc)), Image.LANCZOS)
            crop.close(); crop = tmp

        bw = BORDER_WIDTH
        bordered = Image.new('RGB',
                              (crop.width + 2*bw, crop.height + 2*bw),
                              BBOX_COLORS_PIL[bi])
        bordered.paste(crop, (bw, bw))
        crop.close(); del crop
        tmp_rows.append(bordered)
        total_h += bordered.height

    max_rw = max(r.width for r in tmp_rows)
    stacked = Image.new('RGB', (max_rw, total_h), (255, 255, 255))
    y_off = 0
    for bordered in tmp_rows:
        stacked.paste(bordered, (0, y_off))
        y_off += bordered.height
        bordered.close(); del bordered
    del tmp_rows

    stacked_arr = np.array(stacked)
    stacked.close(); del stacked

    ax_crops.imshow(stacked_arr, aspect='auto', interpolation='nearest')
    ax_crops.set_xlim([0, stacked_arr.shape[1]])
    ax_crops.set_ylim([stacked_arr.shape[0], 0])
    del stacked_arr

    # 4. Free upscaled image
    img_big.close(); del img_big
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_comparison_figure(text_files, output_path):
    if len(text_files) != 3:
        raise ValueError("Exactly 3 text files required.")

    all_data = [parse_text_file(tf) for tf in text_files]
    num_rows = len(all_data[0][2])

    width_ratios = [1.0, 0.4, 1.0, 0.4, 1.0, 0.4]
    fig_width  = 8.0
    fig_height = fig_width * num_rows / 3.5 * 0.67

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        num_rows, 6,
        width_ratios=width_ratios,
        hspace=0.02, wspace=0.02,
        left=0.04, right=0.998, top=0.99, bottom=0.01
    )

    for col_idx, (bboxes, index, entries) in enumerate(all_data):
        input_path = next(
            (p for lbl, p in entries if lbl.lower().startswith('ip')), None
        )
        if input_path is None:
            raise ValueError(f"No 'ip f/2.8' entry in {text_files[col_idx]}")
        input_key = os.path.basename(input_path)

        # Get gt_size from first entry without keeping it in memory
        first_label, first_path = entries[0]
        tmp = load_one(first_label, first_path, input_key, None)
        gt_size = tmp.size
        tmp.close(); del tmp
        gc.collect()
        print(f"Col {col_idx}: gt_size={gt_size}, key={input_key}")

        for row_idx, (label, path) in enumerate(entries):
            print(f"  Row {row_idx} — {label} ... ", end='', flush=True)
            try:
                render_cell(fig, gs, row_idx, col_idx,
                            label, path, input_key, gt_size,
                            bboxes, is_first_col=(col_idx == 0))
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                fig.add_subplot(gs[row_idx, col_idx * 2]).axis('off')
                fig.add_subplot(gs[row_idx, col_idx * 2 + 1]).axis('off')

        gc.collect()

    print("Saving figure ...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.005)
    print(f"Saved -> {output_path}")
    plt.close()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text_files', nargs=3)
    parser.add_argument('--output', default='comparison_deblur.png')
    args = parser.parse_args()

    for tf in args.text_files:
        if not Path(tf).exists():
            print(f"Error: not found: {tf}"); sys.exit(1)

    generate_comparison_figure(args.text_files, args.output)


if __name__ == '__main__':
    main()

"""
generate_deblur_comparison_figure.py  (memory-efficient, correct aspect ratio)
===============================================================================
- Processes ONE cell at a time, frees all arrays immediately after imshow()
- Uses Liberation Serif (Times New Roman metric-compatible) with fallback chain
- Fixes squished images: axes are sized to match image aspect ratio exactly

Usage:
    python generate_deblur_comparison_figure.py \
        /path/scene1/roi_coords_deblur.txt \
        /path/scene2/roi_coords_deblur.txt \
        /path/scene3/roi_coords_deblur.txt \
        --output comparison_deblur.png
"""

# import os
# import sys
# import gc
# import argparse
# from pathlib import Path

# import h5py
# import numpy as np
# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.font_manager as fm
# from matplotlib import rcParams

# # ---------------------------------------------------------------------------
# # Font: prefer Times New Roman → Liberation Serif → DejaVu Serif → serif
# # ---------------------------------------------------------------------------
# def _pick_font():
#     preferred = ['Times New Roman', 'Liberation Serif',
#                  'FreeSerif', 'DejaVu Serif']
#     available = {f.name for f in fm.fontManager.ttflist}
#     for name in preferred:
#         if name in available:
#             print(f"[font] Using: {name}")
#             return name
#     print("[font] Falling back to generic serif")
#     return 'serif'

# FONT_NAME = _pick_font()
# rcParams['font.family'] = 'serif'
# rcParams['font.serif']  = [FONT_NAME, 'DejaVu Serif', 'serif']
# rcParams['axes.unicode_minus'] = False

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# BBOX_COLORS = [
#     (0.20, 1.00, 0.20),
#     (1.00, 0.20, 0.60),
#     (0.10, 0.45, 1.00),
# ]
# BBOX_COLORS_PIL = [
#     ( 51, 255,  51),
#     (255,  51, 153),
#     ( 26, 115, 255),
# ]

# SCALE_FACTOR = 1.5
# BORDER_WIDTH = 20


# # ---------------------------------------------------------------------------
# # Parsing
# # ---------------------------------------------------------------------------

# def parse_text_file(filepath):
#     with open(filepath, 'r') as f:
#         lines = [ln.strip() for ln in f if ln.strip()]

#     bboxes = []
#     for i in range(0, 6, 2):
#         y1, x1 = map(int, lines[i].split(','))
#         y2, x2 = map(int, lines[i + 1].split(','))
#         bboxes.append((x1, y1, x2, y2))

#     index = int(lines[6].split(',')[1].strip())

#     entries = []
#     for line in lines[7:]:
#         parts = line.split(',', 1)
#         if len(parts) == 2:
#             entries.append((parts[0].strip(), parts[1].strip()))

#     return bboxes, index, entries


# # ---------------------------------------------------------------------------
# # Loaders
# # ---------------------------------------------------------------------------

# def load_png(path, target_size=None):
#     img = Image.open(path).convert('RGB')
#     if target_size and img.size != target_size:
#         tmp = img.resize(target_size, Image.LANCZOS)
#         img.close(); img = tmp
#     return img


# def load_h5(h5_path, key, target_size=None):
#     with h5py.File(h5_path, 'r') as hf:
#         if key not in hf:
#             avail = list(hf.keys())[:5]
#             raise KeyError(f"Key '{key}' not in {h5_path}. First 5: {avail}")
#         arr = hf[key][:]

#     if arr.dtype != np.uint8:
#         arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
#     if arr.ndim == 2:
#         arr = np.stack([arr] * 3, axis=-1)

#     img = Image.fromarray(arr, 'RGB')
#     del arr

#     if target_size and img.size != target_size:
#         tmp = img.resize(target_size, Image.LANCZOS)
#         img.close(); img = tmp
#     return img


# def load_one(label, path, input_key, target_size):
#     ll = label.lower()
#     if ll.startswith('ip') or ll.startswith('gt'):
#         return load_png(path, target_size)
#     return load_h5(path, input_key, target_size)


# # ---------------------------------------------------------------------------
# # Render one cell — correct aspect ratio via imshow extent + equal axes
# # ---------------------------------------------------------------------------

# def render_cell(fig, gs, row_idx, col_idx,
#                 label, path, input_key, gt_size,
#                 bboxes, is_first_col):

#     # 1. Load + upscale
#     img = load_one(label, path, input_key, gt_size)
#     new_w = int(img.width  * SCALE_FACTOR)
#     new_h = int(img.height * SCALE_FACTOR)
#     img_big = img.resize((new_w, new_h), Image.LANCZOS)
#     img.close(); del img

#     scaled = [
#         (int(x1*SCALE_FACTOR), int(y1*SCALE_FACTOR),
#          int(x2*SCALE_FACTOR), int(y2*SCALE_FACTOR))
#         for (x1, y1, x2, y2) in bboxes
#     ]

#     # 2. Original-image panel — preserve true aspect ratio
#     ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
#     arr_big = np.array(img_big)
#     # Use 'equal' so pixels are square; extent locks axes to image dimensions
#     ax_orig.imshow(arr_big,
#                    aspect='equal',
#                    interpolation='nearest',
#                    extent=[0, new_w, new_h, 0])
#     del arr_big

#     ax_orig.set_xlim(0, new_w)
#     ax_orig.set_ylim(new_h, 0)
#     ax_orig.set_aspect('equal', adjustable='box')
#     ax_orig.axis('off')
#     ax_orig.margins(0)
#     plt.setp(ax_orig.spines.values(), visible=False)

#     for bi, (x1, y1, x2, y2) in enumerate(scaled):
#         ax_orig.add_patch(patches.Rectangle(
#             (x1, y1), x2-x1, y2-y1,
#             linewidth=1.2, edgecolor=BBOX_COLORS[bi], facecolor='none'
#         ))

#     if is_first_col:
#         ax_orig.text(-0.015, 0.5, label,
#                      transform=ax_orig.transAxes,
#                      fontsize=7, va='center', ha='right', rotation=90,
#                      fontname=FONT_NAME)

#     # 3. Stacked crops panel
#     ax_crops = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
#     ax_crops.axis('off')
#     ax_crops.margins(0)
#     plt.setp(ax_crops.spines.values(), visible=False)

#     max_cw = max(sx2 - sx1 for (sx1, _, sx2, _) in scaled)

#     total_h = 0
#     tmp_rows = []
#     for bi, (x1, y1, x2, y2) in enumerate(scaled):
#         crop = img_big.crop((x1, y1, x2, y2))
#         if crop.width < max_cw:
#             sc = max_cw / crop.width
#             tmp = crop.resize((max_cw, int(crop.height * sc)), Image.LANCZOS)
#             crop.close(); crop = tmp

#         bw = BORDER_WIDTH
#         bordered = Image.new('RGB',
#                               (crop.width + 2*bw, crop.height + 2*bw),
#                               BBOX_COLORS_PIL[bi])
#         bordered.paste(crop, (bw, bw))
#         crop.close(); del crop
#         tmp_rows.append(bordered)
#         total_h += bordered.height

#     max_rw = max(r.width for r in tmp_rows)
#     stacked = Image.new('RGB', (max_rw, total_h), (255, 255, 255))
#     y_off = 0
#     for bordered in tmp_rows:
#         stacked.paste(bordered, (0, y_off))
#         y_off += bordered.height
#         bordered.close(); del bordered
#     del tmp_rows

#     stacked_arr = np.array(stacked)
#     stacked.close(); del stacked

#     sh, sw = stacked_arr.shape[:2]
#     ax_crops.imshow(stacked_arr,
#                     aspect='equal',
#                     interpolation='nearest',
#                     extent=[0, sw, sh, 0])
#     ax_crops.set_xlim(0, sw)
#     ax_crops.set_ylim(sh, 0)
#     ax_crops.set_aspect('equal', adjustable='box')
#     del stacked_arr

#     img_big.close(); del img_big
#     gc.collect()


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def generate_comparison_figure(text_files, output_path):
#     if len(text_files) != 3:
#         raise ValueError("Exactly 3 text files required.")

#     all_data = [parse_text_file(tf) for tf in text_files]
#     num_rows = len(all_data[0][2])

#     # Width ratios: orig columns wider, crop columns narrower
#     width_ratios = [1.0, 0.4, 1.0, 0.4, 1.0, 0.4]
#     fig_width  = 8.0
#     fig_height = fig_width * num_rows / 3.5 * 0.67

#     fig = plt.figure(figsize=(fig_width, fig_height))
#     gs  = fig.add_gridspec(
#         num_rows, 6,
#         width_ratios=width_ratios,
#         hspace=0.02, wspace=0.02,
#         left=0.04, right=0.998, top=0.99, bottom=0.01
#     )

#     for col_idx, (bboxes, index, entries) in enumerate(all_data):
#         input_path = next(
#             (p for lbl, p in entries if lbl.lower().startswith('ip')), None
#         )
#         if input_path is None:
#             raise ValueError(f"No 'ip f/2.8' entry in {text_files[col_idx]}")
#         input_key = os.path.basename(input_path)

#         first_label, first_path = entries[0]
#         tmp = load_one(first_label, first_path, input_key, None)
#         gt_size = tmp.size
#         tmp.close(); del tmp
#         gc.collect()
#         print(f"Col {col_idx}: gt_size={gt_size}, key={input_key}")

#         for row_idx, (label, path) in enumerate(entries):
#             print(f"  Row {row_idx} — {label} ... ", end='', flush=True)
#             try:
#                 render_cell(fig, gs, row_idx, col_idx,
#                             label, path, input_key, gt_size,
#                             bboxes, is_first_col=(col_idx == 0))
#                 print("done")
#             except Exception as e:
#                 print(f"ERROR: {e}")
#                 import traceback; traceback.print_exc()
#                 fig.add_subplot(gs[row_idx, col_idx * 2]).axis('off')
#                 fig.add_subplot(gs[row_idx, col_idx * 2 + 1]).axis('off')

#         gc.collect()

#     print("Saving figure ...")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.005)
#     print(f"Saved -> {output_path}")
#     plt.close()
#     gc.collect()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('text_files', nargs=3)
#     parser.add_argument('--output', default='comparison_deblur.png')
#     args = parser.parse_args()

#     for tf in args.text_files:
#         if not Path(tf).exists():
#             print(f"Error: not found: {tf}"); sys.exit(1)

#     generate_comparison_figure(args.text_files, args.output)


# if __name__ == '__main__':
#     main()