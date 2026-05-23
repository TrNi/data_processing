# -*- coding: utf-8 -*-
import os
import csv
import fnmatch

import cv2
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# -------- Helpers --------
def get_image_files(directory):
    """Returns a sorted list of image files in a directory."""
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def load_image(path):
    """Loads an image, converts BGR to RGB, and normalizes to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# -------- Core metric computation --------
def compute_metrics_for_folder(pred_dir, gt_dir, lpips_fn, device,
                                downscale_factor=2, show_debug=False, folder_label=""):
    """
    Computes average PSNR, SSIM, LPIPS for all matched image pairs in pred_dir vs gt_dir.

    Returns:
        (mean_psnr, mean_ssim, mean_lpips) or None if no pairs found.
    """
    pred_files = get_image_files(pred_dir)
    gt_files   = get_image_files(gt_dir)
    num_pairs  = min(len(pred_files), len(gt_files))

    if num_pairs == 0:
        print(f"  Skipping '{folder_label}': No pairs found "
              f"(Preds: {len(pred_files)}, GTs: {len(gt_files)})")
        return None

    print(f"  Evaluating {num_pairs} pairs (downscale factor: {downscale_factor}x)...")

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for idx in range(num_pairs):
        pred = load_image(os.path.join(pred_dir, pred_files[idx]))
        gt   = load_image(os.path.join(gt_dir,   gt_files[idx]))

        h_gt, w_gt, _ = gt.shape
        new_h = h_gt // downscale_factor
        new_w = w_gt // downscale_factor

        # cv2.INTER_AREA is recommended for downsampling to avoid aliasing/moiré
        gt_down   = cv2.resize(gt,   (new_w, new_h), interpolation=cv2.INTER_AREA)
        pred_down = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ----- PSNR -----
        psnr = peak_signal_noise_ratio(gt_down, pred_down, data_range=1.0)

        # ----- SSIM -----
        ssim = structural_similarity(gt_down, pred_down, channel_axis=2, data_range=1.0)

        # ----- LPIPS -----
        # Convert to [1, C, H, W] and scale to [-1, 1] for LPIPS
        gt_t = torch.from_numpy(gt_down).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        pr_t = torch.from_numpy(pred_down).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1

        with torch.no_grad():
            lp = lpips_fn(gt_t, pr_t).item()

        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lpips_vals.append(lp)

        # ----- Visual sanity check (only first image per folder) -----
        if show_debug and idx == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(gt_down)
            axes[0].set_title(f"GT  ({new_w}x{new_h})")
            axes[0].axis("off")
            axes[1].imshow(pred_down)
            axes[1].set_title(f"Pred ({new_w}x{new_h})")
            axes[1].axis("off")
            plt.suptitle(folder_label)
            plt.tight_layout()
            plt.show()

    return np.mean(psnr_vals), np.mean(ssim_vals), np.mean(lpips_vals)


# -------- CSV writer --------
def _write_csv(output_csv, root_dir, method_name, subfolder_pattern,
               pred_subdir, gt_path, results):
    """
    Writes metric results to a CSV.

    CSV layout:
      - Info rows (root_dir, method_name, subfolder_pattern, pred_subdir, gt_path)
      - Blank row
      - Header row : folder_name, [5 empty cols], PSNR, SSIM, LPIPS
      - K data rows: folder_name, [5 empty cols], psnr_val, ssim_val, lpips_val
    """
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(out_dir, exist_ok=True)

    EMPTY = ["", "", "", "", ""]  # 5 empty columns

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # --- Info header ---
        writer.writerow(["root_dir",           root_dir])
        writer.writerow(["method_name",        method_name])
        writer.writerow(["subfolder_pattern",  subfolder_pattern])
        writer.writerow(["pred_subdir",        pred_subdir])
        writer.writerow(["gt_path",            gt_path])
        writer.writerow([])

        # --- Column header ---
        writer.writerow(["folder_name"] + EMPTY + ["PSNR", "SSIM", "LPIPS"])

        # --- Data rows (all metrics side-by-side) ---
        for folder, psnr, ssim, lp in results:
            psnr_val = f"{psnr:.4f}" if psnr is not None else "N/A"
            ssim_val = f"{ssim:.4f}" if ssim is not None else "N/A"
            lp_val   = f"{lp:.4f}"   if lp   is not None else "N/A"
            writer.writerow([folder] + EMPTY + [psnr_val, ssim_val, lp_val])


# -------- Main evaluation function --------
def evaluate_method(
    root_dir,
    method_name,
    subfolder_pattern,
    pred_subdir,
    gt_path,
    output_csv,
    downscale_factor=2,
    show_debug=False,
    device=None,
):
    """
    Evaluates a method across all subfolders of root_dir that match subfolder_pattern.

    For each matched subfolder, computes average PSNR, SSIM and LPIPS between images
    in <subfolder>/<pred_subdir> and gt_path. Results are printed and saved to output_csv.

    Args:
        root_dir:           Directory containing candidate subfolders (e.g. /inference).
        method_name:        Label for this method, used in CSV header (e.g. "DrBokeh").
        subfolder_pattern:  Glob pattern to match subfolders (e.g. "fl_*").
        pred_subdir:        Prediction subfolder name within each matched subfolder.
                            Supports nested paths (e.g. "bokehdiff/demo_aligned").
        gt_path:            Ground-truth absolute folder path.
        output_csv:         CSV filename; written inside root_dir.
        downscale_factor:   Both images are downscaled by this factor before metric
                            computation (default 2). Uses cv2.INTER_AREA.
        show_debug:         If True, displays the first pred/GT pair for each subfolder.
        device:             Torch device string ("cuda" / "cpu"); auto-detected if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading LPIPS (VGG) on {device}...")
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # Discover matching subfolders
    try:
        entries = sorted(os.listdir(root_dir))
    except FileNotFoundError:
        print(f"root_dir not found: {root_dir}")
        return
    print(entries)
    matched = [
        e for e in entries
        if os.path.isdir(os.path.join(root_dir, e)) and fnmatch.fnmatch(e, subfolder_pattern)
    ]

    if not matched:
        print(f"No subfolders matching '{subfolder_pattern}' found in '{root_dir}'.")
        return

    print(f"\nFound {len(matched)} subfolder(s): {matched}")
    print(f"Method : {method_name}")
    print(f"Device : {device}\n")

    results = []  # list of (folder_name, psnr|None, ssim|None, lpips|None)
    output_csv = os.path.join(root_dir, output_csv)

    for folder in matched:
        print(f"\n{'='*40}")
        print(f"Processing: {folder}")
        print("="*40)

        pred_dir = os.path.join(root_dir, folder, pred_subdir)

        metrics = compute_metrics_for_folder(
            pred_dir, gt_path, lpips_fn, device,
            downscale_factor=downscale_factor,
            show_debug=show_debug,
            folder_label=folder,
        )

        if metrics is not None:
            psnr, ssim, lp = metrics
            results.append((folder, psnr, ssim, lp))
            print(f"  Mean PSNR  : {psnr:.3f}")
            print(f"  Mean SSIM  : {ssim:.4f}")
            print(f"  Mean LPIPS : {lp:.4f}")
        else:
            results.append((folder, None, None, None))

    # Grand average
    valid = [(f, p, s, l) for f, p, s, l in results if p is not None]
    if valid:
        all_p = [r[1] for r in valid]
        all_s = [r[2] for r in valid]
        all_l = [r[3] for r in valid]
        print(f"\n{'#'*40}")
        print(f"GRAND AVERAGE ACROSS {len(valid)} SUBFOLDER(S)  [{method_name}]")
        print("#"*40)
        print(f"Overall Mean PSNR  : {np.mean(all_p):.3f}")
        print(f"Overall Mean SSIM  : {np.mean(all_s):.4f}")
        print(f"Overall Mean LPIPS : {np.mean(all_l):.4f}")
    else:
        print("\nNo pairs were evaluated across any subfolder.")

    # Save CSV
    _write_csv(output_csv, root_dir, method_name, subfolder_pattern,
               pred_subdir, gt_path, results)
    print(f"\nResults saved to: {output_csv}")


# ================= USAGE EXAMPLE =================
if __name__ == "__main__":
    evaluate_method(
        root_dir          = "/path/to/inference",
        method_name       = "DrBokeh",
        subfolder_pattern = "fl_*",
        pred_subdir       = "Drbokeh_K25_fp0p25_aligned",
        gt_path           = "/path/to/inference/GT_folder",
        output_csv        = "results_drbokeh.csv",
        downscale_factor  = 2,
        show_debug        = False,
    )
