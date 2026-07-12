"""
plot_param_sensitivity.py

Heatmap sensitivity plot for focal-length vs F-number parameter sweep.
Rows in each CSV = different F-number values; one CSV per focal length.
"""

import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from matplotlib.colors import LinearSegmentedColormap as _LSC
import matplotlib.patheffects as _pe
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_param_value(name: str) -> float:
    """Try to pull the first numeric value (int or float) from a folder name."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", name)
    if match:
        return float(match.group())
    raise ValueError(f"Cannot extract numeric parameter from name: {name!r}")


def load_csv(csv_path: str):
    """
    Parse the CSV format:
      - Skip leading metadata/header rows (lines that don't start with a data row).
      - Data rows: first token is the folder name, last three tokens are PSNR, SSIM, LPIPS.
    Returns a DataFrame with columns: [label, x_param, PSNR, SSIM, LPIPS].
    """
    records = []
    with open(csv_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            # Split on comma or tab
            parts = [p.strip() for p in re.split(r"[,\t]+", line)]
            # Filter out completely empty parts caused by repeated delimiters
            parts_noempty = [p for p in parts if p]
            if len(parts_noempty) < 4:
                continue
            # Last three non-empty parts should be numeric metrics
            try:
                lpips = float(parts_noempty[-1])
                ssim = float(parts_noempty[-2])
                psnr = float(parts_noempty[-3])
            except ValueError:
                continue
            label = parts_noempty[0]
            try:
                x_val = _extract_param_value(label)
            except ValueError:
                x_val = float("nan")
            records.append({"label": label, "x_param": x_val,
                            "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips})

    if not records:
        raise ValueError(f"No valid data rows found in {csv_path!r}")

    df = pd.DataFrame(records)
    df = df.sort_values("x_param").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Metric config
# ---------------------------------------------------------------------------

METRIC_HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# JMLR single-column text width in inches
JMLR_TEXTWIDTH  = 6.75
JMLR_FONT_SIZE  = 9
JMLR_LABEL_SIZE = 9
JMLR_TICK_SIZE  = 8

METRICS = ["PSNR", "SSIM", "LPIPS"]


def _apply_jmlr_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         JMLR_FONT_SIZE,
        "axes.labelsize":    JMLR_LABEL_SIZE,
        "xtick.labelsize":   JMLR_TICK_SIZE,
        "ytick.labelsize":   JMLR_TICK_SIZE,
        "legend.fontsize":   JMLR_TICK_SIZE,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth":   1.2,
        "axes.grid":         True,
        "grid.linestyle":    ":",
        "grid.linewidth":    0.4,
        "grid.alpha":        0.6,
        "figure.dpi":        300,
    })


def _save(fig, out_dir, stem):
    for ext in (".pdf", ".png"):
        p = os.path.join(out_dir, stem + ext)
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot -- Heatmap (FL x parameter-% bins)
# ---------------------------------------------------------------------------

def plot_heatmap(
    all_dfs: list,
    out_dir: str,
    fig_title: str = "",
    out_stem: str = "sensitivity",
    n_bins: int = 10,
    norm_gamma: float = 0.4,
):
    """
    Rows = focal lengths, columns = %-deviation bins from optimal parameter.
    Cell colour = % metric degradation.  One panel per metric.
    """
    _apply_jmlr_style()
    os.makedirs(out_dir, exist_ok=True)
    height = JMLR_TEXTWIDTH * 0.30

    fig, axes = plt.subplots(1, 3, figsize=(JMLR_TEXTWIDTH, height))
    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        show_ylabel = (idx == 0)
        _draw_heatmap_metric(ax, all_dfs, metric, n_bins,
                             show_ylabel=show_ylabel, norm_gamma=norm_gamma)
    if fig_title:
        fig.suptitle(fig_title, fontsize=JMLR_FONT_SIZE + 1)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    _save(fig, out_dir, out_stem ) #+ "_combined")


def _draw_heatmap_metric(ax, all_dfs, metric, n_bins,
                         show_ylabel=True, norm_gamma=0.4):
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    # Colormap: turbo_r for higher-is-better (PSNR, SSIM), turbo for LPIPS
    cmap_name = "turbo" #if higher_better else "turbo_r"

    # Symmetric %-deviation axis centred on 0
    all_pct_devs = []
    for df, ideal_1based, focal_length, fl_label in all_dfs:
        opt_x = df.loc[ideal_1based - 1, "x_param"]
        all_pct_devs.extend(((df["x_param"] - opt_x) / (abs(opt_x) + 1e-12) * 100.0).tolist())
    abs_max = max(abs(min(all_pct_devs)), abs(max(all_pct_devs)))
    # Ensure n_bins is odd so 0 lands on a bin centre
    if n_bins % 2 == 0:
        n_bins += 1
    bin_edges   = np.linspace(-abs_max, abs_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fl_values = []
    fl_annotations = []  # list of (pct_dev_min, F_min, v_min, pct_dev_max, F_max, v_max)
    grid = []  # will be shape (n_bins, n_fl) after transpose
    for df, ideal_1based, focal_length, fl_label in all_dfs:
        ideal_idx = ideal_1based - 1
        opt_x   = df.loc[ideal_idx, "x_param"]
        opt_val = df.loc[ideal_idx, metric]
        pct_dev = (df["x_param"] - opt_x) / (abs(opt_x) + 1e-12) * 100.0
        raw_deg = (opt_val - df[metric]) if higher_better else (df[metric] - opt_val)
        pct_deg = (raw_deg / (abs(opt_val) + 1e-12) * 100.0).clip(lower=0.0)

        # annotation at 0% dev (optimal row) and at max |% dev|;
        dev_min, dev_max = pct_dev.min(), pct_dev.max()
        drawn_bc = bin_centres[(bin_centres >= dev_min) & (bin_centres <= dev_max)]
        i_opt = ideal_idx
        y_opt = drawn_bc[np.argmin(np.abs(drawn_bc - 0.0))]
        extremes = []
        pos_mask = pct_dev > 0
        neg_mask = pct_dev < 0
        if pos_mask.any():
            i_pos = pct_dev[pos_mask].idxmax()
            y_pos = drawn_bc[np.argmin(np.abs(drawn_bc - pct_dev[i_pos]))]
            extremes.append((y_pos, df.loc[i_pos, metric]))
        if neg_mask.any():
            i_neg = pct_dev[neg_mask].idxmin()
            y_neg = drawn_bc[np.argmin(np.abs(drawn_bc - pct_dev[i_neg]))]
            extremes.append((y_neg, df.loc[i_neg, metric]))
        fl_annotations.append((y_opt, df.loc[i_opt, metric], extremes))
        pct_dev_s = pct_dev.sort_values()
        pct_deg_s = pct_deg.loc[pct_dev_s.index]
        col = np.full(n_bins, np.nan)
        for b in range(n_bins):
            bc = bin_centres[b]
            if dev_min <= bc <= dev_max:
                col[b] = np.interp(bc, pct_dev_s.values, pct_deg_s.values)

        grid.append(col)
        fl_values.append(focal_length)

    # grid[i] = one FL's column vector; transpose so rows=bins, cols=FLs
    grid = np.array(grid).T  # shape: (n_bins, n_fl)

    valid = grid[~np.isnan(grid)]
    vmin = max(float(np.nanmin(grid)), 0.0)
    vmax = float(np.nanmax(grid)) if len(valid) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    n_fl = len(fl_values)
    # Use integer indices so all FLs are equally spaced regardless of mm values
    fl_indices = np.arange(n_fl, dtype=float)
    cell_w  = 0.6   # fraction of unit spacing occupied by cell
    half_cell = cell_w / 2

    # Slice the colormap to exclude dark ends (keep middle 5%-75% of turbo)
    base_cmap = plt.colormaps[cmap_name]
    cmap = _LSC.from_list(
        cmap_name + "_clipped",
        base_cmap(np.linspace(0.15, 0.98, 256))
    )
    if metric == "LPIPS":
        norm = _mcolors.PowerNorm(gamma=norm_gamma, vmin=vmin, vmax=vmax)
    else:
        # norm = _mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm = _mcolors.PowerNorm(gamma=0.85, vmin=vmin, vmax=vmax)

    bin_h = bin_edges[1] - bin_edges[0]   # height of one bin in %-dev units
    for fi in range(n_fl):
        for b in range(n_bins):
            val = grid[b, fi]
            if np.isnan(val):
                continue
            color = cmap(norm(val))
            rect = mpatches.Rectangle(
                (fl_indices[fi] - half_cell, bin_edges[b]),
                2 * half_cell, bin_h,
                facecolor=color, edgecolor="none", zorder=2
            )
            ax.add_patch(rect)

    # Annotate min and max x_param positions per FL
    ann_fs = JMLR_TICK_SIZE * 0.65
    for fi, (y_opt, v_opt, extremes) in enumerate(fl_annotations):
        x = fl_indices[fi]
        for y, v in [(y_opt, v_opt)] + extremes:
            if metric == "PSNR":
                label = f"{v:.1f}"
            else:
                label = f"{v:.2f}".lstrip("0")
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=ann_fs, color="orange",
                linespacing=1.1, zorder=5,
                path_effects=[
                    _pe.Stroke(linewidth=1.5, foreground="black"),
                    _pe.Normal(),
                ],
            )

    # Axes limits and ticks
    ax.set_xlim(fl_indices[0] - 0.6, fl_indices[-1] + 0.6)
    ax.set_ylim(bin_edges[0], bin_edges[-1])

    tick_fs = JMLR_TICK_SIZE * 0.75
    ax.set_xticks(fl_indices)
    ax.set_xticklabels([str(int(f)) for f in fl_values], fontsize=tick_fs)
    ax.set_xlabel("Focal length (mm)", fontsize=tick_fs, labelpad=1)
    ax.tick_params(axis="x", length=2, pad=1)

    yticks = [-100, -50, 0, 50, 100]
    ax.set_yticks(yticks)
    if show_ylabel:
        ax.set_yticklabels(["\u2212100", "\u221250", "0", "50", "100"], fontsize=tick_fs)
        ax.set_ylabel("% deviation from optimal F", fontsize=tick_fs, labelpad=0.5)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.tick_params(axis="y", length=2, pad=1)
    ax.axhline(0, color="#555", linewidth=0.6, linestyle="--", zorder=3)
    ax.set_title(metric, fontsize=JMLR_LABEL_SIZE, pad=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=1.0)
    cb.set_label("")  # no rotated label
    cb.ax.tick_params(labelsize=tick_fs - 1, length=2, pad=1)
    cb.ax.set_title("degradation \n (%)", fontsize=tick_fs - 2, pad=2)


# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------

# Each entry: (csv_path, ideal_row, focal_length, fl_label)
#   csv_path      : path to the CSV file
#   ideal_row     : 1-based row index (sorted by x_param) to use as reference
#   focal_length  : numeric focal length in mm (used as x-axis position in box plot)
#   fl_label      : legend label for the norm-degradation plot (empty -> filename)
CSVS = [
    (r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehlicious.csv", 9, 28, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehlicious.csv", 7, 36, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehlicious.csv", 8, 45, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehlicious.csv", 4, 60, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehlicious.csv", 6, 70, ""),
    # (r"C:/path/to/fl_50.csv", 6, 50, "FL 50mm"),
]

OUT_DIR    = r"I:\My Drive\DOF_benchmarking\inference\DoF_sensitivity_plots" # None → figures saved next to the first CSV
PLOT_TITLE = ""   # optional suptitle (empty → no title)
OUT_STEM     = "sensitivity_bokehlicious"  # output filename prefix
HEATMAP_BINS = 17             # number of %-deviation bins in the heatmap
NORM_GAMMA   = 0.7            # colormap power-norm gamma (<1 expands low-degradation colours)

# ---------------------------------------------------------------------------


def main():
    all_dfs = []
    for csv_path_raw, ideal_row, focal_length, fl_label in CSVS:
        csv_path = os.path.abspath(csv_path_raw)
        print(f"\n--- Loading: {csv_path} (FL={focal_length}mm) ---")
        df = load_csv(csv_path)
        print(f"Loaded {len(df)} data rows")
        print(df[["label", "x_param", "PSNR", "SSIM", "LPIPS"]].to_string(index=True))
        label = fl_label or os.path.splitext(os.path.basename(csv_path))[0]
        all_dfs.append((df, ideal_row, focal_length, label))

    first_csv = os.path.abspath(CSVS[0][0])
    out_dir   = OUT_DIR if OUT_DIR else os.path.dirname(first_csv)
    fig_title = PLOT_TITLE
    out_stem  = OUT_STEM

    plot_heatmap(
        all_dfs=all_dfs,
        out_dir=out_dir,
        fig_title=fig_title,
        out_stem=out_stem,
        n_bins=HEATMAP_BINS,
        norm_gamma=NORM_GAMMA,
    )


if __name__ == "__main__":
    main()
