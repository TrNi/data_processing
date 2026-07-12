"""
plot_K_sensitivity.py

Heatmap sensitivity plot for scene-group vs K-parameter sweep.
Rows in each CSV = different K values; one CSV per scene/condition group.

Label format expected:  bokehme_dfs20_K40_dispfocus0.15
K value is extracted from the _K<number>_ token in the label.
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

def _extract_K_value(name: str) -> float:
    """Extract the K-parameter value from a label like bokehme_dfs20_K40_dispfocus0.15."""
    match = re.search(r"_K(\d+(?:\.\d+)?)", name)
    if match:
        return float(match.group(1))
    # Fallback: first number in string
    match = re.search(r"[-+]?\d+(?:\.\d+)?", name)
    if match:
        return float(match.group())
    raise ValueError(f"Cannot extract K value from name: {name!r}")


def load_csv(csv_path: str):
    """
    Parse the CSV format:
      - Skip leading metadata/header rows.
      - Data rows: first token is the label, last three tokens are PSNR, SSIM, LPIPS.
    Returns a DataFrame with columns: [label, x_param, PSNR, SSIM, LPIPS].
    x_param is the K value extracted from the label.
    """
    records = []
    with open(csv_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            parts = [p.strip() for p in re.split(r"[,\t]+", line)]
            parts_noempty = [p for p in parts if p]
            if len(parts_noempty) < 4:
                continue
            try:
                lpips = float(parts_noempty[-1])
                ssim  = float(parts_noempty[-2])
                psnr  = float(parts_noempty[-3])
            except ValueError:
                continue
            label = parts_noempty[0]
            try:
                x_val = _extract_K_value(label)
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
# Plot -- Heatmap (Focal length-group x K-parameter-% bins)
# ---------------------------------------------------------------------------

def plot_heatmap(
    all_dfs: list,
    out_dir: str,
    fig_title: str = "",
    out_stem: str = "sensitivity_K",
    n_bins: int = 10,
    xlabel: str = "Focal length (mm)",
    norm_gamma: float = 0.4,
):
    """
    X-axis = Focal length/condition groups (one per CSV).
    Y-axis = % deviation from each group's optimal K value.
    Cell colour = % metric degradation.  One panel per metric.
    """
    _apply_jmlr_style()
    os.makedirs(out_dir, exist_ok=True)
    height = JMLR_TEXTWIDTH * 0.30

    fig, axes = plt.subplots(1, 3, figsize=(JMLR_TEXTWIDTH, height))
    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        show_ylabel = (idx == 0)
        _draw_heatmap_metric(ax, all_dfs, metric, n_bins,
                             show_ylabel=show_ylabel, xlabel=xlabel,
                             norm_gamma=norm_gamma)
    if fig_title:
        fig.suptitle(fig_title, fontsize=JMLR_FONT_SIZE + 1)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    _save(fig, out_dir, out_stem)


def _draw_heatmap_metric(ax, all_dfs, metric, n_bins,
                         show_ylabel=True, xlabel="Focal length (mm)", norm_gamma=0.4):
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    cmap_name = "turbo"

    # Symmetric %-deviation axis centred on 0
    all_pct_devs = []
    for df, ideal_1based, group_label, _ in all_dfs:
        opt_x = df.loc[ideal_1based - 1, "x_param"]
        all_pct_devs.extend(((df["x_param"] - opt_x) / (abs(opt_x) + 1e-12) * 100.0).tolist())
    _pct_min = min(all_pct_devs)
    _pct_max = max(all_pct_devs)
    abs_max  = max(abs(_pct_min), abs(_pct_max))
    y_min = int(np.floor(_pct_min / 25.0)) * 25
    y_max = int(np.ceil (_pct_max / 25.0)) * 25
    if n_bins % 2 == 0:
        n_bins += 1
    bin_edges   = np.linspace(-abs_max, abs_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    group_labels = []
    fl_annotations = []
    grid = []
    for df, ideal_1based, group_label, _ in all_dfs:
        ideal_idx = ideal_1based - 1
        opt_x   = df.loc[ideal_idx, "x_param"]
        opt_val = df.loc[ideal_idx, metric]
        pct_dev = (df["x_param"] - opt_x) / (abs(opt_x) + 1e-12) * 100.0
        raw_deg = (opt_val - df[metric]) if higher_better else (df[metric] - opt_val)
        pct_deg = (raw_deg / (abs(opt_val) + 1e-12) * 100.0).clip(lower=0.0)

        dev_min, dev_max = pct_dev.min(), pct_dev.max()
        _eps = 1e-9
        drawn_bc = bin_centres[(bin_centres >= dev_min - _eps) & (bin_centres <= dev_max + _eps)]
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
            if dev_min - _eps <= bc <= dev_max + _eps:
                col[b] = np.interp(bc, pct_dev_s.values, pct_deg_s.values)

        grid.append(col)
        group_labels.append(str(group_label))

    grid = np.array(grid).T  # shape: (n_bins, n_groups)

    valid = grid[~np.isnan(grid)]
    vmin = max(float(np.nanmin(grid)), 0.0)
    vmax = float(np.nanmax(grid)) if len(valid) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    n_groups = len(group_labels)
    fl_indices = np.arange(n_groups, dtype=float)
    cell_w    = 0.6
    half_cell = cell_w / 2

    base_cmap = plt.colormaps[cmap_name]
    cmap = _LSC.from_list(
        cmap_name + "_clipped",
        base_cmap(np.linspace(0.15, 0.98, 256))
    )
    # if metric == "LPIPS":
    #     norm = _mcolors.PowerNorm(gamma=norm_gamma, vmin=vmin, vmax=vmax)
    # else:
    norm = _mcolors.Normalize(vmin=vmin, vmax=vmax)

    bin_h = bin_edges[1] - bin_edges[0]
    for fi in range(n_groups):
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

    # Annotations: optimal row and max-deviation row
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
    half_bin = (bin_edges[1] - bin_edges[0]) / 2
    ax.set_ylim(y_min - half_bin, y_max + half_bin)

    tick_fs = JMLR_TICK_SIZE * 0.75
    ax.set_xticks(fl_indices)
    ax.set_xticklabels(group_labels, fontsize=tick_fs)
    ax.set_xlabel(xlabel, fontsize=tick_fs, labelpad=1)
    ax.tick_params(axis="x", length=2, pad=1)

    yticks = list(range(y_min, y_max + 1, 25))
    ax.set_yticks(yticks)
    def _fmt_ytick(v):
        return f"\u2212{abs(v)}" if v < 0 else str(v)
    if show_ylabel:
        ax.set_yticklabels([_fmt_ytick(t) for t in yticks], fontsize=tick_fs)
        ax.set_ylabel("% deviation from optimal K", fontsize=tick_fs, labelpad=0.5)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.tick_params(axis="y", length=2, pad=1)
    ax.axhline(0, color="#555", linewidth=0.6, linestyle="--", zorder=3)
    ax.set_title(metric, fontsize=JMLR_LABEL_SIZE, pad=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, shrink=1.0)
    cb.set_label("")
    cb.ax.tick_params(labelsize=tick_fs - 1, length=2, pad=1)
    cb.ax.set_title("degradation \n (%)", fontsize=tick_fs - 1, pad=2)


# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------

# Each entry: (csv_path, ideal_row, group_label, legend_label)
#   csv_path    : path to the CSV file
#   ideal_row   : 1-based row index (sorted by K) to use as reference
#   group_label : label shown on x-axis for this CSV (e.g. Focal length)
#   legend_label: unused, keep "" or set a description
CSVS = [
    (r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehme_dispfocus0.25_dfs_20.csv", 1, 28, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehme.csv", 1, 36, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehme.csv", 1, 45, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehme.csv", 1, 60, ""),
    (r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehme.csv", 1, 70, ""),
    # (r"C:/path/to/fl_50.csv", 6, 50, "FL 50mm"),
]
# CSVS = [
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_28\drbokeh_fp0.3.csv", 1, 28, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_drbokeh_all.csv", 1, 36, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_drbokeh_all.csv", 1, 45, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_drbokeh_all.csv", 1, 60, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_drbokeh_all.csv", 1, 70, ""),
#     # (r"C:/path/to/fl_50.csv", 6, 50, "FL 50mm"),
# ]

# CSVS = [
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehdiff_fp0.3.csv", 1, 28, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehdiff.csv", 1, 36, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehdiff.csv", 1, 45, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehdiff.csv", 1, 60, ""),
#     (r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehdiff.csv", 1, 70, ""),
#     # (r"C:/path/to/fl_50.csv", 6, 50, "FL 50mm"),
# ]

OUT_DIR      = r"I:\My Drive\DOF_benchmarking\inference\DoF_sensitivity_plots"
PLOT_TITLE   = ""
OUT_STEM     = "sensitivity_bokehme"
HEATMAP_BINS = 10
NORM_GAMMA   = 1            # colormap power-norm gamma (<1 expands low-degradation colours)
XLABEL       = "Focal length (mm)"

# ---------------------------------------------------------------------------


def main():
    all_dfs = []
    for csv_path_raw, ideal_row, group_label, legend_label in CSVS:
        csv_path = os.path.abspath(csv_path_raw)
        print(f"\n--- Loading: {csv_path} (group={group_label}) ---")
        df = load_csv(csv_path)
        print(f"Loaded {len(df)} data rows")
        print(df[["label", "x_param", "PSNR", "SSIM", "LPIPS"]].to_string(index=True))
        all_dfs.append((df, ideal_row, group_label, legend_label))

    first_csv = os.path.abspath(CSVS[0][0])
    out_dir   = OUT_DIR if OUT_DIR else os.path.dirname(first_csv)

    plot_heatmap(
        all_dfs=all_dfs,
        out_dir=out_dir,
        fig_title=PLOT_TITLE,
        out_stem=OUT_STEM,
        n_bins=HEATMAP_BINS,
        xlabel=XLABEL,
        norm_gamma=NORM_GAMMA,
    )


if __name__ == "__main__":
    main()
