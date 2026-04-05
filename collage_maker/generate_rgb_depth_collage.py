'''
Generate a 2-row (RGB / depth) x 3-column collage for paper publication.

Usage:
    python generate_rgb_depth_collage.py \
        rgb1.jpg rgb2.jpg rgb3.jpg \
        depth1.npy depth2.npy depth3.npy \
        --names "GT" "Ours" "Baseline" \
        --output collage.png
'''

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# ── Font ──────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# ── Page dimensions ───────────────────────────────────────────────────────────
PAGE_WIDTH = 7           # inches (paper page width)
FIG_WIDTH  = PAGE_WIDTH * 0.67   # ≈ 4.69 in

# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_depth(path, key=None):
    """Load a depth map from .npy / .npz / .h5 / .hdf5."""
    p   = Path(path)
    ext = p.suffix.lower()
    if ext == '.npy':
        return np.load(str(p)).astype(np.float32)
    if ext == '.npz':
        data = np.load(str(p))
        k = key if (key and key in data) else list(data.keys())[0]
        return data[k].astype(np.float32)
    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(str(p), 'r') as f:
            k = key if (key and key in f) else list(f.keys())[0]
            return f[k][()].astype(np.float32)
    raise ValueError(f"Unsupported depth format: {ext!r}")


# ── Main figure generator ─────────────────────────────────────────────────────
def generate_collage(rgb_paths, depth_paths, names, output_path,
                     depth_key=None, unit='m'):

    # ── Load data ──────────────────────────────────────────────────────────
    rgb_imgs = [np.array(Image.open(p).convert('RGB')) for p in rgb_paths]
    depths   = [load_depth(p, depth_key) for p in depth_paths]

    # ── Global vmin/vmax: 5th–95th percentile across all three depth maps ──
    all_valid = np.concatenate([d[np.isfinite(d)].ravel() for d in depths])
    vmin = float(np.percentile(all_valid, 5))
    vmax = float(np.percentile(all_valid, 95))

    # ── Per-depth 5–95th pct ranges for titles ─────────────────────────────
    depth_ranges = []
    for d in depths:
        v = d[np.isfinite(d)].ravel()
        depth_ranges.append((float(np.percentile(v, 5)),
                             float(np.percentile(v, 95))))

    # ── Figure geometry ────────────────────────────────────────────────────
    #
    #   fig_width is fixed.  Derive row height from the first RGB image's
    #   aspect ratio, then compute the exact figure height so that:
    #     - each image row occupies exactly cell_h inches
    #     - the inter-row gap fits the 2-line depth title + a tiny whitespace
    #
    fig_width = FIG_WIDTH                          # 4.69 in

    h0, w0   = rgb_imgs[0].shape[:2]
    img_ar   = h0 / w0                            # height / width ratio
    col_w_in = fig_width / 3.0                   # approx per-column width
    cell_h   = col_w_in * img_ar                 # row height in inches

    TITLE_FONT = 5.5                              # pt
    title_h_in = 1 * (TITLE_FONT / 72) #+ 0.04   # 1-line title + small pad
    row_gap_in = 0.02                             # whitespace above the title

    # GridSpec hspace is fraction of average row height consumed as gap
    m      = 0.003                                # tiny figure margin (fraction)
    hspace = (title_h_in + row_gap_in) / cell_h
    fig_h  = cell_h * (2 + hspace) / (1 - 2 * m)

    # ── Build figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_width, fig_h))

    cb_wr = 0.055   # colorbar col width relative to one image column
    gs = gridspec.GridSpec(
        2, 4,
        width_ratios=[1, 1, 1, cb_wr],
        height_ratios=[1, 1],
        hspace=hspace,
        wspace=0.018,
        left=m, right=1 - m,
        top=1 - m, bottom=m,
    )

    # ── Row 0: RGB images ──────────────────────────────────────────────────
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(rgb_imgs[col], aspect='auto', interpolation='lanczos')
        ax.axis('off')

    # ── Row 1: depth maps ──────────────────────────────────────────────────
    im = None
    for col in range(3):
        p5, p95 = depth_ranges[col]
        title   = f"{names[col]} ({p5:.1f}–{p95:.1f} {unit})"
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(title,
                     fontsize=TITLE_FONT, pad=1.2,
                     fontfamily='serif', linespacing=1)
        im = ax.imshow(depths[col],
                       cmap='turbo_r', vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        ax.axis('off')

    # ── Colorbar (right of depth row) ──────────────────────────────────────
    ax_cb = fig.add_subplot(gs[1, 3])
    cb = fig.colorbar(im, cax=ax_cb, orientation='vertical')
    cb.locator = MultipleLocator(0.5)
    cb.update_ticks()
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    cb.ax.tick_params(labelsize=4.5, length=2, pad=.7, width=0.3)

    # ── Save ───────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=450, bbox_inches='tight', pad_inches=0.005)
    print(f"Saved → {output_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='2-row (RGB + depth) x 3-col publication collage.'
    )
    parser.add_argument('rgb',   nargs=3, metavar='RGB',
                        help='Three RGB image paths')
    parser.add_argument('depth', nargs=3, metavar='DEPTH',
                        help='Three depth-map paths (.npy / .npz / .h5)')
    parser.add_argument('--names', nargs=3, metavar='NAME',
                        default=['Depth 1', 'Depth 2', 'Depth 3'],
                        help='Depth-map names shown in subplot titles')
    parser.add_argument('--output', '-o', default='collage.png',
                        help='Output file path (default: collage.png)')
    parser.add_argument('--depth-key', default=None,
                        help='Dataset key inside .npz / .h5 files')
    parser.add_argument('--unit', default='m',
                        help='Depth unit shown in titles (default: m)')

    args = parser.parse_args()

    for p in args.rgb + args.depth:
        if not Path(p).exists():
            print(f"Error: not found: {p}")
            sys.exit(1)

    generate_collage(
        args.rgb, args.depth, args.names, args.output,
        depth_key=args.depth_key, unit=args.unit,
    )


if __name__ == '__main__':
    main()
