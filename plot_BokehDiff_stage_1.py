"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
 
# ── Style to match reference figure ──────────────────────────────────────────
# Font sizes are scaled from the original 15-in figure to the ICCP page
# textwidth of 6.875 in using a sqrt-damped factor (√(6.875/15) ≈ 0.677)
# so text stays legible at 600 dpi while respecting physical proportions.
 
# ── Load & parse ──────────────────────────────────────────────────────────────
raw = pd.read_csv("./bokehdiff/results_bokehdiff_fl_28.csv", header=None)
 
data_rows = raw[raw[0].str.startswith("bokehdiff_K", na=False)].copy()
 
 
def parse_name(name):
    k  = int(re.search(r"K(\d+)",  name).group(1))
    fp = float(re.search(r"fp_([\d.]+)", name).group(1))
    return k, fp
 
 
parsed             = data_rows[0].apply(parse_name)
data_rows["K"]     = parsed.apply(lambda x: x[0])
data_rows["fp"]    = parsed.apply(lambda x: x[1])
data_rows["PSNR"]  = data_rows[6].astype(float)
data_rows["SSIM"]  = data_rows[7].astype(float)
data_rows["LPIPS"] = data_rows[8].astype(float)
 
df = data_rows[["K", "fp", "PSNR", "SSIM", "LPIPS"]].reset_index(drop=True)
 
# ── Grid ──────────────────────────────────────────────────────────────────────
K_vals  = sorted(df["K"].unique())
fp_vals = sorted(df["fp"].unique())
 
 
def make_grid(col):
    pivot = df.pivot(index="fp", columns="K", values=col)
    return pivot.reindex(index=fp_vals, columns=K_vals).values
 
 
metrics = {
    "PSNR":  {"grid": make_grid("PSNR"),  "colorscale": "turbo_r", "reversescale": False, "label": "PSNR (dB)", "better": "↑", "find_opt": "max"},
    "SSIM":  {"grid": make_grid("SSIM"),  "colorscale": "turbo_r", "reversescale": False, "label": "SSIM",      "better": "↑", "find_opt": "max"},
    "LPIPS": {"grid": make_grid("LPIPS"), "colorscale": "turbo", "reversescale": True, "label": "LPIPS",     "better": "↓", "find_opt": "min"},
}
 
# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
# From the official LaTeX template:
#   \textwidth  = 6.875 in   (full spanning-figure width)
#   \columnsep  = 0.3125 in  → single column ≈ 3.28 in
# PAMI camera-ready instructions recommend 600 ppi for figures.
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.42   # preserve aspect ratio of original script
 
# Plotly sizes in pixels; render at base 100 ppi and use write_image's `scale`
# argument to achieve the requested 600 ppi output (scale = DPI / BASE_PPI).
BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI * 0.65)
SCALE     = DPI / BASE_PPI
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.13,
)
 
# Calculate the exact normalized coordinate width of a 2-pixel space
PIXEL_SPACE = 2
cbar_pad_fraction = PIXEL_SPACE / WIDTH_PX
 
for i, (name, m) in enumerate(metrics.items()):
    Z   = m["grid"]
    col = i + 1

    # REFINED: Clip top 5% and bottom 5% percentiles to adjust dynamic range limits
    # This prevents extreme outliers from squishing the color gradient.
    z_min_clip = float(np.nanpercentile(Z, 5))
    z_max_clip = float(np.nanpercentile(Z, 95))

    # Dynamically locate the right edge of the current subplot's X-axis domain
    xaxis_key = f"xaxis{col}" if col > 1 else "xaxis"
    subplot_right_edge = fig.layout[xaxis_key].domain[1]
    
    # Precise position: Right edge + the 2-pixel fraction
    cbar_x_pos = subplot_right_edge + cbar_pad_fraction

    contour = go.Contour(
        z=Z,
        x=K_vals,
        y=fp_vals,
        colorscale=m["colorscale"],
        reversescale=m["reversescale"],
        zmin=z_min_clip,  # REFINED: Bound the bottom color dynamic range
        zmax=z_max_clip,  # REFINED: Bound the top color dynamic range
        ncontours=14,
        contours=dict(
            coloring="fill",
            showlines=True,
            showlabels=False,
            labelfont=dict(family="Times New Roman", size=5.0, color="#222222"),
        ),
        line=dict(color="#333333", width=0.4),
        colorbar=dict(
            title=dict(
                text=m["label"],
                font=dict(family="Times New Roman", size=5.8, color="#222222"),
                side="right",
            ),
            tickfont=dict(family="Times New Roman", size=5.1, color="#333333"),
            thickness=7,
            len=0.92,
            x=cbar_x_pos,        
            xanchor="left",      
            outlinewidth=0.6,
            outlinecolor="#555555",
            ticklen=2,
            tickwidth=0.5,
        ),
        showscale=True,
    )
    fig.add_trace(contour, row=1, col=col)
    
    # ── Add Optimal Hyperparameter Marker (Fixed Layering) ────────────────────
    if m["find_opt"] == "max":
        opt_idx = df[name].idxmax()
    else:
        opt_idx = df[name].idxmin()
        
    opt_k = df.loc[opt_idx, "K"]
    opt_fp = df.loc[opt_idx, "fp"]
    opt_val = df.loc[opt_idx, name]
    
    # Using go.Scattergl ensures it renders cleanly on a layer above the contour
    marker = go.Scattergl(
        x=[opt_k],
        y=[opt_fp],
        mode="markers",
        marker=dict(
            symbol="star",
            size=11,                  # Scaled up slightly to be highly visible at 600 DPI
            color="#ff1a1a",          # Ultra-vibrant red
            line=dict(color="#ffffff", width=1.5) # Thicker border for crisp separation
        ),
        name=f"Optimal {name}",
        hovertemplate=f"Optimal {name}<br>K: %{{x}}<br>fp: %{{y}}<br>Value: {opt_val:.4f}<extra></extra>"
    )
    fig.add_trace(marker, row=1, col=col)
    # ──────────────────────────────────────────────────────────────────────────
 
    fig.update_xaxes(
        title=dict(
            text="K  (kernel size)",
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickvals=K_vals,
        range=[K_vals[0] - 1.5, K_vals[-1] + 1.5],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )
    fig.update_yaxes(
        title=dict(
            text="fp  (focal plane)",
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickvals=fp_vals,
        range=[fp_vals[0] - 0.025, fp_vals[-1] + 0.025],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )
 
# Style and position the subplot titles (annotations)
# We map the text base directly to the top edge of the plot domain, then offset by exactly 3 pixels.
TITLE_GAP_PX = 3
subplot_top_edge = fig.layout.yaxis.domain[1]  # Shared top edge fraction across subplots

for ann in fig.layout.annotations:
    ann.font = dict(family="Times New Roman", size=6.8, color="#222222")
    ann.y = subplot_top_edge       # Anchors the text's coordinate to the exact top of the subplot
    ann.yanchor = "bottom"         # Forces the bottom of the font text box to sit on that coordinate
    ann.yshift = TITLE_GAP_PX      # Introduces the exact pixel padding upward

fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=45, r=15, t=20, b=20),
    showlegend=False,
)

# Save interactive HTML
out_html = "bokehdiff_stage_1.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")


out = "./bokehdiff_sstage_1.png"
# Requires `kaleido` (pip install -U kaleido) for static image export.
fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
print(f"Saved → {out}")
print(f"Size  : {TEXTWIDTH_IN:.3f} in × {FIG_HEIGHT:.3f} in  @  {DPI} dpi")
print(f"Pixels: {int(TEXTWIDTH_IN*DPI)} × {int(FIG_HEIGHT*DPI)}")
"""


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
 
# ── Style to match reference figure ──────────────────────────────────────────
# Font sizes are scaled from the original 15-in figure to the ICCP page
# textwidth of 6.875 in using a sqrt-damped factor (√(6.875/15) ≈ 0.677)
# so text stays legible at 600 dpi while respecting physical proportions.
 
# ── Load & parse ──────────────────────────────────────────────────────────────
raw = pd.read_csv("./bokehdiff/results_bokehdiff_fl_28.csv", header=None)
 
data_rows = raw[raw[0].str.startswith("bokehdiff_K", na=False)].copy()
 
 
def parse_name(name):
    k  = int(re.search(r"K(\d+)",  name).group(1))
    fp = float(re.search(r"fp_([\d.]+)", name).group(1))
    return k, fp
 
 
parsed             = data_rows[0].apply(parse_name)
data_rows["K"]     = parsed.apply(lambda x: x[0])
data_rows["fp"]    = parsed.apply(lambda x: x[1])
data_rows["PSNR"]  = data_rows[6].astype(float)
data_rows["SSIM"]  = data_rows[7].astype(float)
data_rows["LPIPS"] = data_rows[8].astype(float)
 
df = data_rows[["K", "fp", "PSNR", "SSIM", "LPIPS"]].reset_index(drop=True)
 
# ── Grid ──────────────────────────────────────────────────────────────────────
K_vals  = sorted(df["K"].unique())
fp_vals = sorted(df["fp"].unique())
 
 
def make_grid(col):
    pivot = df.pivot(index="fp", columns="K", values=col)
    return pivot.reindex(index=fp_vals, columns=K_vals).values
 
 
metrics = {
    "PSNR":  {"grid": make_grid("PSNR"),  "colorscale": "turbo_r", "reversescale": False, "label": "PSNR (dB)", "better": "↑", "find_opt": "max", "plot_opt": False},
    "SSIM":  {"grid": make_grid("SSIM"),  "colorscale": "turbo_r", "reversescale": False, "label": "SSIM",      "better": "↑", "find_opt": "max", "plot_opt": False},
    "LPIPS": {"grid": make_grid("LPIPS"), "colorscale": "turbo", "reversescale": False, "label": "LPIPS",     "better": "↓", "find_opt": "min", "plot_opt": True},
}
 
# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
# From the official LaTeX template:
#   \textwidth  = 6.875 in   (full spanning-figure width)
#   \columnsep  = 0.3125 in  → single column ≈ 3.28 in
# PAMI camera-ready instructions recommend 600 ppi for figures.
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.42   # preserve aspect ratio of original script
 
# Plotly sizes in pixels; render at base 100 ppi and use write_image's `scale`
# argument to achieve the requested 600 ppi output (scale = DPI / BASE_PPI).
BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI * 0.65)
SCALE     = DPI / BASE_PPI


#SUBPLOT_GAP_PX = 2.0
#SUBPLOT_SPACING_FRACTION = SUBPLOT_GAP_PX / WIDTH_PX


# ── Plot ──────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.1
)


# Calculate the exact normalized coordinate width of a 2-pixel space
PIXEL_SPACE = 0.5
cbar_pad_fraction = PIXEL_SPACE / WIDTH_PX

for i, (name, m) in enumerate(metrics.items()):
    Z   = m["grid"]
    col = i + 1

    # REFINED: Clip top 5% and bottom 5% percentiles to adjust dynamic range limits
    # This prevents extreme outliers from squishing the color gradient.
    z_min_clip = float(np.nanpercentile(Z, 5))
    z_max_clip = float(np.nanpercentile(Z, 95))

    # Dynamically locate the right edge of the current subplot's X-axis domain
    xaxis_key = f"xaxis{col}" if col > 1 else "xaxis"
    subplot_right_edge = fig.layout[xaxis_key].domain[1]
    
    # Precise position: Right edge + the 2-pixel fraction
    cbar_x_pos = subplot_right_edge - 0.015 + cbar_pad_fraction

    contour = go.Contour(
        z=Z,
        x=K_vals,
        y=fp_vals,
        colorscale=m["colorscale"],
        reversescale=m["reversescale"],
        zmin=z_min_clip,  # REFINED: Bound the bottom color dynamic range
        zmax=z_max_clip,  # REFINED: Bound the top color dynamic range
        ncontours=14,
        contours=dict(
            coloring="fill",
            showlines=True,
            showlabels=False,
            labelfont=dict(family="Times New Roman", size=5.0, color="#222222"),
        ),
        line=dict(color="#333333", width=0.4),
        colorbar=dict(
            #title=dict(
            #    text=m["label"],
            #    font=dict(family="Times New Roman", size=5.8, color="#222222"),
            #    side="right",
            #),
            tickfont=dict(family="Times New Roman", size=5.1, color="#333333"),
            thickness=7,
            len=0.92,
            x=cbar_x_pos,        
            xanchor="left",      
            outlinewidth=0.6,
            outlinecolor="#555555",
            ticklen=2,
            tickwidth=0.5,
        ),
        showscale=True,
    )
    fig.add_trace(contour, row=1, col=col)
    
    # ── Add Optimal Hyperparameter Marker (Only for targeted metrics) ─────────
    if m["plot_opt"]:
        # HARDCODED FIX: Overriding idxmin() to target your precise optimal configuration
        opt_k = 15          # Change this to the specific K value that pairs with fp=0.3 in your csv
        opt_fp = 0.3        # Your verified optimal focal plane
        
        # Extract the exact calculated value matching this coordinate pair from the DataFrame
        matched_row = df[(df["K"] == opt_k) & (df["fp"] == opt_fp)]
        
        if not matched_row.empty:
            opt_val = matched_row[name].values[0]
        else:
            # Fallback to idxmin if the exact pair isn't found in your dataframe
            opt_idx = df[name].idxmin()
            opt_k = df.loc[opt_idx, "K"]
            opt_fp = df.loc[opt_idx, "fp"]
            opt_val = df.loc[opt_idx, name]
        
        # Using go.Scattergl ensures it renders cleanly on a layer above the contour
        marker = go.Scattergl(
            x=[opt_k],
            y=[opt_fp],
            mode="markers",
            marker=dict(
                symbol="star",
                size=11,                  # Highly visible at 600 DPI
                color="#ff1a1a",          # Ultra-vibrant red
                line=dict(color="#ffffff", width=1.5) # Thicker border for crisp separation
            ),
            name=f"Optimal {name}",
            hovertemplate=f"Optimal {name}<br>K: %{{x}}<br>fp: %{{y}}<br>Value: {opt_val:.4f}<extra></extra>"
        )
        fig.add_trace(marker, row=1, col=col)
    # ──────────────────────────────────────────────────────────────────────────
 
    fig.update_xaxes(
        title=dict(
            text="K (Blur Strength)",
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickvals=K_vals,
        range=[K_vals[0] - 1.5, K_vals[-1] + 1.5],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )
    fig.update_yaxes(
        title=dict(
            text="fp  (focal plane)",
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickvals=fp_vals,
        range=[fp_vals[0] - 0.025, fp_vals[-1] + 0.025],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )
 
# Style and position the subplot titles (annotations)
# We map the text base directly to the top edge of the plot domain, then offset by exactly 3 pixels.
TITLE_GAP_PX = 3
subplot_top_edge = fig.layout.yaxis.domain[1]  # Shared top edge fraction across subplots

for ann in fig.layout.annotations:
    ann.font = dict(family="Times New Roman", size=6.8, color="#222222")
    ann.y = subplot_top_edge       # Anchors the text's coordinate to the exact top of the subplot
    ann.yanchor = "bottom"         # Forces the bottom of the font text box to sit on that coordinate
    ann.yshift = TITLE_GAP_PX      # Introduces the exact pixel padding upward

fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=45, r=15, t=20, b=20),
    showlegend=False,
)

# Save interactive HTML
out_html = "bokehdiff_stage_1.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")


out = "./bokehdiff_sstage_1.png"
# Requires `kaleido` (pip install -U kaleido) for static image export.
fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
print(f"Saved → {out}")
print(f"Size  : {TEXTWIDTH_IN:.3f} in × {FIG_HEIGHT:.3f} in  @  {DPI} dpi")
print(f"Pixels: {int(TEXTWIDTH_IN*DPI)} × {int(FIG_HEIGHT*DPI)}")

