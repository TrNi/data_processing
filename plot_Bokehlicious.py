import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
 
# ── Style to match reference figure ──────────────────────────────────────────
# Font sizes are scaled from the original 15-in figure to the ICCP page
# textwidth of 6.875 in using a sqrt-damped factor (√(6.875/15) ≈ 0.677)
# so text stays legible at 600 dpi while respecting physical proportions.

fl_val = 70
# ── Load & parse ──────────────────────────────────────────────────────────────
# raw = pd.read_csv(f"./bokehlicious_new/results_bokehlicious_p4_fl_{fl_val}.csv", header=None)
raw = pd.read_csv(f"./bokehlicious_new/results_bokehlicious_p4_fl_{fl_val}.csv", skiprows=6)


# Filter rows matching the bokehlicious file format
# data_rows = raw[raw[0].str.startswith("bokehlicious_F", na=False)].copy()
data_rows = raw[raw["folder_name"].str.startswith("bokehlicious_F", na=False)].copy()


def parse_name(name):
    # Extracts the F-number value, handling trailing notation anomalies (e.g., F12.0.0 -> 12.0)
    match = re.search(r"F([\d.]+)", name)
    val_str = match.group(1)
    if val_str.count('.') > 1:
        parts = val_str.split('.')
        val_str = parts[0] + '.' + parts[1]
    return float(val_str)

parsed             = data_rows["folder_name"].apply(parse_name)
data_rows["F"]     = parsed
data_rows["PSNR"]  = data_rows["PSNR"].astype(float)
data_rows["SSIM"]  = data_rows["SSIM"].astype(float)
data_rows["LPIPS"] = data_rows["LPIPS"].astype(float)



# Sort values by the independent variable F to ensure continuous line traces
df = data_rows[["F", "PSNR", "SSIM", "LPIPS"]].sort_values(by="F").reset_index(drop=True)
 
# ── 1D Independent Variable Axis ─────────────────────────────────────────────
F_vals = df["F"].tolist()
 
metrics = {
    "PSNR":  {"values": df["PSNR"].tolist(),  "color": "#117733", "label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]},
    "SSIM":  {"values": df["SSIM"].tolist(),  "color": "#44AA99", "label": "SSIM",      "better": "↑", "range": [0.5, 1.0]},
    "LPIPS": {"values": df["LPIPS"].tolist(), "color": "#882255", "label": "LPIPS",     "better": "↓", "range": [0.0, 0.7]},
}
 
# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.35
 
BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,  # Adjusted to leave enough space for independent y-axis titles
)

# Change these numbers to the exact F values you want to see displayed.
# selected_F_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
selected_F_ticks = [1.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 27.0]


for i, (name, m) in enumerate(metrics.items()):
    col = i + 1
 
    # Create the 1D trend line trace
    line_trace = go.Scatter(
        x=F_vals,
        y=m["values"],
        mode="lines+markers",
        # ── Added dash="dash" to format connection lines ──
        line=dict(color=m["color"], width=1.2, dash="dash"),
        marker=dict(size=4, symbol="circle", color=m["color"], line=dict(width=0.5, color="#222222")),
        showlegend=False,
    )
    fig.add_trace(line_trace, row=1, col=col)
 
    fig.update_xaxes(
        title=dict(
            text="F",
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        #tickvals=F_vals,
        # nticks=10,
        tickmode="array",
        tickvals=selected_F_ticks,
        range=[F_vals[0] - 0.5, F_vals[-1] + 0.5],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )
    
    fig.update_yaxes(
        title=dict(
            text=m["label"],
            font=dict(family="Times New Roman", size=6.1, color="#222222"),
            standoff=3,
        ),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        range=m["range"],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False,
        row=1, col=col,
    )


# Define precise domains to shrink the plots while leaving internal gaps at exactly 0.05
#x_domains = [[0.10, 0.32], [0.37, 0.59], [0.64, 0.86]]
#y_domain  = [0.15, 0.85]
# Left subplot now starts at 0.04 (closer to left edge), right ends at 0.96 (closer to right edge).
#x_domains = [[0.01, 0.29], [0.34, 0.62], [0.67, 0.99]]
#y_domain  = [0.07, 0.92] # Maximum expansion vertically to eliminate padding voids

#x_domains = [[0.035, 0.3033], [0.3533, 0.6216], [0.6716, 0.9399]]
#y_domain  = [0.07, 0.92]
# 3.5% MARGINS ON BOTH EDGES
x_domains = [[0.035, 0.3106], [0.3620, 0.6380], [0.6894, 0.9650]]
y_domain  = [0.07, 0.92]



fig.update_layout(
    xaxis=dict(domain=x_domains[0]),
    xaxis2=dict(domain=x_domains[1]),
    xaxis3=dict(domain=x_domains[2]),
    yaxis=dict(domain=y_domain),
    yaxis2=dict(domain=y_domain),
    yaxis3=dict(domain=y_domain)
)

# Calculate the exact center point for each domain array to properly align the text
x_centers = [sum(domain) / 2 for domain in x_domains]


# Style the subplot titles (annotations created by make_subplots)
for idx, ann in enumerate(fig.layout.annotations):
    ann.font = dict(family="Times New Roman", size=6.8, color="#222222")
    # Shift title horizontally to perfectly center above its shrunken subplot frame
    ann.x = x_centers[idx]
    # Drop the title down vertically so it sits cleanly right above the shrunken Y domain frame
    ann.y = y_domain[1] + 0.03


fig.update_layout(
    #title=dict(
    #    text="<i>Bokehlicious — Hyperparameter Grid Search</i>",
    #    font=dict(family="Times New Roman", size=7.4, color="#222222"),
    #    x=0.5, xanchor="center", y=0.985, yanchor="top",
    #),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    # margin=dict(l=45, r=20, t=50, b=40),
    #margin=dict(l=25, r=10, t=20, b=20),
    #margin=dict(l=10, r=10, t=12, b=12),
    margin=dict(l=0, r=0, t=0, b=0),
)

# Save interactive HTML
out_html = f"./bokehlicious_plots_fl_{fl_val}.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out = f"./bokehlicious_plots_fl_{fl_val}.png"
# Requires `kaleido` (pip install -U kaleido) for static image export.
try:
    fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved → {out}")
    print(f"Size  : {TEXTWIDTH_IN:.3f} in × {FIG_HEIGHT:.3f} in  @  {DPI} dpi")
    print(f"Pixels: {int(TEXTWIDTH_IN*DPI)} × {int(FIG_HEIGHT*DPI)}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open the HTML output to view.")