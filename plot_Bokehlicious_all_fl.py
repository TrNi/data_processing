import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

# ── Define the 5 different FL values to read ──────────────────────────────────
# Modify these numbers to match the exact filenames in your bokehlicious folder
fl_values = [28, 36, 45, 60, 70]


# High-contrast, publication-safe color palette (Paul Tol standard)
# Bundled with unique symbols and dash patterns to maximize distinction
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},    # Clear Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60: {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70: {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

def parse_name(name):
    match = re.search(r"F([\d.]+)", name)
    if not match:
        return None
    val_str = match.group(1)
    if val_str.count('.') > 1:
        parts = val_str.split('.')
        val_str = parts[0] + '.' + parts[1]
    return float(val_str)


# Helper function to convert Hex to RGBA with a custom opacity
def hex_to_rgba(hex_str, opacity=0.3):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"


# ── Container for aggregated data ─────────────────────────────────────────────
fl_data = {}

for fl in fl_values:
    # file_path = f"./bokehlicious/results_bokehlicious_fl_{fl}.csv"
    file_path = f"./bokehlicious_new/results_bokehlicious_p4_fl_{fl}.csv"


    

    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}. Skipping.")
        continue
        
    raw = pd.read_csv(file_path, skiprows=6)
    # data_rows = raw[raw[0].str.startswith("bokehlicious_F", na=False)].copy()
    data_rows = raw[raw["folder_name"].str.startswith("bokehlicious_F", na=False)].copy()

    #parsed = data_rows[0].apply(parse_name)
    #data_rows["F"]     = parsed
    #data_rows["PSNR"]  = data_rows[6].astype(float)
    #data_rows["SSIM"]  = data_rows[7].astype(float)
    #data_rows["LPIPS"] = data_rows[8].astype(float)

    parsed             = data_rows["folder_name"].apply(parse_name)
    data_rows["F"]     = parsed
    data_rows["PSNR"]  = data_rows["PSNR"].astype(float)
    data_rows["SSIM"]  = data_rows["SSIM"].astype(float)
    data_rows["LPIPS"] = data_rows["LPIPS"].astype(float)

    df = data_rows[["F", "PSNR", "SSIM", "LPIPS"]].dropna(subset=["F"]).sort_values(by="F").reset_index(drop=True)
    fl_data[fl] = df

# ── Metric Metadata Configurations ────────────────────────────────────────────
metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]}, 
    "SSIM":  {"label": "SSIM",      "better": "↑", "range": [0.4, 1.0]},
    "LPIPS": {"label": "LPIPS",     "better": "↓", "range": [0.0, 0.8]},
}

# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          
DPI          = 600            
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38 

BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI

# ── Plot Setup ────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

# selected_F_ticks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
selected_F_ticks = [1.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 27.0]
all_F_vals = []
lowest_lpips_points = []  # To track the min LPIPS points for annotations

# Loop over subplots
for i, (m_name, m) in enumerate(metrics.items()):
    col = i + 1
    
    for fl, df in fl_data.items():
        F_vals = df["F"].tolist()
        all_F_vals.extend(F_vals)
        
        show_in_legend = (i == 0)
        style = fl_styles[fl]

        # Convert hex color to an RGBA color with transparency (e.g., 0.3 opacity)
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)

        # Track the lowest LPIPS point for this FL curve
        if m_name == "LPIPS":
            min_idx = df["LPIPS"].idxmin()
            min_row = df.loc[min_idx]
            lowest_lpips_points.append({
                "fl": fl,
                "x": min_row["F"],
                "y": min_row["LPIPS"],
                "color": style["color"]
            })
        
        line_trace = go.Scatter(
            x=F_vals,
            y=df[m_name].tolist(),
            mode="lines+markers",
            name=f"FL {fl}",
            # Applied explicitly contrasting stroke styles
            # line=dict(color=style["color"], width=1.4, dash=style["dash"]),
            line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),

            # Applied explicitly contrasting marker geometry
            marker=dict(
                size=4.0, 
                symbol=style["symbol"], 
                color=style["color"], 
                # line=dict(width=0.4, color="#111111")
                line=dict(width=0.4, color=style["color"])
            ),
            showlegend=show_in_legend,
        )
        fig.add_trace(line_trace, row=1, col=col)

    min_F = min(all_F_vals) if all_F_vals else 1.0
    max_F = max(all_F_vals) if all_F_vals else 14.0

    fig.update_xaxes(
        title=dict(text="F", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickmode="array",
        tickvals=selected_F_ticks,
        range=[min_F - 0.5, max_F + 0.5],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False, row=1, col=col,
    )
    
    fig.update_yaxes(
        title=dict(text=m["label"], font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        range=m["range"],
        showgrid=True, gridcolor="#cccccc", gridwidth=0.4, griddash="dash",
        showline=True, linecolor="#555555", linewidth=0.6,
        mirror=True, ticks="inside", ticklen=2.5, tickwidth=0.5, tickcolor="#333333",
        zeroline=False, row=1, col=col,
    )

# 3.5% Margins
x_domains = [[0.035, 0.3106], [0.3620, 0.6380], [0.6894, 0.9650]]
y_domain  = [0.18, 0.92]  

fig.update_layout(
    xaxis=dict(domain=x_domains[0]), xaxis2=dict(domain=x_domains[1]), xaxis3=dict(domain=x_domains[2]),
    yaxis=dict(domain=y_domain), yaxis2=dict(domain=y_domain), yaxis3=dict(domain=y_domain)
)

x_centers = [sum(domain) / 2 for domain in x_domains]
for idx, ann in enumerate(fig.layout.annotations):
    ann.font = dict(family="Times New Roman", size=6.8, color="#222222")
    ann.x = x_centers[idx]
    ann.y = y_domain[1] + 0.03

# ── ADDING ARROWS & TEXT FOR MINIMUM LPIPS ────────────────────────────────────
annotations = list(fig.layout.annotations) # keep existing subplot title annotations

for pt in lowest_lpips_points:
    if pt["fl"] == 60:
        offset_x, offset_y = 14, -10  # Point up and left
    elif pt["fl"] == 70:
        offset_x, offset_y = 30, -15    # Point down and right
    else:
        offset_x, offset_y = 10, -10   # Default short position
    

    annotations.append(
        go.layout.Annotation(
            x=pt["x"],
            y=pt["y"],
            xref="x3",  # Targets LPIPS X-axis
            yref="y3",  # Targets LPIPS Y-axis
            # text=f"{pt['y']:.3f}",
            # Displays both the target F coordinate value and the raw LPIPS value
            text=f"<b>F={pt['x']:.1f}, {pt['y']:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.8,  # change this.
            arrowwidth=0.6,
            arrowcolor=pt["color"],
            # Offsetting text position slightly above/right to remain clean
            ax=offset_x,
            ay=offset_y,
            font=dict(family="Times New Roman", size=5.0, color=pt["color"]),
            bgcolor="rgba(255, 255, 255, 0.75)",
            bordercolor=pt["color"],
            borderwidth=0.3,
            borderpad=1
        )
    )

fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=0, b=0),
    annotations=annotations, # Inject the new layout annotations safely
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="center",
        x=0.5,
        font=dict(family="Times New Roman", size=6.0, color="#222222"),
    )
)

# Save configurations
out_html = f"./bokehlicious_multi_fl_plots.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out = f"./bokehlicious_multi_fl_plots.png"
try:
    fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved → {out}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open the HTML output to view.")