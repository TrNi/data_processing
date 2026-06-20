import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os

# ── Define the 5 different FL values to read ──────────────────────────────────
fl_values = [28, 36, 45, 60, 70]

# High-contrast, publication-safe color palette (Paul Tol standard) from reference script
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},    # Clear Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60:  {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70:  {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

def parse_k_val(folder_name):
    """Extracts the K value from folder names like drbokeh_K15_fp0.3_ls71"""
    match = re.search(r"K(\d+)", str(folder_name))
    if match:
        return int(match.group(1))
    return None

def hex_to_rgba(hex_str, opacity=0.3):
    """Convert Hex to RGBA with a custom opacity for smooth layout lines"""
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# ── Step 1: Load, Clean, and Aggregate Data ──────────────────────────────────
fl_data = {}
all_K_vals = []

for fl in fl_values:
    file_path = f"./Drbokeh/results_drbokeh_fl{fl}.csv"
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}. Skipping.")
        continue
        
    # Skip any potential metadata header lines to locate the true data matrix table
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    start_idx = 0
    for idx, line in enumerate(lines):
        if "folder_name" in line and "PSNR" in line:
            start_idx = idx
            break
            
    df_raw = pd.read_csv(file_path, skiprows=start_idx)
    df_raw = df_raw.dropna(subset=["folder_name", "PSNR", "SSIM", "LPIPS"])
    
    # Process X-axis mapping coordinates (K values)
    df_raw["K_val"] = df_raw["folder_name"].apply(parse_k_val)
    df_raw = df_raw.dropna(subset=["K_val"])
    
    # Apply strict evaluation filters defined by data constraints
    if fl == 28:
        # 1. Only consider fp=0.3
        df_raw = df_raw[df_raw['folder_name'].str.contains('fp0.3', na=False)]
        # 2. Only consider K=15, 25, and 35
        df_raw = df_raw[df_raw['K_val'].isin([15, 25, 35])]
        
    # Clean and sort matrix sequence natively
    df_clean = df_raw[["K_val", "PSNR", "SSIM", "LPIPS"]].sort_values(by="K_val").reset_index(drop=True)
    
    if not df_clean.empty:
        fl_data[fl] = df_clean
        all_K_vals.extend(df_clean["K_val"].tolist())

# ── Step 2: Metric Metadata Configurations ────────────────────────────────────
metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]}, 
    "SSIM":  {"label": "SSIM", "better": "↑", "range": [0.4, 1.0]},
    "LPIPS": {"label": "LPIPS", "better": "↓", "range": [0.0, 0.6]},
}

# ── Step 3: ICCP / IEEE PAMI Page Geometry Formatting ─────────────────────────
TEXTWIDTH_IN = 6.875          
DPI          = 600            
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38 

BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI

# ── Step 4: Plot Setup & Initialization ───────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

selected_K_ticks = [15, 25, 35]
lowest_lpips_points = [] 

# Loop over the metrics to fill subplots
for i, (m_name, m) in enumerate(metrics.items()):
    col = i + 1
    
    for fl, df in fl_data.items():
        K_vals = df["K_val"].tolist()
        show_in_legend = (i == 0)
        style = fl_styles[fl]

        # Convert hex color to an RGBA color with transparency matching reference style
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)

        # Track minimal LPIPS points across trends for direct annotation calls
        if m_name == "LPIPS":
            min_idx = df["LPIPS"].idxmin()
            min_row = df.loc[min_idx]
            lowest_lpips_points.append({
                "fl": fl,
                "x": min_row["K_val"],
                "y": min_row["LPIPS"],
                "color": style["color"]
            })
        
        line_trace = go.Scatter(
            x=K_vals,
            y=df[m_name].tolist(),
            mode="lines+markers",
            name=f"FL {fl}mm",
            line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),
            marker=dict(
                size=4.0, 
                symbol=style["symbol"], 
                color=style["color"], 
                line=dict(width=0.4, color=style["color"])
            ),
            showlegend=show_in_legend,
        )
        fig.add_trace(line_trace, row=1, col=col)

    min_K = min(all_K_vals) if all_K_vals else 15
    max_K = max(all_K_vals) if all_K_vals else 35

    fig.update_xaxes(
        title=dict(text="K (Blur Strength)", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickmode="array",
        tickvals=selected_K_ticks,
        range=[min_K - 2.0, max_K + 2.0],
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

# Strict 3.5% padding margin geometric configuration from script source
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

# ── Step 5: Arrow Annotations targeting Optimal Performance ───────────────────
annotations = list(fig.layout.annotations) 

for pt in lowest_lpips_points:
    # Fine-tuned offsetting adjustments to maintain uniform visual breathing room
    if pt["fl"] == 60:
        offset_x, offset_y = 25, -12  
    elif pt["fl"] == 70:
        offset_x, offset_y = 25, -8    
    elif pt["fl"] == 36:
        offset_x, offset_y = 25, 10
    elif pt["fl"] == 28:
        offset_x, offset_y = 25, 5
    else:
        offset_x, offset_y = 25, -10   

    annotations.append(
        go.layout.Annotation(
            x=pt["x"],
            y=pt["y"],
            xref="x3",  # Dynamic anchoring tracking targeting LPIPS chart
            yref="y3",  
            text=f"<b>K={pt['x']}, {pt['y']:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.8,  
            arrowwidth=0.6,
            arrowcolor=pt["color"],
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
    annotations=annotations, 
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="center",
        x=0.5,
        font=dict(family="Times New Roman", size=6.0, color="#222222"),
    )
)

# ── Step 6: File Export Pipelines ─────────────────────────────────────────────
out_html = "./drbokeh_stage_2.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out_png = "./drbokeh_stage_2.png"
try:
    fig.write_image(out_png, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved Publication-Ready Figure → {out_png}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open HTML output file to view.")