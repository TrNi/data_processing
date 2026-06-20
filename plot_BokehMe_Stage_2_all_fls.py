"""
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38  

BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI

# ── High-Contrast, Publication-Safe Palette (Tol Standard) ───────────────────
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},     # dash Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60:  {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70:  {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

def hex_to_rgba(hex_str, opacity=0.5):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# ── Data Parsing Functions ───────────────────────────────────────────────────
def parse_bokehme_folder(folder_name):
    if not isinstance(folder_name, str):
        return None
    dfs_match = re.search(r"dfs(\d+)", folder_name)
    k_match = re.search(r"K(\d+)", folder_name)
    df_match = re.search(r"dispfocus([\d.]+)", folder_name)
    
    if dfs_match and k_match and df_match:
        return {
            "dfs": int(dfs_match.group(1)),
            "K": int(k_match.group(1)),
            "dispfocus": float(df_match.group(1))
        }
    return None

def load_and_process_data(fl_values):
    fl_data = {}
    
    for fl in fl_values:
        file_path = f"./bokehme/results_bokehme_fl_{fl}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}. Skipping.")
            continue
            
        # ── UNTOUCHED: Original Parsing Engine for Standard FL Files ─────────
        if fl != 28:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            header_idx = -1
            for idx, line in enumerate(lines):
                if "PSNR" in line:
                    header_idx = idx
                    break
                    
            if header_idx == -1:
                continue
                
            df = pd.read_csv(file_path, skiprows=header_idx)
            df = df.dropna(subset=[df.columns[0]])
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            
            folder_col = [c for c in df.columns if "folder" in c or c == "Unnamed: 0" or df[c].astype(str).str.contains("bokehme").any()][0]
            df = df[df[folder_col].str.contains("bokehme_dfs", na=False)].copy()
            
            parsed = df[folder_col].apply(parse_bokehme_folder)
            df = df[parsed.notna()].copy()
            
            df["dfs"] = parsed.dropna().apply(lambda x: x["dfs"])
            df["K"] = parsed.dropna().apply(lambda x: x["K"])
            df["dispfocus"] = parsed.dropna().apply(lambda x: x["dispfocus"])
            
            for m in ["PSNR", "SSIM", "LPIPS"]:
                df[m] = pd.to_numeric(df[m], errors='coerce')
                
            # Filter standard sweep files
            df = df[(df["dfs"] == 20) & (np.isclose(df["dispfocus"], 0.15))].copy()

        # ── SPECIAL HANDLING: Robust Interleaved Reader for Multi-Part FL 28 ─
        else:
            raw_df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
            valid_rows = []
            
            for idx, row in raw_df.iterrows():
                row_str = str(row.iloc[0])
                if "bokehme_dfs" in row_str:
                    parsed = parse_bokehme_folder(row_str)
                    if parsed and parsed["dfs"] == 20 and np.isclose(parsed["dispfocus"], 0.15):
                        metrics_line = row.dropna().tolist()
                        if len(metrics_line) >= 4:
                            try:
                                valid_rows.append({
                                    "K": parsed["K"],
                                    "PSNR": float(metrics_line[-3]),
                                    "SSIM": float(metrics_line[-2]),
                                    "LPIPS": float(metrics_line[-1])
                                })
                            except ValueError:
                                pass
                                
            df = pd.DataFrame(valid_rows)
        
        df = df.sort_values(by=["K"]).reset_index(drop=True)
        fl_data[fl] = df
        print(f"Loaded FL {fl}: found {len(df)} configurations matching dfs=20, dispfocus=0.15")
        
    return fl_data

# ── Plot Generation (Restored Styles) ─────────────────────────────────────────
fl_values = [28, 36, 45, 60, 70]
fl_data = load_and_process_data(fl_values)

metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]}, 
    "SSIM":  {"label": "SSIM",      "better": "↑", "range": [0.4, 1.00]},
    "LPIPS": {"label": "LPIPS",     "better": "↓", "range": [0.0, 0.6]},
}


fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

lowest_lpips_points = []

for i, (m_name, m) in enumerate(metrics.items()):
    col = i + 1
    
    for fl, df in fl_data.items():
        if df.empty:
            continue
            
        x_vals = [int(r['K']) for _, r in df.iterrows()]
        
        show_in_legend = (i == 0)
        style = fl_styles[fl]
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)
        
        if m_name == "LPIPS":
            min_idx = df["LPIPS"].idxmin()
            if pd.notna(min_idx):
                min_row = df.loc[min_idx]
                lowest_lpips_points.append({
                    "fl": fl,
                    "x_str": f"K{int(min_row['K'])}",
                    "y": min_row["LPIPS"],
                    "color": style["color"]
                })
            
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[m_name].tolist(),
                mode="lines+markers",
                name=f"FL {fl} mm",
                line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),
                marker=dict(
                    size=4.5, 
                    symbol=style["symbol"], 
                    color=style["color"], 
                    line=dict(width=0.4, color=style["color"])
                ),
                showlegend=show_in_legend,
            ),
            row=1, col=col
        )

    fig.update_xaxes(
        title=dict(text="K", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
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

# ── Geometry Margins & Layout Normalization (Restored Styles) ──────────────────
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

# ── Annotation Positioning ────────────────────────────────────────────────────
annotations = list(fig.layout.annotations)

for pt in lowest_lpips_points:
    if pt["fl"] == 28:    offset_x, offset_y = -30, -35
    elif pt["fl"] == 36:  offset_x, offset_y = 25, -25
    elif pt["fl"] == 45:  offset_x, offset_y = 20, 20
    elif pt["fl"] == 60:  offset_x, offset_y = -25, 25
    else:                 offset_x, offset_y = 30, -10

    annotations.append(
        go.layout.Annotation(
            x=pt["x_str"], y=pt["y"], xref="x3", yref="y3",
            text=f"<b> {pt["x_str"]}, {pt['y']:.3f} </b>",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.0, arrowcolor=pt["color"],
            ax=offset_x, ay=offset_y,
            font=dict(family="Times New Roman", size=4.8, color=pt["color"]),
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor=pt["color"], borderwidth=0.3, borderpad=1
        )
    )

fig.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX, height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=0, b=0),
    annotations=annotations,
    legend=dict(
        orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5,
        font=dict(family="Times New Roman", size=5.8, color="#222222"),
    )
)

# ── Save Pipelines ────────────────────────────────────────────────────────────
fig.write_html("./results_bokehme_stage_2_all_fls.html")
try:
    fig.write_image("./results_bokehme_stage_2_all_fls.png", width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print("Plot generated successfully.")
except Exception:
    print("Saved HTML layout trace successfully.")
"""

import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38  

BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI

# ── High-Contrast, Publication-Safe Palette (Tol Standard) ───────────────────
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},     # dash Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60:  {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70:  {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

def hex_to_rgba(hex_str, opacity=0.5):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# ── Data Parsing Functions ───────────────────────────────────────────────────
def parse_bokehme_folder(folder_name):
    if not isinstance(folder_name, str):
        return None
    dfs_match = re.search(r"dfs(\d+)", folder_name)
    k_match = re.search(r"K(\d+)", folder_name)
    df_match = re.search(r"dispfocus([\d.]+)", folder_name)
    
    if dfs_match and k_match and df_match:
        return {
            "dfs": int(dfs_match.group(1)),
            "K": int(k_match.group(1)),
            "dispfocus": float(df_match.group(1))
        }
    return None

def load_and_process_data(fl_values):
    fl_data = {}
    
    for fl in fl_values:
        file_path = f"./bokehme/results_bokehme_fl_{fl}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}. Skipping.")
            continue
            
        # ── UNTOUCHED: Original Parsing Engine for Standard FL Files ─────────
        if fl != 28:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            header_idx = -1
            for idx, line in enumerate(lines):
                if "PSNR" in line:
                    header_idx = idx
                    break
                    
            if header_idx == -1:
                continue
                
            df = pd.read_csv(file_path, skiprows=header_idx)
            df = df.dropna(subset=[df.columns[0]])
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            
            folder_col = [c for c in df.columns if "folder" in c or c == "Unnamed: 0" or df[c].astype(str).str.contains("bokehme").any()][0]
            df = df[df[folder_col].str.contains("bokehme_dfs", na=False)].copy()
            
            parsed = df[folder_col].apply(parse_bokehme_folder)
            df = df[parsed.notna()].copy()
            
            df["dfs"] = parsed.dropna().apply(lambda x: x["dfs"])
            df["K"] = parsed.dropna().apply(lambda x: x["K"])
            df["dispfocus"] = parsed.dropna().apply(lambda x: x["dispfocus"])
            
            for m in ["PSNR", "SSIM", "LPIPS"]:
                df[m] = pd.to_numeric(df[m], errors='coerce')
                
            # Filter standard sweep files
            df = df[(df["dfs"] == 20) & (np.isclose(df["dispfocus"], 0.15))].copy()

        # ── SPECIAL HANDLING: Robust Interleaved Reader for Multi-Part FL 28 ─
        else:
            raw_df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
            valid_rows = []
            
            for idx, row in raw_df.iterrows():
                row_str = str(row.iloc[0])
                if "bokehme_dfs" in row_str:
                    parsed = parse_bokehme_folder(row_str)
                    if parsed and parsed["dfs"] == 20 and np.isclose(parsed["dispfocus"], 0.15):
                        metrics_line = row.dropna().tolist()
                        if len(metrics_line) >= 4:
                            try:
                                valid_rows.append({
                                    "K": parsed["K"],
                                    "PSNR": float(metrics_line[-3]),
                                    "SSIM": float(metrics_line[-2]),
                                    "LPIPS": float(metrics_line[-1])
                                })
                            except ValueError:
                                pass
                                
            df = pd.DataFrame(valid_rows)
        
        df = df.sort_values(by=["K"]).reset_index(drop=True)
        fl_data[fl] = df
        print(f"Loaded FL {fl}: found {len(df)} configurations matching dfs=20, dispfocus=0.15")
        
    return fl_data

# ── Plot Generation (Restored Styles) ─────────────────────────────────────────
fl_values = [28, 36, 45, 60, 70]
fl_data = load_and_process_data(fl_values)

metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]}, 
    "SSIM":  {"label": "SSIM",      "better": "↑", "range": [0.4, 1.00]},
    "LPIPS": {"label": "LPIPS",     "better": "↓", "range": [0.0, 0.6]},
}

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

lowest_lpips_points = []

for i, (m_name, m) in enumerate(metrics.items()):
    col = i + 1
    
    for fl, df in fl_data.items():
        if df.empty:
            continue
            
        x_vals = [int(r['K']) for _, r in df.iterrows()]
        
        show_in_legend = (i == 0)
        style = fl_styles[fl]
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)
        
        # Track minimum LPIPS points (and cache them for optimal annotations/markers)
        if m_name == "LPIPS":
            min_idx = df["LPIPS"].idxmin()
            if pd.notna(min_idx):
                min_row = df.loc[min_idx]
                lowest_lpips_points.append({
                    "fl": fl,
                    "x_val": int(min_row['K']),  # Fixed: Keep as integer to match axis type
                    "y": min_row["LPIPS"],
                    "color": style["color"],
                    "symbol": style["symbol"]
                })
            
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[m_name].tolist(),
                mode="lines+markers",
                name=f"FL {fl} mm",
                line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),
                marker=dict(
                    size=4.5, 
                    symbol=style["symbol"], 
                    color=style["color"], 
                    line=dict(width=0.4, color=style["color"])
                ),
                showlegend=show_in_legend,
            ),
            row=1, col=col
        )

    fig.update_xaxes(
        title=dict(text="K (Blur Strength)", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
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

# ── 1. Draw High-Contrast Optimal Markers on LPIPS Subplot (Col 3) ────────────
for pt in lowest_lpips_points:
    fig.add_trace(
        go.Scatter(
            x=[pt["x_val"]],
            y=[pt["y"]],
            mode="markers",
            marker=dict(
                size=8.5,            # Visibly larger than base markers
                symbol=pt["symbol"], # Preserves corresponding style shape
                color=pt["color"],   # 100% solid opacity
                line=dict(width=1.0, color="#ffffff") # Crisp boundary wrap
            ),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=3
    )

# ── Geometry Margins & Layout Normalization (Restored Styles) ──────────────────
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

# ── 2. Annotation Positioning with Clean Axis Metrics ───────────────────────
annotations = list(fig.layout.annotations)

for pt in lowest_lpips_points:
    if pt["fl"] == 28:    offset_x, offset_y = -30, 30 #-30, 15
    elif pt["fl"] == 36:  offset_x, offset_y = -30, 15 #-30, 40
    elif pt["fl"] == 45:  offset_x, offset_y = -30, -10
    elif pt["fl"] == 60:  offset_x, offset_y = -30, 10
    else:                 offset_x, offset_y = -30, 0

    annotations.append(
        go.layout.Annotation(
            x=pt["x_val"], y=pt["y"], xref="x3", yref="y3", # Matches integer axis
            text=f"<b>K={float(pt['x_val'])}, {pt['y']:.3f}</b>",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.0, arrowcolor=pt["color"],
            ax=offset_x, ay=offset_y,
            font=dict(family="Times New Roman", size=4.8, color=pt["color"]),
            bgcolor="rgba(255, 255, 255, 0.90)",
            bordercolor=pt["color"], borderwidth=0.3, borderpad=1
        )
    )

fig.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX, height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=0, b=0),
    annotations=annotations,
    legend=dict(
        orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5,
        font=dict(family="Times New Roman", size=5.8, color="#222222"),
    )
)

# ── Save Pipelines ────────────────────────────────────────────────────────────
fig.write_html("./results_bokehme_stage_2_all_fls.html")
try:
    fig.write_image("./results_bokehme_stage_2_all_fls.png", width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print("Plot generated successfully with sub-plot markers.")
except Exception as e:
    print(f"Saved HTML layout successfully. Static image generation bypassed: {e}")
