'''
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
 
# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38  
 
BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI
 
# ── Configuration & Data Files ────────────────────────────────────────────────
fl_vals = [28, 36, 45, 60, 70]

# RESTORED STYLE: High-contrast Paul Tol palette with unique shapes/dashes from your first script
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},    # Clear Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60:  {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70:  {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

# Extracts a specific value based on its prefix key identifier ('K' or 'fp_')
def extract_param(name_str, prefix_key):
    name_str = str(name_str)
    if prefix_key == "fp_":
        match = re.search(r"fp_([\d.]+)", name_str)
    else:
        match = re.search(r"K_?([\d.]+)", name_str)
        
    if match:
        val_str = match.group(1)
        if val_str.count('.') > 1:
            parts = val_str.split('.')
            val_str = parts[0] + '.' + parts[1]
        return float(val_str)
    return None

# RESTORED STYLE: Helper function to convert Hex to RGBA with custom opacity
def hex_to_rgba(hex_str, opacity=0.3):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# ── Load & Parse Data ─────────────────────────────────────────────────────────
# CONTENT UNCHANGED: All bokehdiff content logic preserved exactly
dfs = {}
all_unique_ks = set()
optimal_lpips_points = []  # Tracks lowest LPIPS points across curves


for fl_val in fl_vals:
    file_path = f"./bokehdiff/results_bokehdiff_fl_{fl_val}.csv"
    if not os.path.exists(file_path):
        file_path = f"results_bokehdiff_fl_{fl_val}.csv"
        
    try:
        raw = pd.read_csv(file_path, skiprows=6, header=None)
        
        raw = raw.dropna(how='all', axis=1)
        raw = raw.dropna(how='all', axis=0)

        if raw.shape[1] >= 4:
            raw.columns = ["folder_name"] + [f"extra_{idx}" for idx in range(raw.shape[1] - 4)] + ["PSNR", "SSIM", "LPIPS"]
        else:
            print(f"Warning: Unexpected layout columns in fl_{fl_val}. Skipping.")
            continue
            
        data_rows = raw[raw["folder_name"].str.contains("K|bokeh", na=False, case=False)].copy()
        if data_rows.empty:
            data_rows = raw.copy()  
        
        data_rows["PSNR"]  = data_rows["PSNR"].astype(float)
        data_rows["SSIM"]  = data_rows["SSIM"].astype(float)
        data_rows["LPIPS"] = data_rows["LPIPS"].astype(float)
        
        if fl_val == 28:
            data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
            data_rows["fp"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "fp_"))
            data_rows = data_rows[np.isclose(data_rows["fp"], 0.2)]
        else:
            data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
            
        data_rows = data_rows.dropna(subset=["K"])

        # ── REFINEMENT: Only keep K values equal to 15, 20, 25, 40 ───────────
        data_rows = data_rows[data_rows["K"].isin([15.0, 20.0, 25.0, 40.0])]

        all_unique_ks.update(data_rows["K"].tolist())
        
        df = data_rows[["K", "PSNR", "SSIM", "LPIPS"]].sort_values(by="K").reset_index(drop=True)
        if not df.empty:
            dfs[fl_val] = df
            
            min_idx = df["LPIPS"].idxmin()
            min_row = df.loc[min_idx]
            optimal_lpips_points.append({
                "fl": fl_val,
                "x": min_row["K"],
                "y": min_row["LPIPS"],
                "color": fl_styles[fl_val]["color"]
            })
            print(f"Successfully loaded fl_{fl_val}: parsed {len(df)} entries. Min LPIPS at K={min_row['K']}.")
        else:
            print(f"Warning: No matching parameters left for fl_{fl_val}.")
            
    except Exception as e:
        print(f"Could not load or parse file for fl_{fl_val}: {e}")

metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]},
    "SSIM":  {"label": "SSIM",      "better": "↑", "range": [0.5, 1.0]},
    "LPIPS": {"label": "LPIPS",     "better": "↓", "range": [0.0, 0.7]},
}
 
# ── Plot Generation ───────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

# selected_K_ticks = [10, 15, 20, 25, 40, 45] # sorted(list(all_unique_ks))
# ── REFINEMENT: Set tick marks strictly to the selected targets ──────────────
selected_K_ticks = sorted(list(all_unique_ks)) if all_unique_ks else [15, 20, 25, 40]

#if not selected_K_ticks:
#    selected_K_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]  

for i, (name, m) in enumerate(metrics.items()):
    col = i + 1
 
    for fl_val in fl_vals:
        if fl_val not in dfs or dfs[fl_val].empty:
            continue
        df = dfs[fl_val]
        
        # RESTORED STYLE: Legend strictly maps to the first subplot column evaluation
        show_legend = (i == 0)
        style = fl_styles[fl_val]
        
        # RESTORED STYLE: Re-introduced hex transparency configuration opacity value (0.5)
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)
        
        line_trace = go.Scatter(
            x=df["K"].tolist(),
            y=df[name].tolist(),
            mode="lines+markers",
            name=f"fl = {fl_val} mm" if fl_val == 28 else f"fl = {fl_val} mm",
            # RESTORED STYLE: Applied transparent stroke lines and corresponding dash arrays
            line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),
            # RESTORED STYLE: Reverted to unique marker geometry rules with explicit inline color tracking
            marker=dict(
                size=4.0, 
                symbol=style["symbol"], 
                color=style["color"], 
                line=dict(width=0.4, color=style["color"])
            ),
            showlegend=show_legend,
        )
        fig.add_trace(line_trace, row=1, col=col)
 
    fig.update_xaxes(
        title=dict(text="K", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickmode="array",
        tickvals=selected_K_ticks,
        # range=[min(selected_K_ticks) - 0.05, max(selected_K_ticks) + 0.05],
        # ── REFINEMENT: Set tick marks strictly to the selected targets ──────────────
        range=[min(selected_K_ticks) - 2.0, max(selected_K_ticks) + 2.0],
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

# RESTORED STYLE: 3.5% Margins and original layout domains configuration
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
annotations = list(fig.layout.annotations)

for pt in optimal_lpips_points:
    # RESTORED STYLE: Arrow offset selection logic from original script
    if pt["fl"] == 60:
        offset_x, offset_y = 14, -10  # Point up and left
    elif pt["fl"] == 28:
        offset_x, offset_y = 60, 20
    elif pt["fl"] == 36:
        offset_x, offset_y = 10, 20
    else:
        offset_x, offset_y = 10, -10  # Default short position
        
    annotations.append(
        go.layout.Annotation(
            x=pt["x"],
            y=pt["y"],
            xref="x3",  
            yref="y3",  
            # RESTORED STYLE: Compact text format string tracking K targets
            text=f"<b>K={pt['x']:.1f}, {pt['y']:.3f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,  
            arrowwidth=1.0,
            arrowcolor=pt["color"],
            ax=offset_x,
            ay=offset_y,
            font=dict(family="Times New Roman", size=5.0, color=pt["color"]),
            bgcolor="rgba(255, 255, 255, 0.75)", # Restored 0.75 opacity background
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
        x=0.5,                  
        y=0.01,                 
        xanchor="center",
        yanchor="bottom",
        font=dict(family="Times New Roman", size=6.0, color="#222222"),
    )
)

out_html = "./bokehdiff_stage_2_all_fls.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out = "./bokehdiff_stage_2_all_fls.png"
try:
    fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved → {out}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open the HTML output to view.")
'''



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
 
# ── ICCP / IEEE PAMI page geometry ───────────────────────────────────────────
TEXTWIDTH_IN = 6.875          # full page textwidth
DPI          = 600            # PAMI recommended resolution
FIG_HEIGHT   = TEXTWIDTH_IN * 0.38  
 
BASE_PPI  = 100
WIDTH_PX  = int(TEXTWIDTH_IN * BASE_PPI)
HEIGHT_PX = int(FIG_HEIGHT   * BASE_PPI)
SCALE     = DPI / BASE_PPI
 
# ── Configuration & Data Files ────────────────────────────────────────────────
fl_vals = [28, 36, 45, 60, 70]

# High-contrast Paul Tol palette with unique shapes/dashes
fl_styles = {
    28:  {"color": "#4477AA", "symbol": "circle",          "dash": "solid"},    # Clear Blue
    36:  {"color": "#EE6677", "symbol": "square",          "dash": "dash"},     # Vibrant Rose/Red
    45:  {"color": "#228833", "symbol": "diamond",         "dash": "dot"},      # Deep Green
    60:  {"color": "#CCBB44", "symbol": "triangle-up",     "dash": "dashdot"},  # Muted Yellow
    70:  {"color": "#AA3377", "symbol": "cross",           "dash": "longdash"}, # Dark Purple/Magenta
}

# Extracts a specific value based on its prefix key identifier ('K' or 'fp_')
def extract_param(name_str, prefix_key):
    name_str = str(name_str)
    if prefix_key == "fp_":
        match = re.search(r"fp_([\d.]+)", name_str)
    else:
        match = re.search(r"K_?([\d.]+)", name_str)
        
    if match:
        val_str = match.group(1)
        if val_str.count('.') > 1:
            parts = val_str.split('.')
            val_str = parts[0] + '.' + parts[1]
        return float(val_str)
    return None

# Helper function to convert Hex to RGBA with custom opacity
def hex_to_rgba(hex_str, opacity=0.3):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"

# ── Load & Parse Data ─────────────────────────────────────────────────────────
dfs = {}
optimal_lpips_points = []  # Tracks lowest LPIPS points across curves

# Fixed target ordered mapping for the category axis sequence
target_ks = [15, 20, 25, 40]
str_target_ks = [str(k) for k in target_ks]

for fl_val in fl_vals:
    file_path = f"./bokehdiff/results_bokehdiff_fl_{fl_val}.csv"
    if not os.path.exists(file_path):
        file_path = f"results_bokehdiff_fl_{fl_val}.csv"
        
    try:
        raw = pd.read_csv(file_path, skiprows=6, header=None)
        
        raw = raw.dropna(how='all', axis=1)
        raw = raw.dropna(how='all', axis=0)

        if raw.shape[1] >= 4:
            raw.columns = ["folder_name"] + [f"extra_{idx}" for idx in range(raw.shape[1] - 4)] + ["PSNR", "SSIM", "LPIPS"]
        else:
            print(f"Warning: Unexpected layout columns in fl_{fl_val}. Skipping.")
            continue
            
        data_rows = raw[raw["folder_name"].str.contains("K|bokeh", na=False, case=False)].copy()
        if data_rows.empty:
            data_rows = raw.copy()  
        
        data_rows["PSNR"]  = data_rows["PSNR"].astype(float)
        data_rows["SSIM"]  = data_rows["SSIM"].astype(float)
        data_rows["LPIPS"] =  round(data_rows["LPIPS"].astype(float), 2)

        if fl_val == 28:
            data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
            data_rows["fp"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "fp_"))
            data_rows = data_rows[np.isclose(data_rows["fp"], 0.2)]
        else:
            data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
            
        data_rows = data_rows.dropna(subset=["K"])
        
        # Filter strictly down to targets
        data_rows = data_rows[data_rows["K"].isin([float(k) for k in target_ks])]
        
        # Convert K to int then string to force identical category strings across subsets
        data_rows["K_str"] = data_rows["K"].astype(int).astype(str)
        
        df = data_rows[["K", "K_str", "PSNR", "SSIM", "LPIPS"]].sort_values(by="K").reset_index(drop=True)
        if not df.empty:
            dfs[fl_val] = df
            
            min_idx = df["LPIPS"].idxmin()
            min_row = df.loc[min_idx]
            
            # Find integer positional index relative to our categorical sequence map
            k_val_int = int(min_row["K"])
            cat_index = target_ks.index(k_val_int)
            
            optimal_lpips_points.append({
                "fl": fl_val,
                "x_idx": cat_index,    # Store categorical position integer (0, 1, 2, or 3)
                "y": min_row["LPIPS"],
                "raw_k": min_row["K"],
                "color": fl_styles[fl_val]["color"]
            })
            print(f"Successfully loaded fl_{fl_val}: parsed {len(df)} entries. Min LPIPS at K={min_row['K']}.")
        else:
            print(f"Warning: No matching parameters left for fl_{fl_val}.")
            
    except Exception as e:
        print(f"Could not load or parse file for fl_{fl_val}: {e}")

metrics = {
    "PSNR":  {"label": "PSNR (dB)", "better": "↑", "range": [10.0, 35.0]},
    "SSIM":  {"label": "SSIM",      "better": "↑", "range": [0.5, 1.0]},
    "LPIPS": {"label": "LPIPS",     "better": "↓", "range": [0.0, 0.7]},
}
 
# ── Plot Generation ───────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"{m['label']}  {m['better']}" for name, m in metrics.items()],
    horizontal_spacing=0.05,
)

for i, (name, m) in enumerate(metrics.items()):
    col = i + 1
 
    for fl_val in fl_vals:
        if fl_val not in dfs or dfs[fl_val].empty:
            continue
        df = dfs[fl_val]
        
        show_legend = (i == 0)
        style = fl_styles[fl_val]
        transparent_line_color = hex_to_rgba(style["color"], opacity=0.5)
        
        line_trace = go.Scatter(
            x=df["K_str"].tolist(), 
            y=df[name].tolist(),
            mode="lines+markers",
            name=f"fl = {fl_val} mm",
            line=dict(color=transparent_line_color, width=1.2, dash=style["dash"]),
            marker=dict(
                size=4.0, 
                symbol=style["symbol"], 
                color=style["color"], 
                line=dict(width=0.4, color=style["color"])
            ),
            showlegend=show_legend,
        )
        fig.add_trace(line_trace, row=1, col=col)
 
    # Force uniform categorical layout spacing
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=str_target_ks,
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

# 3.5% Margins and layout domains configuration
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

# ── FIXED REFINEMENT: ALIGN MARKER ANNOTATIONS VIA INDEX VALUES ───────────────
annotations = list(fig.layout.annotations)

for pt in optimal_lpips_points:
    # Manual pixel displacement tuning based on their shared K destinations
    if pt["fl"] == 28:    # K=15
        offset_x, offset_y = 35, 20
    elif pt["fl"] == 36:  # K=15
        offset_x, offset_y = 35, 30
    elif pt["fl"] == 45:  # K=15
        offset_x, offset_y = 35, -10
    elif pt["fl"] == 60:  # K=15
        offset_x, offset_y = 35, -10
    elif pt["fl"] == 70:  # K=15
        offset_x, offset_y = 35, -12
    
    annotations.append(
        go.layout.Annotation(
            x=pt["x_idx"],  # FIXED: Numeric index mapping (0 for '15', 2 for '25')
            y=pt["y"],      
            xref="x3",      
            yref="y3",      
            text=f"<b>K={pt['raw_k']:.1f}, {pt['y']:.2f}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,  
            arrowwidth=1.0,
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
        x=0.5,                  
        y=0.01,                 
        xanchor="center",
        yanchor="bottom",
        font=dict(family="Times New Roman", size=6.0, color="#222222"),
    )
)

out_html = "./bokehdiff_stage_2_all_fls.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out = "./bokehdiff_stage_2_all_fls.png"
try:
    fig.write_image(out, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved → {out}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open the HTML output to view.")

