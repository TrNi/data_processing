'''
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load & parse (Maintained exactly from original) ───────────────────────────
raw = pd.read_csv("./bokehme/results_bokehme_fl_28.csv", header=None)
data_rows = raw[raw[0].str.startswith("bokehme_dfs", na=False)].copy()
 
def parse_name(name):
    dfs = int(re.search(r"dfs(\d+)",          name).group(1))
    K   = int(re.search(r"K(\d+)",             name).group(1))
    df  = float(re.search(r"dispfocus([\d.]+)", name).group(1))
    return dfs, K, df
 
parsed                 = data_rows[0].apply(parse_name)
data_rows["dfs"]       = parsed.apply(lambda x: x[0])
data_rows["K"]         = parsed.apply(lambda x: x[1])
data_rows["dispfocus"] = parsed.apply(lambda x: x[2])
data_rows["PSNR"]      = data_rows[6].astype(float)
data_rows["SSIM"]      = data_rows[7].astype(float)
data_rows["LPIPS"]     = data_rows[8].astype(float)
 
df = data_rows[["dfs", "K", "dispfocus", "PSNR", "SSIM", "LPIPS"]].reset_index(drop=True)
 
dfs_vals  = sorted(df["dfs"].unique())
K_vals    = sorted(df["K"].unique())
disp_vals = sorted(df["dispfocus"].unique())
 
KK, DD, FF = np.meshgrid(K_vals, disp_vals, dfs_vals, indexing='ij')
 
def make_grid3d(col):
    lookup = {(r.K, r.dispfocus, r.dfs): getattr(r, col) for r in df.itertuples()}
    cube = np.full((len(K_vals), len(disp_vals), len(dfs_vals)), np.nan)
    for i, k in enumerate(K_vals):
        for j, d in enumerate(disp_vals):
            for l, f in enumerate(dfs_vals):
                if (k, d, f) in lookup:
                    cube[i, j, l] = lookup[(k, d, f)]
    return cube
 
metrics = {
    "PSNR":  {"cube": make_grid3d("PSNR"),  "colorscale": "YlGn", "label": "PSNR (dB)", "better": "↑"},
    "SSIM":  {"cube": make_grid3d("SSIM"),  "colorscale": "YlGn", "label": "SSIM",      "better": "↑"},
    "LPIPS": {"cube": make_grid3d("LPIPS"), "colorscale": "YlGn_r", "label": "LPIPS",    "better": "↓"},
}
 
# ── Layout Dimensions ────────────────────────────────────────────────────────
WIDTH_PX  = 1680
HEIGHT_PX = 430
DPI       = 600            
BASE_PPI  = 100
SCALE     = DPI / BASE_PPI

# 1. Overlapping domains to force the actual 3D objects closer together
domains = [
    [0.00, 0.36],   # Subplot 1
    [0.32, 0.68],   # Subplot 2
    [0.64, 1.00]    # Subplot 3
]

# 2. Dynamically position colorbars right against the edge of the expanded domains
cbar_x_positions = [
    domains[0][1] - 0.015,
    domains[1][1] - 0.015,
    domains[2][1] - 0.015
]
 
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=[f"<b>{m['label']}</b> {m['better']}" for m in metrics.values()],
    horizontal_spacing=0.0, 
)
 
AXIS_TITLE_FONT = dict(family="Times New Roman", size=12, color="#222222")
AXIS_TICK_FONT  = dict(family="Times New Roman", size=9.5, color="#333333")
 
for i, (name, m) in enumerate(metrics.items()):
    cube = m["cube"]
    col  = i + 1
 
    volume = go.Volume(
        x=KK.flatten(),
        y=DD.flatten(),
        z=FF.flatten(),
        value=cube.flatten(),
        colorscale=m["colorscale"],
        isomin=float(np.nanmin(cube)),
        isomax=float(np.nanmax(cube)),
        opacity=0.12,          
        surface_count=15,      
        colorbar=dict(
            title=dict(
                text=m["label"],
                font=dict(family="Times New Roman", size=10, color="#222222"),
                side="right",
            ),
            tickfont=dict(family="Times New Roman", size=8.5, color="#333333"),
            thickness=8, 
            len=0.85,
            x=cbar_x_positions[i],
            xanchor="right",   # Anchor right so it directly aligns next to the cube 
            outlinewidth=0.6,
            outlinecolor="#555555",
            ticklen=2,
            tickwidth=0.5,
        ),
        showscale=True,
        hovertemplate=(
            "K=%{x}<br>dispfocus=%{y}<br>dfs=%{z}"
            f"<br>{m['label']}=%{{value:.4f}}<extra></extra>"
        ),
    )
    fig.add_trace(volume, row=1, col=col)
 
# ── Scene Axis Limits & Styling ─────────────────────────────────────────────
K_range    = [K_vals[0] - 4, K_vals[-1] + 4]
disp_range = [disp_vals[0] - 0.06, disp_vals[-1] + 0.06]
dfs_span   = dfs_vals[-1] - dfs_vals[0]
dfs_range  = [dfs_vals[0] - 0.55 * dfs_span, dfs_vals[-1] + 0.20 * dfs_span]
 
scene_style = dict(
    xaxis=dict(
        title=dict(text="K", font=AXIS_TITLE_FONT),
        tickvals=K_vals, tickfont=AXIS_TICK_FONT, range=K_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="dispfocus", font=AXIS_TITLE_FONT),
        tickvals=disp_vals, tickfont=AXIS_TICK_FONT, range=disp_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    zaxis=dict(
        title=dict(text="dfs", font=AXIS_TITLE_FONT),
        tickvals=dfs_vals, tickfont=AXIS_TICK_FONT, range=dfs_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    camera=dict(eye=dict(x=1.40, y=1.40, z=1.15)), # Shifted slightly closer still
    aspectmode='manual',
    aspectratio=dict(x=1, y=1, z=0.65), # Lowered Z height ratio to encourage box expansion
)

fig.update_layout(
    scene=scene_style, 
    scene2=scene_style, 
    scene3=scene_style
)
 
fig.update_layout(
    scene =dict(domain=dict(x=domains[0])),
    scene2=dict(domain=dict(x=domains[1])),
    scene3=dict(domain=dict(x=domains[2])),
)
 
# Re-center title coordinates dynamically based on the adjusted domains
for idx, ann in enumerate(fig.layout.annotations):
    ann.font = dict(family="Times New Roman", size=13, color="#222222")
    ann.y = 0.98  
    ann.yanchor = "bottom"
    ann.x = sum(domains[idx]) / 2.0
 
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=20, b=0),
    showlegend=False,
)
 
fig.write_html("./bokehme_4d_surface_plots_stage_1.html", config={"responsive": False})
try:
    fig.write_image("./bokehme_4d_surface_plots_stage_1.png", width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print("Successfully generated tightly-packed visualization layout.")
except Exception as e:
    print(f"HTML saved. Static image generation skipped: {e}")


'''


import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load & parse (Maintained exactly from original) ───────────────────────────
raw = pd.read_csv("./bokehme/results_bokehme_fl_28.csv", header=None)
data_rows = raw[raw[0].str.startswith("bokehme_dfs", na=False)].copy()
 
def parse_name(name):
    dfs = int(re.search(r"dfs(\d+)",          name).group(1))
    K   = int(re.search(r"K(\d+)",             name).group(1))
    df  = float(re.search(r"dispfocus([\d.]+)", name).group(1))
    return dfs, K, df
 
parsed                 = data_rows[0].apply(parse_name)
data_rows["dfs"]       = parsed.apply(lambda x: x[0])
data_rows["K"]         = parsed.apply(lambda x: x[1])
data_rows["dispfocus"] = parsed.apply(lambda x: x[2])
data_rows["PSNR"]      = data_rows[6].astype(float)
data_rows["SSIM"]      = data_rows[7].astype(float)
data_rows["LPIPS"]     = data_rows[8].astype(float)
 
df = data_rows[["dfs", "K", "dispfocus", "PSNR", "SSIM", "LPIPS"]].reset_index(drop=True)
 
dfs_vals  = sorted(df["dfs"].unique())
K_vals    = sorted(df["K"].unique())
disp_vals = sorted(df["dispfocus"].unique())
 
KK, DD, FF = np.meshgrid(K_vals, disp_vals, dfs_vals, indexing='ij')
 
def make_grid3d(col):
    lookup = {(r.K, r.dispfocus, r.dfs): getattr(r, col) for r in df.itertuples()}
    cube = np.full((len(K_vals), len(disp_vals), len(dfs_vals)), np.nan)
    for i, k in enumerate(K_vals):
        for j, d in enumerate(disp_vals):
            for l, f in enumerate(dfs_vals):
                if (k, d, f) in lookup:
                    cube[i, j, l] = lookup[(k, d, f)]
    return cube

# ── Find Optimal Parameters for Each Metric ───────────────────────────────────
opt_psnr  = df.loc[df["PSNR"].idxmax()]
opt_ssim  = df.loc[df["SSIM"].idxmax()]
opt_lpips = df.loc[df["LPIPS"].idxmin()]

metrics = {
    "PSNR":  {
        "cube": make_grid3d("PSNR"),  
        "colorscale": "turbo_r",
        "reversescale": False, 
        "label": "PSNR (dB)", 
        "better": "↑",
        "opt_coords": (opt_psnr["K"], opt_psnr["dispfocus"], opt_psnr["dfs"]),
        "opt_val": opt_psnr["PSNR"]
    },
    "SSIM":  {
        "cube": make_grid3d("SSIM"),  
        "colorscale": "turbo_r",
        "reversescale": False, 
        "label": "SSIM",      
        "better": "↑",
        "opt_coords": (opt_ssim["K"], opt_ssim["dispfocus"], opt_ssim["dfs"]),
        "opt_val": opt_ssim["SSIM"]
    },
    "LPIPS": {
        "cube": make_grid3d("LPIPS"), 
        "colorscale": "turbo",
        "reversescale": False, 
        "label": "LPIPS",    
        "better": "↓",
        "opt_coords": (opt_lpips["K"], opt_lpips["dispfocus"], opt_lpips["dfs"]),
        "opt_val": opt_lpips["LPIPS"]
    },
}
 
# ── Layout Dimensions ────────────────────────────────────────────────────────
WIDTH_PX  = 1680
HEIGHT_PX = 430
DPI       = 600            
BASE_PPI  = 100
SCALE     = DPI / BASE_PPI

domains = [
    [0.00, 0.36],   
    [0.32, 0.68],   
    [0.64, 1.00]    
]

cbar_x_positions = [
    domains[0][1] - 0.02,
    domains[1][1] - 0.02,
    domains[2][1] - 0.02
]
 
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
    subplot_titles=[f"<b>{m['label']}</b> {m['better']}" for m in metrics.values()],
    horizontal_spacing=0.0, 
)
 
AXIS_TITLE_FONT = dict(family="Times New Roman", size=12, color="#222222")
AXIS_TICK_FONT  = dict(family="Times New Roman", size=9.5, color="#333333")

K_range    = [K_vals[0] - 4, K_vals[-1] + 4]
disp_range = [disp_vals[0] - 0.06, disp_vals[-1] + 0.06]
dfs_span   = dfs_vals[-1] - dfs_vals[0]
dfs_range  = [dfs_vals[0] - 0.55 * dfs_span, dfs_vals[-1] + 0.20 * dfs_span]

# Dictionaries to store valid 3D annotations
scene_annotations = {"scene": [], "scene2": [], "scene3": []}
 
for i, (name, m) in enumerate(metrics.items()):
    cube = m["cube"]
    col  = i + 1
    scene_key = "scene" if col == 1 else f"scene{col}"

    # Calculate bounds clipping the bottom 5th and top 5th percentiles
    isomin_val = float(np.nanpercentile(cube, 5))
    isomax_val = float(np.nanpercentile(cube, 95))

    volume = go.Volume(
        x=KK.flatten(),
        y=DD.flatten(),
        z=FF.flatten(),
        value=cube.flatten(),
        colorscale=m["colorscale"],
        reversescale=m["reversescale"],
        isomin=isomin_val,
        isomax=isomax_val,
        opacity=0.12,          
        surface_count=15,      
        colorbar=dict(
            #title=dict(
            #    text=m["label"],
            #    font=dict(family="Times New Roman", size=10, color="#222222"),
            #    side="right",
            #),
            tickfont=dict(family="Times New Roman", size=8.5, color="#333333"),
            thickness=8, 
            len=0.85,
            x=cbar_x_positions[i],
            xanchor="right",   
            outlinewidth=0.6,
            outlinecolor="#555555",
            ticklen=2,
            tickwidth=0.5,
        ),
        showscale=True,
        hovertemplate=(
            "K=%{x}<br>dispfocus=%{y}<br>dfs=%{z}"
            f"<br>{m['label']}=%{{value:.4f}}<extra></extra>"
        ),
    )
    fig.add_trace(volume, row=1, col=col)

    # ── Add Red Cross Optimal Marker Trace ─────────────────────────────────────
    opt_x, opt_y, opt_z = m["opt_coords"]
    
    optimal_marker = go.Scatter3d(
        x=[opt_x],
        y=[opt_y],
        z=[opt_z],
        mode="markers",
        marker=dict(
            size=10,
            symbol="cross",    
            color="#D62728",   
            line=dict(color="#222222", width=1.5)
        ),
        hovertemplate=(
            f"<b>Optimal {name} Point</b><br>"
            "K=%{x}<br>dispfocus=%{y}<br>dfs=%{z}<br>"
            f"{m['label']}={m['opt_val']:.4f}<extra></extra>"
        )
    )
    fig.add_trace(optimal_marker, row=1, col=col)

    # ── Configure 3D Arrow and Text Annotation (Strictly Valid Properties) ───
    annotation_text = (
        f"<b>Optimal {name} Marker</b><br>"
        f"Value: {m['opt_val']:.4f}<br>"
        f"dispfocus: {opt_y}, dfs: {opt_z}"
    )
    
    scene_annotations[scene_key].append(
        dict(
            showarrow=True,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#D62728",
            # Coords where the arrow points (Data units)
            x=opt_x,
            y=opt_y,
            z=opt_z,
            # Position offset for the textbox text block relative to target (in Screen Pixels)
            ax=40,    # 40 pixels right on screen
            ay=-50,   # 50 pixels up on screen
            text=annotation_text,
            font=dict(family="Times New Roman", size=9, color="#222222"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#D62728",
            borderwidth=1,
            borderpad=4
        )
    )
 
# ── Scene Axis Limits & Styling ─────────────────────────────────────────────
scene_style = dict(
    xaxis=dict(
        title=dict(text="K (Blur Strength)", font=AXIS_TITLE_FONT),
        tickvals=K_vals, tickfont=AXIS_TICK_FONT, range=K_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="dispfocus", font=AXIS_TITLE_FONT),
        tickvals=disp_vals, tickfont=AXIS_TICK_FONT, range=disp_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    zaxis=dict(
        title=dict(text="dfs", font=AXIS_TITLE_FONT),
        tickvals=dfs_vals, tickfont=AXIS_TICK_FONT, range=dfs_range,
        backgroundcolor="white", gridcolor="#cccccc", gridwidth=0.4,
        showline=True, linecolor="#555555", linewidth=0.6, zeroline=False,
    ),
    camera=dict(eye=dict(x=1.40, y=1.40, z=1.15)), 
    aspectmode='manual',
    aspectratio=dict(x=1, y=1, z=0.65), 
)

# Apply validated baseline styles alongside annotations array
fig.update_layout(
    scene=dict(**scene_style, annotations=scene_annotations["scene"], domain=dict(x=domains[0])),
    scene2=dict(**scene_style, annotations=scene_annotations["scene2"], domain=dict(x=domains[1])),
    scene3=dict(**scene_style, annotations=scene_annotations["scene3"], domain=dict(x=domains[2]))
)
 
# Re-center title coordinates dynamically based on the adjusted domains
for idx, ann in enumerate(fig.layout.annotations):
    ann.font = dict(family="Times New Roman", size=13, color="#222222")
    ann.y = 0.98  
    ann.yanchor = "bottom"
    ann.x = sum(domains[idx]) / 2.0
 
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=20, b=0),
    showlegend=False,
)
 
fig.write_html("./bokehme_4d_surface_plots_stage_1.html", config={"responsive": False})
try:
    fig.write_image("./bokehme_4d_surface_plots_stage_1.png", width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print("Successfully generated visualization with clean optimal parameters annotations.")
except Exception as e:
    print(f"HTML saved. Static image generation skipped: {e}")