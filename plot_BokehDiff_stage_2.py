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
 
# ── Configuration & Input Selection ───────────────────────────────────────────
TARGET_FL = 60

fl_colors = {
    28: "#CC6677",  # Rose
    36: "#44AA99",  # Teal
    45: "#DDCC77",  # Sand
    60: "#882255"   # Wine
}
PLOT_COLOR = fl_colors.get(TARGET_FL, "#4477AA") 

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

# ── Load & Parse Data ─────────────────────────────────────────────────────────
file_path = f"./bokehdiff/results_bokehdiff_fl_{TARGET_FL}.csv"
if not os.path.exists(file_path):
    file_path = f"results_bokehdiff_fl_{TARGET_FL}.csv"

try:
    raw = pd.read_csv(file_path, skiprows=6, header=None)
    
    raw = raw.dropna(how='all', axis=1)
    raw = raw.dropna(how='all', axis=0)

    if raw.shape[1] >= 4:
        raw.columns = ["folder_name"] + [f"extra_{idx}" for idx in range(raw.shape[1] - 4)] + ["PSNR", "SSIM", "LPIPS"]
    else:
        raise ValueError(f"Unexpected layout columns in fl_{TARGET_FL}.")
        
    data_rows = raw[raw["folder_name"].str.contains("K|bokeh", na=False, case=False)].copy()
    if data_rows.empty:
        data_rows = raw.copy()  
    
    data_rows["PSNR"]  = data_rows["PSNR"].astype(float)
    data_rows["SSIM"]  = data_rows["SSIM"].astype(float)
    data_rows["LPIPS"] = data_rows["LPIPS"].astype(float)
    
    if TARGET_FL == 28:
        data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
        data_rows["fp"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "fp_"))
        data_rows = data_rows[np.isclose(data_rows["fp"], 0.2)]
    else:
        data_rows["K"] = data_rows["folder_name"].apply(lambda x: extract_param(x, "K"))
        
    df = data_rows.dropna(subset=["K"])[["K", "PSNR", "SSIM", "LPIPS"]].sort_values(by="K").reset_index(drop=True)
    
    if df.empty:
        raise ValueError(f"No matching parameters left for fl_{TARGET_FL} after cleaning.")
        
    print(f"Successfully loaded fl_{TARGET_FL}: parsed {len(df)} entries.")

except Exception as e:
    print(f"Fatal Error: Could not load or parse file for fl_{TARGET_FL}: {e}")
    exit()

# Define metrics and their standard plotting attributes
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

selected_K_ticks = df["K"].tolist()

for i, (name, m) in enumerate(metrics.items()):
    col = i + 1
    
    line_trace = go.Scatter(
        x=df["K"].tolist(),
        y=df[name].tolist(),
        mode="lines+markers",
        line=dict(color=PLOT_COLOR, width=1.2, dash="dash"),
        marker=dict(size=5, symbol="circle", color=PLOT_COLOR, line=dict(width=0.5, color="#222222")),
        showlegend=False  # Removed trace from layout legend completely
    )
    fig.add_trace(line_trace, row=1, col=col)
 
    fig.update_xaxes(
        title=dict(text="K Value", font=dict(family="Times New Roman", size=6.1, color="#222222"), standoff=3),
        tickfont=dict(family="Times New Roman", size=5.4, color="#333333"),
        tickmode="array",
        tickvals=selected_K_ticks,
        range=[min(selected_K_ticks) - (max(selected_K_ticks)*0.05), max(selected_K_ticks) + (max(selected_K_ticks)*0.05)],
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

x_domains = [[0.035, 0.3106], [0.3620, 0.6380], [0.6894, 0.9650]]
y_domain  = [0.16, 0.95]

fig.update_layout(
    xaxis=dict(domain=x_domains[0]), xaxis2=dict(domain=x_domains[1]), xaxis3=dict(domain=x_domains[2]),
    yaxis=dict(domain=y_domain), yaxis2=dict(domain=y_domain), yaxis3=dict(domain=y_domain)
)

x_centers = [sum(domain) / 2 for domain in x_domains]
for idx, ann in enumerate(fig.layout.annotations):
    ann.font = dict(family="Times New Roman", size=6.8, color="#222222")
    ann.x = x_centers[idx]
    ann.y = y_domain[1] + 0.02

# Layout modifications to strictly hide global legends and annotations
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Times New Roman", size=6.1, color="#222222"),
    width=WIDTH_PX,
    height=HEIGHT_PX,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,  # Explicitly disables the canvas legend structure entirely
)

# Output generation
out_html = f"./bokehdiff_fl_{TARGET_FL}.html"
fig.write_html(out_html)
print(f"Saved Interactive Plot → {out_html}")

out_png = f"./bokehdiff_fl_{TARGET_FL}.png"
try:
    fig.write_image(out_png, width=WIDTH_PX, height=HEIGHT_PX, scale=SCALE)
    print(f"Saved Static Figure → {out_png}")
except Exception:
    print("Static image generation requires 'pip install kaleido'. Open the HTML output to view.")