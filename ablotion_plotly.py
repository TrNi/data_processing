import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Data ──────────────────────────────────────────────────────
focal_lengths = [28, 36, 45, 60, 70]

bd_psnr  = [27.53, 27.45, 21.72, 21.79, 21.37]
bd_ssim  = [0.92, 0.93, 0.88, 0.88, 0.85]
bd_lpips = [0.17, 0.16, 0.19, 0.23, 0.26]
dr_psnr  = [26.40, 26.55, 21.38, 21.68, 21.22]
dr_ssim  = [0.91, 0.92, 0.87, 0.89, 0.86]
dr_lpips = [0.22, 0.19, 0.24, 0.23, 0.25]
bm_psnr  = [25.75, 25.15, 20.67, 20.98, 21.02]
bm_ssim  = [0.89, 0.89, 0.85, 0.87, 0.86]
bm_lpips = [0.23, 0.23, 0.25, 0.25, 0.25]
bl28_psnr  = [23.45, 22.91, 19.44, 19.74, 19.99]
bl28_ssim  = [0.78, 0.79, 0.75, 0.79, 0.80]
bl28_lpips = [0.46, 0.45, 0.46, 0.43, 0.37]
bl45_psnr  = [24.79, 24.43, 20.25, 20.56, 20.68]
bl45_ssim  = [0.82, 0.84, 0.80, 0.83, 0.84]
bl45_lpips = [0.39, 0.37, 0.38, 0.35, 0.31]

res_psnr  = [27.19, 28.51, 26.99, 26.42, 25.62]
res_ssim  = [0.91, 0.91, 0.89, 0.90, 0.87]
res_lpips = [0.03, 0.03, 0.05, 0.06, 0.08]
nrk_psnr  = [27.50, 28.97, 27.30, 26.41, 25.53]
nrk_ssim  = [0.91, 0.91, 0.89, 0.89, 0.87]
nrk_lpips = [0.04, 0.03, 0.06, 0.08, 0.11]
vit_psnr  = [26.83, 27.94, 27.00, 26.54, 25.79]
vit_ssim  = [0.91, 0.91, 0.89, 0.90, 0.87]
vit_lpips = [0.04, 0.03, 0.05, 0.06, 0.08]

apertures = [2.8, 5, 9, 16]
ap_vit_psnr  = [25.79, 24.77, 26.30, 30.48]; ap_vit_ssim  = [0.8735, 0.8317, 0.8545, 0.9199]; ap_vit_lpips = [0.0836, 0.0668, 0.0467, 0.0258]
ap_res_psnr  = [25.62, 24.96, 26.93, 31.96]; ap_res_ssim  = [0.8729, 0.8353, 0.8596, 0.9275]; ap_res_lpips = [0.0750, 0.0656, 0.0454, 0.0231]
ap_nrk_psnr  = [25.53, 24.90, 26.82, 31.95]; ap_nrk_ssim  = [0.8668, 0.8326, 0.8581, 0.9268]; ap_nrk_lpips = [0.1138, 0.0860, 0.0531, 0.0262]

# ── Helpers ───────────────────────────────────────────────────
MARKERS = ['circle', 'triangle-up', 'square', 'diamond', 'x']
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def make_trace(x, y, name, i, showlegend=True):
    return go.Scatter(
        x=x, y=y, name=name, mode='lines+markers',
        line=dict(dash='dash', width=2, color=COLORS[i % len(COLORS)]),
        marker=dict(symbol=MARKERS[i % len(MARKERS)], size=8),
        showlegend=showlegend
    )

def grid_layout(fig, xvals, xlabel):
    fig.update_xaxes(tickvals=xvals, title_text=xlabel)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', griddash='dash')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', griddash='dash')
    fig.update_layout(
        font=dict(family='serif', size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2, bgcolor='rgba(0,0,0,0)'),
        margin=dict(b=100)
    )

def save(fig, stem):
    fig.write_html(f"{stem}.html")
    fig.write_image(f"{stem}.png", scale=2, width=1400, height=400)
    fig.write_image(f"{stem}.pdf", width=1400, height=400)
    print(f"Saved {stem}.html / .png / .pdf")


# ── 1. DoF combined ───────────────────────────────────────────
dof_names = ['BokehDiff', 'Dr.Bokeh', 'BokehMe', 'Bokehlicious-2.8', 'Bokehlicious-4.5']
dof_psnr  = [bd_psnr, dr_psnr, bm_psnr, bl28_psnr, bl45_psnr]
dof_ssim  = [bd_ssim, dr_ssim, bm_ssim, bl28_ssim, bl45_ssim]
dof_lpips = [bd_lpips, dr_lpips, bm_lpips, bl28_lpips, bl45_lpips]

fig = make_subplots(rows=1, cols=3, subplot_titles=("PSNR ↑", "SSIM ↑", "LPIPS ↓"))
for i, (name, p, s, l) in enumerate(zip(dof_names, dof_psnr, dof_ssim, dof_lpips)):
    fig.add_trace(make_trace(focal_lengths, p, name, i),               row=1, col=1)
    fig.add_trace(make_trace(focal_lengths, s, name, i, False),        row=1, col=2)
    fig.add_trace(make_trace(focal_lengths, l, name, i, False),        row=1, col=3)
grid_layout(fig, focal_lengths, "Focal Length (mm)")
fig.update_yaxes(title_text="PSNR ↑",  row=1, col=1)
fig.update_yaxes(title_text="SSIM ↑",  row=1, col=2)
fig.update_yaxes(title_text="LPIPS ↓", row=1, col=3)
save(fig, "dof_focal_combined")


# ── 2. DoF LPIPS only ─────────────────────────────────────────
fig = go.Figure()
for i, (name, l) in enumerate(zip(dof_names, dof_lpips)):
    fig.add_trace(make_trace(focal_lengths, l, name, i))
fig.update_xaxes(tickvals=focal_lengths, title_text="Focal Length (mm)",
                 showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_yaxes(title_text="LPIPS ↓", showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_layout(font=dict(family='serif', size=14), plot_bgcolor='white',
                  paper_bgcolor='white', width=500, height=450,
                  legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.3))
fig.write_html("dof_focal_lpips_only.html")
fig.write_image("dof_focal_lpips_only.png", scale=2)
fig.write_image("dof_focal_lpips_only.pdf")
print("Saved dof_focal_lpips_only.*")


# ── 3. Defocus deblurring combined ────────────────────────────
dd_names = ['ViT', 'Restormer', 'NRKNet']
dd_psnr  = [vit_psnr,  res_psnr,  nrk_psnr]
dd_ssim  = [vit_ssim,  res_ssim,  nrk_ssim]
dd_lpips = [vit_lpips, res_lpips, nrk_lpips]

fig = make_subplots(rows=1, cols=3, subplot_titles=("PSNR ↑", "SSIM ↑", "LPIPS ↓"))
for i, (name, p, s, l) in enumerate(zip(dd_names, dd_psnr, dd_ssim, dd_lpips)):
    fig.add_trace(make_trace(focal_lengths, p, name, i),               row=1, col=1)
    fig.add_trace(make_trace(focal_lengths, s, name, i, False),        row=1, col=2)
    fig.add_trace(make_trace(focal_lengths, l, name, i, False),        row=1, col=3)
grid_layout(fig, focal_lengths, "Focal Length (mm)")
fig.update_yaxes(title_text="PSNR ↑",  row=1, col=1)
fig.update_yaxes(title_text="SSIM ↑",  row=1, col=2)
fig.update_yaxes(title_text="LPIPS ↓", row=1, col=3)
save(fig, "focal_combined")


# ── 4. Defocus deblurring LPIPS only ──────────────────────────
fig = go.Figure()
for i, (name, l) in enumerate(zip(dd_names, dd_lpips)):
    fig.add_trace(make_trace(focal_lengths, l, name, i))
fig.update_xaxes(tickvals=focal_lengths, title_text="Focal Length (mm)",
                 showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_yaxes(title_text="LPIPS ↓", showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_layout(font=dict(family='serif', size=14), plot_bgcolor='white',
                  paper_bgcolor='white', width=500, height=450,
                  legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.3))
fig.write_html("focal_lpips_only.html")
fig.write_image("focal_lpips_only.png", scale=2)
fig.write_image("focal_lpips_only.pdf")
print("Saved focal_lpips_only.*")


# ── 5. Aperture ablation combined ─────────────────────────────
ap_names = ['ViT', 'Restormer', 'NRKNet']
ap_psnr  = [ap_vit_psnr,  ap_res_psnr,  ap_nrk_psnr]
ap_ssim  = [ap_vit_ssim,  ap_res_ssim,  ap_nrk_ssim]
ap_lpips = [ap_vit_lpips, ap_res_lpips, ap_nrk_lpips]

fig = make_subplots(rows=1, cols=3, subplot_titles=("PSNR ↑", "SSIM ↑", "LPIPS ↓"))
for i, (name, p, s, l) in enumerate(zip(ap_names, ap_psnr, ap_ssim, ap_lpips)):
    fig.add_trace(make_trace(apertures, p, name, i),               row=1, col=1)
    fig.add_trace(make_trace(apertures, s, name, i, False),        row=1, col=2)
    fig.add_trace(make_trace(apertures, l, name, i, False),        row=1, col=3)
grid_layout(fig, apertures, "Aperture (f-number)")
fig.update_yaxes(title_text="PSNR ↑",  row=1, col=1)
fig.update_yaxes(title_text="SSIM ↑",  row=1, col=2)
fig.update_yaxes(title_text="LPIPS ↓", row=1, col=3)
save(fig, "ablation_combined")


# ── 6. Aperture ablation LPIPS only ───────────────────────────
fig = go.Figure()
for i, (name, l) in enumerate(zip(ap_names, ap_lpips)):
    fig.add_trace(make_trace(apertures, l, name, i))
fig.update_xaxes(tickvals=apertures, title_text="Aperture (f-number)",
                 showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_yaxes(title_text="LPIPS ↓", showgrid=True, gridcolor='lightgrey', griddash='dash')
fig.update_layout(font=dict(family='serif', size=14), plot_bgcolor='white',
                  paper_bgcolor='white', width=500, height=450,
                  legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.3))
fig.write_html("ablation_lpips_only.html")
fig.write_image("ablation_lpips_only.png", scale=2)
fig.write_image("ablation_lpips_only.pdf")
print("Saved ablation_lpips_only.*")