import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
from pathlib import Path
import os
import csv
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 5,
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.autolayout': False,
    'figure.constrained_layout.use': False,
})
plt.rcParams['mathtext.fontset'] = 'stix'  # for math equations to also use serif fonts
plt.rcParams['axes.titlepad'] = 0.2
def get_pretty_name(name):
    """Converts a filename string to a display-friendly name."""
    name = name.lower() # Make matching case-insensitive
    if 'monster' in name: return 'MonSter'
    if 'foundation' in name: return 'Foundation Stereo'
    if 'defom' in name: return 'DEFOM Stereo'
    if 'selective' in name: return 'Selective IGEV'
    if 'depthpro' in name: return 'Depth Pro'
    if 'metric3d' in name: return 'Metric3D V2'
    if 'unidepth' in name: return 'UniDepth V2'
    if 'depth_anything' in name: return 'DAV2'
    return name # Return original name if no match

def plot_depth_maps(depth_data, mono_models, stereo_models, save_dir, idx):
    """Plot all depth maps (4 mono on top, 4 stereo on bottom) with single colorbar"""
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(7.6, 4))
    gs = plt.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05])
    
    # Calculate overall min/max from percentiles
    all_mins = []
    all_maxs = []
    for i in range(len(depth_data)):
        data = depth_data[i]
        valid_data = data[~np.isnan(data)]
        all_mins.append(np.percentile(valid_data, 5))
        all_maxs.append(np.percentile(valid_data, 95))
    
    vmin = np.mean(all_mins)
    vmax = np.mean(all_maxs)
    
    # Create trimmed turbo colormap (removing 2% from each end)
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)  # 2% trim
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)
    
    plt.suptitle(f'Depth Maps - Image {idx}')
    
    # Plot mono models on top row
    for i, model in enumerate(mono_models):
        model_name = get_pretty_name(model)
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(depth_data[i], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Mono - {model_name}')
        ax.axis('off')
    
    # Plot stereo models on bottom row
    for i, model in enumerate(stereo_models):
        model_name = get_pretty_name(model)
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(depth_data[i+len(mono_models)], cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
        ax.set_title(f'Stereo - {model_name}')
        ax.axis('off')
    
    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cbar_ax, label='Depth')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'depth_maps_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_maps(error_maps, model_names, save_dir, idx):
    """Plot error maps (1 row, 4 image columns for selected models).
    Preserves existing subplot geometry; places full-height colorbars by creating
    dedicated axes from each image Axes.get_position().
    """
    if idx != 7:
        return

    error_plotnames = {"grad": "Gradient", "plan": "Planarity", "icp": "ICP", "iqr": "IQR"}
    error_types = ['grad', 'plan', 'icp', 'iqr']

    fig = plt.figure(figsize=(5.76, 1.2))
    gs = plt.GridSpec(1, 7, width_ratios=[1, 1, 0.001, 1, 0.001, 1, 0.001], height_ratios=[1])

    # trimmed turbo as before
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)

    # horizontal gap (figure coordinates) between image right edge and cbar left edge
    H_GAP = 0.0005
    # colorbar width (figure coordinates); small consistent width
    CB_WIDTH = 0.006

    for i, model in enumerate(model_names):
        model_name = get_pretty_name(model)
        if "foundation" not in model_name.lower():
            continue

        # compute robust percentiles per-map for consistent vmin/vmax groups
        error_mins, error_maxs = [], []
        for error_type in error_types:
            scale = 1e6 if error_type == 'plan' else 1
            data = error_maps[model_name][error_type][idx] * scale
            valid = data[~np.isnan(data)]
            error_mins.append(np.percentile(valid, 5))
            error_maxs.append(np.percentile(valid, 95))

        vmin1, vmax1 = min(error_mins[:2]), max(error_maxs[:2])
        vmin2, vmax2 = error_mins[2], error_maxs[2]
        vmin3, vmax3 = error_mins[3], error_maxs[3]

        for j, error_type in enumerate(error_types):
            scale = 1e6 if error_type == 'plan' else 1
            data = error_maps[model_name][error_type][idx] * scale

            if j in [0, 1]:
                ax = fig.add_subplot(gs[0, j])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin1, vmax=vmax1)
                # only place a colorbar for the second of the two images to match original
                if j == 1:
                    pos_img = ax.get_position()  # Bbox in figure coords
                    # create a new axes for the colorbar using exact image height
                    cax_pos = [
                        pos_img.x1 - 0.011,# + H_GAP,     # left
                        pos_img.y0,             # bottom aligned to image bottom
                        CB_WIDTH,               # width
                        pos_img.height          # full image height
                    ]
                    cax = fig.add_axes(cax_pos)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    # remove border/outline and frame
                    cbar.outline.set_visible(False)
                    cax.set_frame_on(False)
                    cbar.ax.tick_params(
                        axis='y', direction='out', length=0.1, pad=0.1, labelsize=4
                    )

            elif j == 2:
                ax = fig.add_subplot(gs[0, 3])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin2, vmax=vmax2)
                pos_img = ax.get_position()
                cax_pos = [
                    pos_img.x1 + 0.035, # + 0.02,    # keep same small gap
                    pos_img.y0,            # align bottom
                    CB_WIDTH,              # consistent width
                    pos_img.height         # exact height
                ]
                cax = fig.add_axes(cax_pos)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                cbar.outline.set_visible(False)
                cax.set_frame_on(False)
                cbar.ax.tick_params(axis='y', direction='out', length=0.1, pad=0.1, labelsize=4)

            elif j == 3:
                ax = fig.add_subplot(gs[0, 5])
                im = ax.imshow(data, cmap=trimmed_turbo, vmin=vmin3, vmax=vmax3)
                pos_img = ax.get_position()
                cax_pos = [
                    pos_img.x1 + 0.08,    # keep same small gap
                    pos_img.y0,            # align bottom
                    CB_WIDTH,              # consistent width
                    pos_img.height         # exact height
                ]
                cax = fig.add_axes(cax_pos)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                cbar.outline.set_visible(False)
                cax.set_frame_on(False)
                cbar.ax.tick_params(axis='y', direction='out', length=0.1, pad=0.1, labelsize=4)   

            title_suf = " (x1e6)" if error_type == "plan" else ""
            ax.set_title(f'{error_plotnames[error_type]} Error{title_suf}')
            ax.axis('off')

    # preserve your original global subplot tuning
    plt.subplots_adjust(left=0.001, right=0.97, top=0.98, bottom=0.02,
                        hspace=0.01, wspace=0.09)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'error_maps_{idx}.png', dpi=600, bbox_inches='tight', pad_inches=0.01)    

    plt.savefig(save_dir / f'error_maps_{idx}.pdf', bbox_inches='tight', pad_inches=0.01, dpi=900)
    plt.close(fig)


def fuse_depth_maps(depth_data, error_maps, model_names):
    """Fuse depth maps using error-based weights"""
    weights = np.ones_like(depth_data)
    
    # Calculate weights based on normalized errors
    for i, model in enumerate(model_names):        
        model_weight = np.ones_like(depth_data[i])
        for error_type in ['grad', 'plan', 'icp', 'iqr']:
            error = error_maps[model][error_type]
            # Normalize error to [0,1] range and invert
            norm_error = 1 - (error - np.nanmin(error)) / (np.nanmax(error) - np.nanmin(error))
            model_weight *= norm_error
        weights[i] = model_weight
    
    # Normalize weights
    weights = weights / np.sum(weights, axis=0, keepdims=True)
    
    # Compute weighted average
    fused_depth = np.sum(depth_data * weights, axis=0)
    return fused_depth

def plot_fused_depth(fused_depth, save_dir, idx):
    """Plot fused depth map with trimmed turbo colormap"""
    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.01])
    
    # Calculate min/max from percentiles of valid data
    valid_data = fused_depth[~np.isnan(fused_depth)]
    vmin = np.percentile(valid_data, 5)
    vmax = np.percentile(valid_data, 95)
    
    # Create trimmed turbo colormap (removing 2% from each end)
    turbo = plt.cm.turbo
    colors = turbo(np.linspace(0, 1, 256))
    n_trim = int(256 * 0.02)  # 2% trim
    colors = colors[n_trim:-n_trim]
    trimmed_turbo = LinearSegmentedColormap.from_list('trimmed_turbo', colors)
    
    # Plot depth map
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(fused_depth, cmap=trimmed_turbo, vmin=vmin, vmax=vmax)
    ax.set_title(f'Fused Depth Map - Image {idx}')
    ax.axis('off')
    
    # Add colorbar on the right
    cbar_ax = fig.add_subplot(gs[1])
    plt.colorbar(im, cax=cbar_ax, label='Depth')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'fused_depth_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_error_distributions(error_aggr, save_dir):
    """
    Analyze error distributions across models and create CDF plots
    
    Args:
        error_aggr: Dictionary of error aggregates per model
        save_dir: Directory to save outputs
    """
    error_types = ['grad', 'plan', 'icp', 'iqr']
    percentiles = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100
    
    # Prepare data for CSV
    csv_data = []
    plan_scale = 1e6
    headers = [[f"# Planarity error has been scaled by {plan_scale}."],
               ['model', 'error_type'] + [f'p{p}' for p in percentiles]]
    
    # Process each model and error type
    for model_name, model_errors in error_aggr.items():        
        for error_type in error_types:
            # Flatten the 1000-length arrays across all images
            error_values = model_errors[error_type].flatten()
            if error_type == "plan":                
                error_values = error_values * plan_scale
            percentile_values = np.percentile(error_values, percentiles)
            
            # Add to CSV data
            row = [model_name, error_type] + [f'{v:.6f}' for v in percentile_values]
            csv_data.append(row)
    
    # Save CSV
    csv_path = save_dir / 'error_percentiles.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(headers)
        writer.writerows(csv_data)
    
    # Create CDF plots for each error type
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    for error_type in error_types:
        plt.figure(figsize=(10, 6))
        
        for model_name, model_errors in error_aggr.items():            
            # Get error values and sort them
            scale = plan_scale if error_type == 'plan' else 1
            error_values = model_errors[error_type].flatten()*scale            
            sorted_values = np.sort(error_values)
            
            # Calculate cumulative probabilities
            n = len(sorted_values)
            cumulative_prob = np.arange(1, n + 1) / n
            
            # Plot CDF
            plt.plot(sorted_values, cumulative_prob, label=model_name, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(f'{error_type.upper()} Error' + (' (×1e6)' if error_type == 'plan' else ''))
        plt.ylabel('Cumulative Probability')
        plt.title(f'Cumulative Distribution of {error_type.upper()} Error Across Models')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / f'error_cdf_{error_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main(datalist=None, specific_path=None):
    # Model lists
    MONO_MODELS = ['depthpro', 'metric3d', 'unidepth', 'depth_anything']
    STEREO_MODELS = ['monster', 'foundation', 'defom', 'selective']
    ALL_MODELS = MONO_MODELS + STEREO_MODELS
    
    if specific_path:
        # Use the specific path provided
        error_data_path = specific_path
        save_dir = error_data_path.parent
        with open(error_data_path, 'rb') as f:
            error_data = pickle.load(f)
        
        # Process each image index
        # Plot error maps
        for img_idx in range(error_data['error_maps']['MonSter']['grad'].shape[0]):
            plot_error_maps(error_data['error_maps'], ALL_MODELS, save_dir, img_idx)
        
        # Analyze error distributions
        analyze_error_distributions(error_data['error_aggr'], save_dir)
        
    else:
        pass

if __name__ == "__main__":
    main()
