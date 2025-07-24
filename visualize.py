import matplotlib
# Set the backend to 'TkAgg' for better VSCode compatibility
matplotlib.use('TkAgg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import cv2
from depth_reproj_eval import *
import traceback
'''
visualize_depth_maps(
    base_path='/content/your_data_folder/',
    original_path='sample_data/left_rectified_{}.h5',
    depth_paths={
        'monster': 'depth_outputs/monster/leftview_disp_depth.h5',
        'foundation': 'depth_outputs/foundation_stereo/leftview_disp_depth.h5',
        'selective': 'depth_outputs/selective_stereo_igev/leftview_disp_depth.h5',
        'deform': 'depth_outputs/defom/leftview_disp_depth.h5'
    }
)

Key features of this implementation:

Efficient Memory Usage: Loads and processes one image at a time.
Interactive Visualization: Shows colorbars for depth maps and allows interactive zooming/panning.
Flexible Input: Handles different H5 file structures and image formats.
User Control:
Shows one image at a time by default
After 5 images, asks if you want to continue automatically
Press Enter to proceed to the next image
Error Handling: Properly closes H5 files even if an error occurs.
The script will display the original image alongside the four depth maps in a single row, 
with colorbars for each depth map. The visualization is interactive, 
allowing you to zoom and pan each subplot independently.
'''


def resize_image_hwc(img_hwc, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_hwc: H x W x C numpy array    
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)    
    return resized_hwc

def visualize_depth_maps(base_path='/content/', 
                        left_rectified_path='sample_data/left_rectified_{}.h5',                        
                        depth_paths={
                            'monster': 'depth_outputs/monster/leftview_disp_depth.h5',
                            'foundation': 'depth_outputs/foundation_stereo/leftview_disp_depth.h5',
                            'selective': 'depth_outputs/selective_stereo_igev/leftview_disp_depth.h5',
                            'defom': 'depth_outputs/defom/leftview_disp_depth.h5'
                        },
                        params_path='stereocal_params.npz',
                        chunk_size=1,
                        max_plots=5):
    """
    Visualize depth maps from multiple sources interactively.
    
    Args:
        base_path (str): Base path where all files are located
        left_rectified_path (str): Format string for original image paths
        depth_paths (dict): Dictionary mapping model names to their depth map file paths
        chunk_size (int): Number of images to process in each chunk
        max_plots (int): Number of plots to show before asking to continue
    """
    # Initialize variables
    current_idx = 0
    auto_continue = False
    model_names = list(depth_paths.keys())
    num_models = len(model_names)

    params = load_camera_params(os.path.join(base_path, params_path))
    Kleft, Kright = params['Kleft'], params['Kright']
    R, T = params['R'], params['T']
    P1, P2 = params['P1'], params['P2']

    right_rectified_path = os.path.join(base_path, left_rectified_path.replace("left", "right"))
    
    # Open all H5 files
    h5_files = {}
    try:       
        h5_files['left_rectified'] = h5py.File(os.path.join(base_path, left_rectified_path), 'r')        
        h5_files['right_rectified'] = h5py.File(os.path.join(base_path, right_rectified_path), 'r')        
        
        for name, path in depth_paths.items():
            h5_files[name] = h5py.File(os.path.join(base_path, path), 'r')
        
        # Get the minimum number of images across all files
        min_images = min([h5_files[f]['data'].shape[0] if 'data' in h5_files[f] else h5_files[f]['depth'].shape[0] 
                         for f in h5_files])    

        # Create the first figure with adjusted layout
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, num_models + 1, height_ratios=[1, 1], 
            hspace=0.2, wspace=0.2)
        axes = np.empty((2, num_models + 1), dtype=object)
        axes[0,0] = fig.add_subplot(gs[:, 0]) # Top-left for left image
        axes[1, 0] = fig.add_subplot(gs[1, 0])  # Bottom-left for right image

        for i in range(1, num_models + 1):
            axes[0, i] = fig.add_subplot(gs[0, i])
            axes[1, i] = fig.add_subplot(gs[1, i])

        

        plt.subplots_adjust(top=0.9,hspace=0.2,wspace=0.2)

        base_title = base_path 
        fig.suptitle(f'{base_title} : Image {current_idx + 1}', fontsize=10, y=0.98, color='black')
        
        while current_idx < min_images:
            rectified_left = h5_files['left_rectified']['data'][()][current_idx].transpose(1,2,0)
            rectified_right =h5_files['right_rectified']['data'][()][current_idx].transpose(1,2,0)
            print(f"rectified_left shape: {rectified_left.shape}")
            print(f"rectified_right shape: {rectified_right.shape}")

            # Combine left and right images vertically in the first subplot
            combined = np.vstack((rectified_left, 255*np.ones((200, rectified_left.shape[1], 3), dtype=np.uint8), rectified_right))
            axes[0, 0].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY), cmap='gray')
            axes[0, 0].set_title('Rectified Left (Top) & Right (Bottom) Images', fontsize=9)
            axes[0, 0].axis('off')
            if axes[1, 0] is not None:
                fig.delaxes(axes[1, 0])
            #axes = axes[:, 1:]
            # Plot depth maps
            depth_stats = {}
            for name in depth_paths.keys():
                depth_data = h5_files[name]['depth'][()][current_idx] if 'depth' in h5_files[name] else h5_files[name]['data'][()][current_idx]
                if len(depth_data.shape) == 3 and depth_data.shape[0] == 1:
                    depth_data = depth_data.squeeze(0)
                
                # Calculate statistics
                valid_data = depth_data[:,400:][~np.isnan(depth_data[:,400:])]
                depth_stats[name] = {
                    'min': np.clip(np.min(valid_data), 1e-6, 1000),
                    'max': np.clip(np.max(valid_data), 1e-6, 1000),
                    'num_nan': np.isnan(depth_data).sum(),
                    'pct_nan': np.mean(np.isnan(depth_data)) * 100
                }
            
            # Find global min and max for consistent color scaling (excluding NaNs)
            depth_mins = [stats['min'] for stats in depth_stats.values() if not np.isnan(stats['min'])]
            depth_maxs = [stats['max'] for stats in depth_stats.values() if not np.isnan(stats['max'])]
            
            if depth_mins and depth_maxs:  # Only if we have valid data
                vmin = max(min(depth_mins), 1e-6)
                vmax = max(depth_maxs)
            else:
                vmin, vmax = 1, 100  # Default values if no valid data
            
            # Use a perceptually uniform colormap with log scaling            
            turbo_r = plt.cm.get_cmap('turbo_r')
            # Create a new colormap that's the same as turbo_r but without the last color
            turbo_r_colors = turbo_r(np.linspace(0, 0.8, 256))  # Stop at 0.95 instead of 1.0 to avoid black
            custom_cmap = colors.ListedColormap(turbo_r_colors)
            n_levels = 50
            cmap = plt.get_cmap(custom_cmap, n_levels)
            
            for i, (name, path) in enumerate(depth_paths.items()):                                
                depth_data = h5_files[name]['depth'][()][current_idx] if 'depth' in h5_files[name] else h5_files[name]['data'][()][current_idx]
                rectified_left = resize_image_hwc(rectified_left, depth_data.shape[0], depth_data.shape[1])
                rectified_right = resize_image_hwc(rectified_right, depth_data.shape[0], depth_data.shape[1])
                X_c_left = px_to_camera(depth_data, Kleft)
                x_right_2d = project_to_view(X_c_left, P2)
                err1 = photometric_error_l1(rectified_left, rectified_right, x_right_2d)
                err2 = photometric_error_ssim(rectified_left, rectified_right)
                err_left = np.maximum(err1 + err2, 1e-6)[:,400:]
                norm_err = colors.LogNorm(vmin=err2.min(), vmax=err2.max())

                # Top row: depth maps
                ax_depth = axes[0, 1+i]
                im_depth = ax_depth.imshow(depth_data[:,400:], cmap=cmap, 
                                        norm=colors.LogNorm(vmin=vmin, vmax=vmax), 
                                        interpolation='nearest')
                ax_depth.set_title(name, fontsize=9)
                cbar_depth = plt.colorbar(im_depth, ax=ax_depth, fraction=0.035, pad=0.04)                
                ax_depth.axis('off')
                # Set ticks for depth colorbar
                extra_ticks = np.array([1.0, 1.5, 3.0])
                extra_ticks = extra_ticks[(extra_ticks >= vmin) & (extra_ticks <= vmax)]
                log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
                ticks = np.unique(np.concatenate([extra_ticks, log_ticks]))
                cbar_depth.set_ticks(ticks)
                cbar_depth.set_ticklabels([f"{tick:.1f}" for tick in ticks])

                # Bottom row: error maps
                ax_err = axes[1, 1+i]
                im_err = ax_err.imshow(err_left, cmap=cmap, 
                    norm=colors.LogNorm(vmin=0.1, vmax=err_left.max()), 
                    interpolation='nearest')
                ax_err.set_title(name, fontsize=9)
                cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.035, pad=0.04)                
                ax_err.axis('off')
                
                # Set ticks for error colorbar                

                log_ticks = np.logspace(np.log10(0.1), np.log10(err_left.max()), num=5)                
                cbar_err.set_ticks(log_ticks)
                cbar_err.set_ticklabels([f"{tick:.1f}" for tick in log_ticks])               
                
            
            plt.tight_layout()
            
            # Show the curre2t figure and wait for it to be 10losed
            plt.show(block=True)
            # plt.close('all')
            
            # Create a new figure for the next iteration if there are more images
            if current_idx < min_images - 1:
                fig = plt.figure(figsize=(14, 6))
                gs = fig.add_gridspec(2, num_models + 1, height_ratios=[1, 1], 
                hspace=0.2, wspace=0.2)
                axes = np.empty((2, num_models + 1), dtype=object)
                axes[0,0] = fig.add_subplot(gs[:, 0]) # Top-left for left image
                axes[1,0] = fig.add_subplot(gs[1, 0])  # Bottom-left for right image
                for i in range(1, num_models + 1):
                    axes[0, i] = fig.add_subplot(gs[0, i])
                    axes[1, i] = fig.add_subplot(gs[1, i])

                fig.delaxes(axes[1,0])
                #axes = axes[:, 1:]
                
                plt.subplots_adjust(top=0.95, hspace=0.2, wspace=0.2)
                # Update title for new figure
                fig.suptitle(f'{base_title} : Image {current_idx + 2}', fontsize=10, y=0.98, color='black')
            
            current_idx += 1
            
    except Exception as e:
        print("Error occurred:")
        print(traceback.format_exc())
        print(f"Error details: {str(e)}")
    finally:
        # Close all H5 files
        for f in h5_files.values():
            try:
                f.close()
            except:
                pass

#left_ple usage:

if __name__ == '__main__':
    # base_path = 'I:\\My Drive\\Scene-5\\f-28.0mm\\a-1.4mm\\stereodepth'
    # rectified_path = 'I:\\My Drive\\Scene-5\\f-28.0mm\\a-1.4mm\\stereocal_results_f28.0mm_a1.4mm\\rectified\\rectified_lefts.h5'
    base_path = 'I:\\My Drive\\Scene-6\\stereocal_results_f28mm_a22mm'
    left_rectified_path = 'I:\\My Drive\\Scene-6\\stereocal_results_f28mm_a22mm\\rectified_h5\\rectified_lefts.h5'
    depth_paths = {
        #'traditional': 'raw_depth_h5\\raw_depth_lefts.h5',
        'monster': 'stereodepth\\leftview_disp_depth_monster.h5',        
        'selective': 'stereodepth\\leftview_disp_depth_selective_igev.h5',
        'defom': 'stereodepth\\leftview_disp_depth_defom.h5'
    }
    params_path = 'stereocal_params.npz'
    visualize_depth_maps(base_path, left_rectified_path, depth_paths, params_path)
    