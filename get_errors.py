import matplotlib
# Set the backend to 'TkAgg' for better VSCode compatibility
matplotlib.use('TkAgg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import matplotlib.colors as colors
import cv2
from depth_reproj_eval import *
import traceback
from scipy.ndimage import median_filter
from uncertainty_and_weights import get_iqr_uncertainty
from collections import defaultdict
from point_cloud_opt import get_point_cloud_errors
from geometric_structure_errors import compute_grad_error, get_planarity_error, compute_grad
from pathlib import Path
import pickle

RECT_LEFT_KEY = 'rectified_lefts'
RECT_RIGHT_KEY = 'rectified_rights'
DEPTH_KEY = 'depth'

# get_cmap=colormaps.get_cmap
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

def find_h5_by_keywords(folder: Path, keywords):
    """Return dict keyword->Path of best-matching file in folder (case-insensitive)."""
    found = {}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ('.h5', '.hdf5')]
    lower_names = {f: f.name.lower() for f in files}
    for kw in keywords:
        candidate = None
        for f, lname in lower_names.items():
            if kw.lower() in lname:
                # prefer exact 'leftview' naming if present (not required, but helpful)
                candidate = f
                break
        if candidate is not None:
            found[kw] = candidate
    return found

def load_h5_dataset(h5path: Path, key: str):
    """Load dataset key from h5 file. Return numpy array or raise."""
    with h5py.File(h5path, 'r') as fh:
        if key not in fh:
            raise KeyError(f"Key '{key}' not found in {h5path}")
        data = fh[key][()]
    return data

def get_stats(arr, maxval=1000):
    return {
        'min': np.clip(np.nanmin(arr), 1e-6, maxval),
        'max': np.clip(np.nanmax(arr), 1e-6, maxval),
        '5': np.percentile(arr, 5),
        '95': np.percentile(arr, 95),
        'num_nan': np.isnan(arr).sum(),
        'pct_nan': np.mean(np.isnan(arr)) * 100
    }

def resize_image_hwc(img_hwc, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_hwc: H x W x C numpy array    
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)    
    return resized_hwc

def resize_batch_nhwc(batch_nhwc, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    return np.stack([resize_image_hwc(img, target_h, target_w, interpolation) for img in batch_nhwc])

def resize_image_chw(img_chw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_chw: C x H x W numpy array    
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)
    resized_chw = np.transpose(resized_hwc, (2, 0, 1))    
    return resized_chw

def resize_batch_nchw(batch_nchw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    return np.stack([resize_image_chw(img, target_h, target_w, interpolation) for img in batch_nchw])

def sorted_k(arr, k=10000):
    arr = arr.flatten()    
    p5, p95 = np.percentile(arr, [5, 95])
    arr = arr[(arr >= p5) & (arr <= p95)]
    x_sorted = np.sort(arr)    
    idx = np.linspace(0, len(x_sorted) - 1, k).astype(int)    
    return x_sorted[idx]

class Get_errors_and_GT:
    """
    Compute and save errors and GT.
    
    GT is calculated as a fused depth map = weighted average of available depth maps, 
       with weights calculated based on error components.
    
    Args:
        datalist (list): List of dictionaries containing scene information
        MONO_MODELS (list): List of monocular depth models to use
        STEREO_MODELS (list): List of stereo depth models to use
    """
    def __init__(self, datalist, MONO_MODELS, STEREO_MODELS):
        self.datalist = datalist
        self.MONO_MODELS = MONO_MODELS
        self.STEREO_MODELS = STEREO_MODELS
    
    def load_rects(self, base, left_cam, right_cam, cfg):
        fl = cfg['fl']
        F = float(cfg['F'])
        fl_folder = f"fl_{int(fl)}mm"
        F_folder = f"F{F:.1f}"
        # rectified dir paths
        self.left_rectified_dir = base / left_cam / fl_folder / "inference" / F_folder / "rectified"
        self.right_rectified_dir = base / right_cam / fl_folder / "inference" / F_folder / "rectified"
        self.stereocal_params = base / f'stereocal_params_{fl}mm.npz'
        print(f"Processing config: base={base.name} fl={fl} F={F:.1f} left={left_cam} right={right_cam}")

        # rectified.h5 paths
        self.left_rect_h5 = self.left_rectified_dir / "rectified_lefts.h5"
        self.right_rect_h5 = self.right_rectified_dir / "rectified_rights.h5"

        if not self.left_rect_h5.exists():
            raise FileNotFoundError(f"{self.left_rect_h5} not found")
        if not self.right_rect_h5.exists():
            raise FileNotFoundError(f"{self.right_rect_h5} not found")

        # load rectified arrays
        left_rects = load_h5_dataset(self.left_rect_h5, RECT_LEFT_KEY)   # NxCxHxW
        right_rects = load_h5_dataset(self.right_rect_h5, RECT_RIGHT_KEY) # NxCxHxW

        # canonical shapes
        self.left_rects = np.asarray(left_rects)
        self.right_rects = np.asarray(right_rects)
        if self.left_rects.shape[0] != self.right_rects.shape[0]:
            print("Left and right have different N; using min(N)")
        self.N = min(self.left_rects.shape[0], self.right_rects.shape[0])
    
    def load_depths(self):
        # find depth files (search in left rectified dir)
        mono_found = find_h5_by_keywords(self.left_rectified_dir.parent / "monodepth", self.MONO_MODELS)
        stereo_found = find_h5_by_keywords(self.left_rectified_dir, self.STEREO_MODELS)
        
        # attempt to map keywords in desired order
        mono_depth_paths = []
        stereo_depth_paths = []
        mono_depth_titles = []
        stereo_depth_titles = []
        for kw in self.MONO_MODELS:
            p = mono_found.get(kw)
            if p is not None:
                mono_depth_paths.append(p)
                mono_depth_titles.append(get_pretty_name(p.name))
            else:
                mono_depth_paths.append(None)
                mono_depth_titles.append(kw)  # placeholder

        for kw in self.STEREO_MODELS:
            p = stereo_found.get(kw)
            if p is not None:
                stereo_depth_paths.append(p)
                stereo_depth_titles.append(get_pretty_name(p.name))
            else:
                stereo_depth_paths.append(None)
                stereo_depth_titles.append(kw)  # placeholder
        
        depth_titles = mono_depth_titles + stereo_depth_titles
        depth_paths = mono_depth_paths + stereo_depth_paths
        # load depth arrays for found; otherwise None
        depth_arrays = []
        for p in depth_paths:
            if p is None:
                depth_arrays.append(None)
                continue
            try:
                d = load_h5_dataset(p, DEPTH_KEY)  # expected NxHxW
                d = np.asarray(d).astype(np.float32)# .transpose(1,2,0)
                if d.shape[0] != self.N:
                    # truncate or pad if mismatch
                    if d.shape[0] > self.N:
                        d = d[:self.N]
                    else:
                        # pad with nan
                        # pad = np.full((N - d.shape[0],) + d.shape[1:], np.nan, dtype=d.dtype)
                        # d = np.concatenate([d, pad], axis=0)
                        self.N = d.shape[0]
                print(f"Loaded depth from {p}: {d.shape}")
                depth_arrays.append(d)
            except Exception as e:
                print(f"Failed to load depth from {p}: {e}")
                depth_arrays.append(None)
        
        self.depth_arrays = depth_arrays
        self.depth_titles = depth_titles
        self.depth_paths = depth_paths

    def save_errors(self):
        
        for entry in self.datalist:
            base = Path(entry['base'])
            left_cam = entry['cameras'][0]
            right_cam = entry['cameras'][1]

            for cfg in entry['configs']:
                self.load_rects(base, left_cam, right_cam, cfg)            
                self.load_depths()
                params = load_camera_params(self.stereocal_params)
                P1, P2 = params['P1'], params['P2']
                K, K_inv = params['K_new'], params['K_inv']          
                
                # Get the minimum number of images across all files
                min_images = min([d.shape[0] for d in self.depth_arrays])           

                out_dir = self.left_rectified_dir.parent / "err_GT" 
                os.makedirs(out_dir, exist_ok=True) 
                
                # Initialize variables            
                num_models = len(self.depth_titles)
                col_clip = 150 
                N,C,H,W = self.left_rects.shape
                aspect_ratio = W/H
                min_N, min_h, min_w = N, H, W

                for i,(name,arr) in enumerate(zip(self.depth_titles, self.depth_arrays)):
                        N_d, h_d, w_d = arr.shape
                        if abs(w_d/h_d - aspect_ratio) > 0.01:
                            print(f"Warning: not using {name} depth map: aspect ratio mismatch: {w_d/h_d} vs {aspect_ratio}")
                            continue                
                        min_N = min(min_N, N_d)
                        min_h = min(min_h, h_d)
                        min_w = min(min_w, w_d)
                K_inv_uv1 = get_Kinv_uv1(K_inv, min_h, min_w)
                print(f"Common resizing factor: {min_h/H:0.2f}x{min_w/W:0.2f}, reduced size: {min_h}x{min_w} px")
                self.left_rects = resize_batch_nchw(self.left_rects, min_h, min_w)
                self.right_rects = resize_batch_nchw(self.right_rects, min_h, min_w)                   
                self.error_maps = {}
                self.error_aggr = {}
                error_types = ['grad', 'plan', 'icp', 'iqr']
                aggr_points = 20000

                for i,(name,arr) in enumerate(zip(self.depth_titles, self.depth_arrays)):
                        N_d, h_d, w_d = arr.shape
                        if abs(w_d/h_d - aspect_ratio) > 0.01:
                            print(f"Warning: not using {name} depth map: aspect ratio mismatch: {w_d/h_d} vs {aspect_ratio}")
                            continue
                        self.depth_arrays[i] = resize_batch_nhwc(arr, min_h, min_w)
                        self.error_maps[name] = {k:np.zeros((min_N, min_h, min_w)) for k in error_types}
                        self.error_aggr[name] = {k:np.zeros((min_N, aggr_points)) for k in error_types}

                self.saved_depths = []    
                for current_idx in range(min_N): #[0,12,26]:#[0,1,2,11,12,13,24,25,26]: #range(min_N):
                    rectified_left = self.left_rects[current_idx].transpose(1,2,0)
                    rectified_right =self.right_rects[current_idx].transpose(1,2,0)                
                    depth_data_arr = np.stack([self.depth_arrays[i][current_idx] for i in range(num_models)], axis=0)

                    alpha=0.1
                    kernel=5
                    g_i = compute_grad(rectified_left, k=kernel)
                    g_i /= g_i.max()
                    
                    try:
                        iqr_errors = get_iqr_uncertainty(depth_data_arr)
                        icp_errors = get_point_cloud_errors(depth_data_arr, K_inv)
                    except Exception as e:
                        print(f"Error in get_iqr_uncertainty or get_point_cloud_errors: {e}")
                        continue

                    for i,name in enumerate(self.depth_titles):                    
                        err_data = get_errors(self.depth_arrays[i][current_idx], rectified_left, rectified_right, K_inv, K_inv_uv1, g_i,P2, alpha, kernel) 
                        for k in ["grad", "plan"]:
                            self.error_maps[name][k][current_idx] = err_data[k]
                            self.error_aggr[name][k][current_idx] = sorted_k(err_data[k], k=aggr_points)
                        
                        self.error_maps[name]["icp"][current_idx] = icp_errors[i]
                        self.error_aggr[name]["icp"][current_idx] = sorted_k(icp_errors[i], k=aggr_points)
                        self.error_maps[name]["iqr"][current_idx] = iqr_errors[i]
                        self.error_aggr[name]["iqr"][current_idx] = sorted_k(iqr_errors[i], k=aggr_points)
                    
                    self.saved_depths.append(depth_data_arr)
                    
                # Save error maps and aggregated errors for this configuration
                save_path = out_dir / f"error_data.pkl"
                error_data = {
                    'error_maps': self.error_maps,
                    'error_aggr': self.error_aggr,                    
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(error_data, f)
                print(f"Saved error data to {save_path}")
                
                # depth_path = out_dir / f"depth_data.pkl"
                # with open(depth_path, 'wb') as f:
                #     pickle.dump(self.saved_depths, f)
                # print(f"Saved depth data to {depth_path}")

                pass
                    # error_min, error_max = defaultdict(lambda: np.inf), defaultdict(lambda: -np.inf)    
                    # for j,name in enumerate(err_data):
                    #     if j==0:
                    #         all_err = err_data[name]  
                    #     for err_type in err_data[name]:
                    #         error_min[err_type] = min(error_min[err_type],np.percentile(err_data[name][err_type][:,col_clip:], 1))
                    #         error_max[err_type] = max(error_max[err_type],np.percentile(err_data[name][err_type][:,col_clip:], 99))                                  

                    # for j,name in enumerate(err_data):
                    #     err_data[name] = 0.3* (err_data[name]['grad_error'] + 1e-8 - error_min['grad_error'])/ (error_max['grad_error'] - error_min['grad_error'])+ \
                    #                     0.1* (err_data[name]['planarity_error'] + 1e-8 - error_min['planarity_error'])/ (error_max['planarity_error'] - error_min['planarity_error'])+ \
                    #                     5.5* (err_data[name]['iqr'] + 1e-8 - error_min['iqr'])/ (error_max['iqr'] - error_min['iqr'])+ \
                    #                     4.1* (err_data[name]['icp_error'] + 1e-8 - error_min['icp_error'])/ (error_max['icp_error'] - error_min['icp_error'])
                                        
                    #     err_data[name] = np.maximum(err_data[name],0) / 10

                    #     err_data[name] = err_data[name][:,col_clip:]
                    #     err_stats[name] = get_stats(err_data[name], maxval=500)
                    #     depth_stats[name] = get_stats(depth_data[name][:,col_clip:], maxval=100)
                    
                    # fused_depth = np.zeros((min_h, min_w - col_clip))
                    # weights = np.zeros((min_h, min_w - col_clip))
                    # weight_map = {}
                    # for name in depth_names:
                    #     weights += 1 / (err_data[name] + 1e-8)
                    #     fused_depth += depth_data[name][:,col_clip:] * 1 / (err_data[name] + 1e-8)
                    #     weight_map[name] = 1 / (err_data[name] + 1e-8)
                    # fused_depth /= weights            
                    
                    # for i, (name, path) in enumerate(depth_paths.items()):                                                                
                        
                    #     if i==0:
                    #         err_types = ["grad_error", "planarity_error", "iqr", "icp_error"] #list(all_err.keys())
                    #         for j in range(0,len(err_types)):
                            
                    #             this_err = (all_err[err_types[j]][:,col_clip:] + 1e-8 - error_min[err_types[j]])/ (error_max[err_types[j]] - error_min[err_types[j]])
                    #             lo = np.percentile(this_err, 1)
                    #             hi = np.percentile(this_err, 99)
                    #             print(err_types[j], lo,hi) #, np.percentile(this_err, 0.1), np.percentile(this_err, 99))
                    #             this_err = np.clip(this_err, lo, hi)                            
                    #             log_ticks = np.logspace(np.log10(this_err.min()), np.log10(this_err.max()), num=7)
                    #             print(err_types[j], [f"{tick:.3f}" for tick in log_ticks])                                               
                            
                    #         this_err = weight_map[name]/weights#err_data[name]
                        
                    # # Top row: fused depth map
                    # im_depth = fused_depth.round(3)
                    # depth_stats['fused'] = {
                    #     'min': fused_depth.min(),
                    #     'max': fused_depth.max(),
                    #     '5': np.percentile(fused_depth, 5),
                    #     '95': np.percentile(fused_depth, 95),
                    #     'num_nan': np.sum(np.isnan(fused_depth)),
                    #     'pct_nan': np.sum(np.isnan(fused_depth)) / fused_depth.size * 100
                    # }

                    # with h5py.File(os.path.join(base_path, 'ml_data', f"img_{current_idx}.h5"), 'w') as f:
                        
                    #     for t,name in enumerate(depth_names):
                    #         if t==0:
                    #             subarr = []
                    #             subarr.append(rectified_left)
                    #             subarr.append(rectified_right)
                    #             subarr.append(depth_data[name][...,None])
                    #             for k,v in all_err.items():
                    #                 subarr.append(v[...,None])
                    #                 print(k)
                    #             subarr = np.concatenate(subarr, axis=2)                                
                    #             f.create_dataset('data', data=subarr, compression='gzip', compression_opts=9)
                    #             f.create_dataset('total_weight', data=weight_map[name]/weights, compression='gzip', compression_opts=9)
                    #             f.create_dataset('fused_depth', data=fused_depth, compression='gzip', compression_opts=9)
                    
                    # current_idx += 1
                


if __name__ == '__main__':
    # ---------- USER DATASOURCE (as provided) ----------
    datalist = [  
    {
        "base": r"I:\\My Drive\\Pubdata\\Scene6_illusions",
        "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
        "configs":[            
            {"fl":70, "F":16}, 
            ]
    },
    # {
    #     "base": r"I:\\My Drive\\Pubdata\\Scene6",
    #     "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
    #     "configs":[            
    #         # {"fl":28, "F":22}, 
    #         {"fl":70, "F":2.8},
    #         ]
    # },
    # {
    #     "base": r"I:\\My Drive\\Pubdata\\Scene7",
    #     "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
    #     "configs":[
    #         # {"fl":28, "F":22}, 
    #         {"fl":70, "F":2.8}, 
    #         ]
    # },
    # {
    #     "base": r"I:\\My Drive\\Pubdata\\Scene8",
    #     "cameras": ['EOS6D_A_Left', 'EOS6D_B_Right'],
    #     "configs":[
    #         # {"fl":28, "F":22}, 
    #         {"fl":70, "F":2.8}, 
    #         ]
    # },   
    {
        "base": r"I:\\My Drive\\Pubdata\\Scene9",
        "cameras": ['EOS6D_B_Left', 'EOS6D_A_Right'],
        "configs":[
            # {"fl":28, "F":22}, 
            {"fl":70, "F":2.8}, 
            ]
    },
    ]
    
    MONO_MODELS = ['depthpro', 'metric3d', 'unidepth', 'depth_anything']
    STEREO_MODELS = ['monster', 'foundation', 'defom', 'selective']
    
    GEGT = Get_errors_and_GT(datalist, MONO_MODELS, STEREO_MODELS)
    GEGT.save_errors()
    