from math import e
import os
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim

def load_h5_images(h5_path):
    """Load images from .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        images = f['data'][()] #[:]  # Shape: (24, 3, 2280, 3420)

    if images.ndim == 4:
        return images.transpose(0, 2, 3, 1)
    else:
        return images


def load_camera_params(npz_path):
    """Load stereo camera parameters from .npz file."""
    data = np.load(npz_path)
    return {
        'Kleft': data['Kleft'], 'Kright': data['Kright'],
        'R': data['R'], 'T': data['T'],
        'P1': data['P1'], 'P2': data['P2'],
        'baseline': data['baseline'], 'fB': data['fB']
    }

def compute_grad_error(I_left, eps=1e-4):
    I_gray = cv2.cvtColor(I_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # sobel gradients
    gx = cv2.Sobel(I_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I_gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.abs(gx) + np.abs(gy)
    E_tex = 1.0 / (G + eps)
    return E_tex #texture error



def px_to_camera(depth_map, K, uv1=None):
    """
    Convert pixel coordinates to 3D camera coordinates for the same view,
    scaled to match depth.
    """
    if uv1 is None:
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (H*W, 3)

    # Camera coordinates: X_c = Z_c * K^-1 * [u,v,1]
    Z_c = depth_map.ravel()
    K_inv = np.linalg.inv(K)
    X_c = Z_c[:,None] * (K_inv @ uv1.T).T  # (H*W, 3)
    return X_c

def project_to_view(X_one, P_two):
    """Reproject first camera frame 3D points to second camera 2D image view."""
    length = X_one.shape[0]
    X_one_hom = np.hstack([X_one, np.ones((length, 1))])  # (H*W, 4)
    x_two = (P_two @ X_one_hom.T).T  # (H*W, 3)
    u_two = x_two[:, 0] / x_two[:, 2]
    v_two = x_two[:, 1] / x_two[:, 2]
    return np.stack([u_two, v_two], axis=-1).reshape(length, 2)  # (H*W, 2)


def depth_projection_errors(D_left, x_right, Kright, P_one, error_types=['px', 'depth']):
    H, W = D_left.shape

    # Split u_R and v_R
    u_r = x_right[..., 0].astype(np.float32).reshape(H, W)
    v_r = x_right[..., 1].astype(np.float32).reshape(H, W)    

    # Warp left-depth map in the same left view but at right image pixel location
    D_left_prime_uv = cv2.remap(D_left.astype(np.float32), u_r, v_r, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ur_vr_1 = np.stack([u_r, v_r, np.ones_like(u_r)], axis=-1).reshape(-1, 3)

    X_c_right_prime = px_to_camera(D_left_prime_uv, Kright, ur_vr_1)

    x_left_2d_prime = project_to_view(X_c_right_prime, P_one)

    u_l, v_l = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x_left = np.stack([u_l, v_l], axis=-1).reshape(-1, 2)
    
    errors = []
    if 'px' in error_types:
        errors.append(np.abs(x_left_2d_prime - x_left).sum(-1).astype(np.float32).reshape(H, W))
    if 'depth' in error_types:
        errors.append(np.abs(D_left - D_left_prime_uv))

    total_error = np.sum(errors, axis=0)
    return np.nan_to_num(total_error, nan=500, posinf=500,neginf=500)


def photometric_errors(I_L, I_R, x_right, error_types=['l1', 'l2', 'ssim']):
    """
    Computes L1 and L2 error maps between I_L and I_R(x_right), per pixel.

    I_L: HxWx3 left image
    I_R: HxWx3 right image
    x_right: HxWx2 of (u_R, v_R) reprojected coordinates in right image
    """
    H, W, _ = I_L.shape

    # Split u_R and v_R
    u_R = x_right[..., 0].astype(np.float32).reshape(H, W)
    v_R = x_right[..., 1].astype(np.float32).reshape(H, W)

    valid = (
        (u_R >= 0) & (u_R < W) &
        (v_R >= 0) & (v_R < H)
    )

    # Warp each channel using remap
    I_R_warped = cv2.remap(I_R, u_R, v_R, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    errors = []
    if 'l1' in error_types:
        errors.append(np.mean(np.abs(I_L - I_R_warped), axis=-1))  # H x W scalar map
    if 'l2' in error_types:
        errors.append(np.mean((I_L - I_R_warped)**2, axis=-1))  # H x W scalar map
    if 'ssim' in error_types:        
        #ssim_err[~valid] = 5 # does not make a difference.
        errors.append(photometric_error_ssim(I_L, I_R_warped))  # H x W scalar map

    total_error = np.sum(errors, axis=0)
    return total_error

def photometric_error_ssim(I_L, I_R_warped):
    """
    Returns 1 - SSIM index per pixel (approximate perceptual error)
    """
    # Convert to grayscale
    I_L_gray = cv2.cvtColor(I_L, cv2.COLOR_BGR2GRAY)
    I_R_gray = cv2.cvtColor(I_R_warped, cv2.COLOR_BGR2GRAY)

    ssim_map = ssim(I_L_gray, I_R_gray, data_range=255, full=True)[1]  # Returns (mean SSIM score, full map)
    error_map = 1.0 - ssim_map  # dissimilarity

    return error_map


if __name__ == "__main__":
    # Load parameters
    rootdir = "I:\\My Drive\\Scene-6\\stereocal_results_f28mm_a22mm"
    params_path = os.path.join(rootdir, 'stereocal_params.npz')
    depth_map_path = os.path.join(rootdir, "raw_depth_h5", "raw_depth_lefts.h5")
    pattern_info_path = os.path.join(rootdir, 'pattern_info.json')
    rectified_left_path = os.path.join(rootdir, "rectified_h5", "rectified_lefts.h5")
    rectified_right_path = rectified_left_path.replace("left", "right")

    image_index = 14
    params = load_camera_params(params_path)
    Kleft, Kright = params['Kleft'], params['Kright']
    R, T = params['R'], params['T']
    P1, P2 = params['P1'], params['P2']
    
    depth_map = load_h5_images(depth_map_path)[image_index]
    rectified_left = load_h5_images(rectified_left_path)[image_index]
    rectified_right = load_h5_images(rectified_right_path)[image_index]

    print(f"\n image {image_index}: " + \
            f"rectified_left: {rectified_left.shape}, " + \
            f"rectified_right: {rectified_right.shape}, " + \
            f"depth_map: {depth_map.shape}")
    
    # Comvert left image regular pixel meshgrid to left camera 3D points,
    # scaled to match depth.
    X_c_left = px_to_camera(depth_map, Kleft)
    
    # Project left camera 3D points to right camera 2D image view
    x_right_2d = project_to_view(X_c_left, P2)

    # Compute photometric errors between I_L_{(u,v):regular pixel meshgrid}
    # and I_R_{(u',v'):coordinates of right image obtained by projecting left image (u,v)}
    err1 = photometric_error_l1(rectified_left, rectified_right, x_right_2d)
    err2 = photometric_error_ssim(rectified_left, rectified_right)

    err_left = err1 + err2    

    mean_error = np.nanmean(err_left)
    print(f"Mean photometric error: {mean_error:.2f} over all pixels.")