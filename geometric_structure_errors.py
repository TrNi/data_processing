import numpy as np
from skimage.util import view_as_windows
import cv2
from scipy.ndimage import uniform_filter

def get_planarity_error(X_c, patch_size=7):
    """
    Computes a planarity error map for a given depth map using PCA.
    For each pixel in the image, take a small kxk patch and unproject it to 3D camera frame.
    PCA on these 3D points: 
    - principal components are the eigenvectors of the covariance matrix and give orientation of the best plane fitting the 3D points.
    - Lambda1 >= Lambda2 >= Lambda3.
    - The largest eigenvalue Lambda1 is the variance along the plane
    - The smallest eigenvalue Lambda3 is the variance orthogonal to the plane
    - The ratio of these two eigenvalues Lambda1/Lambda3 tells how co-planar the 3D points are.
    - Lambda3 is the planarity error.

    Args:
        X_c_left (np.ndarray): The 3D points in left camera frame (H, W, 3).
        patch_size (int): The size of the local patch (e.g., 5, 7).

    Returns:
        np.ndarray: The planarity error map (H, W).
    """
    # 1) compute per‐channel local means μ_c = box_filter(X_c[...,c])
    
    mu = np.stack([
        uniform_filter(X_c[..., c], size=patch_size, mode='nearest')
        for c in range(3)
    ], axis=-1)                    # shape (H, W, 3)

    # 2) compute box‐filtered second moments E[x_c * x_d] for c≤d
    E = np.zeros((X_c.shape[0], X_c.shape[1], 3, 3), dtype=np.float32)
    for c in range(3):
        for d in range(c, 3):
            prod = X_c[..., c] * X_c[..., d]
            Ecd = uniform_filter(prod, size=patch_size, mode='nearest')
            E[..., c, d] = Ecd
            E[..., d, c] = Ecd   # symmetry

    # 3) form covariance matrix per‐pixel: Cov = E - μ⊗μ
    #    Cov_{c,d} = E[x_c x_d] - μ_c μ_d
    mu_outer = mu[..., :, None] * mu[..., None, :]  # (H,W,3,3)
    Cov = E - mu_outer

    # 4) batched eigensolve: smallest eigenvalue at index 0
    #    np.linalg.eigvalsh works on stacked last‐two dims
    eigs = np.clip(np.linalg.eigvalsh(Cov), min=0 ) # shape (H, W, 3)
    lambda_min = eigs[..., 0]       # smallest eigenvalue

    return lambda_min.astype(np.float32)    

def compute_grad_error(depth_map, image, alpha=1, k=7):
    """
    Computes the gradient consistency error map based on the provided formula,
    using cv2.Sobel for gradient calculation.

    This function calculates a penalty for a depth map by penalizing large
    depth gradients in regions where the corresponding image is smooth.

    Args:
        depth_map (np.ndarray): A 2D NumPy array representing the depth map.
        image (np.ndarray): A 2D or 3D NumPy array representing the image.
                            If 3D (color), it will be converted to grayscale
                            for gradient calculation.
        alpha (float): A tuning parameter that controls the penalty strength.
        k (int): The size of the Sobel kernel, recommended larger for ful-frame, 
                 high-res images, to avoid pixel-level noisy edges.

    Returns:
        np.ndarray: A 2D NumPy array of the same shape as the depth map,
                    representing the gradient consistency error map.
    """    
    # 1. Convert image to grayscale if it's a color image
    I_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    I_gray = (I_gray - I_gray.min()) / (I_gray.max() - I_gray.min())

    # 2. Compute the gradients using cv2.Sobel for both depth and grayscale image
    # cv2.Sobel calculates either x or y gradient at a time    
    g_d_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=k)
    g_d_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=k)
    
    g_i_x = cv2.Sobel(I_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=k)
    g_i_y = cv2.Sobel(I_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=k)

    # 3. Calculate the L1 norms of the gradients
    # L1 norm of a gradient vector [dx, dy] is |dx| + |dy|
    g_d = np.abs(g_d_x) + np.abs(g_d_y)
    g_i = np.abs(g_i_x) + np.abs(g_i_y)    
    fixed_max_grad = 255
    g_i = g_i/fixed_max_grad

    fixed_min_grad = 1e-3
    g_d = g_d/fixed_min_grad

    # 4. Compute the final consistency error map element-wise
    error_map = g_d * np.exp(-alpha * g_i)
    print('max ∇I:', g_i.max())
    print('min exp(-alpha ∇I):', np.exp(-alpha * g_i).min())

    return error_map