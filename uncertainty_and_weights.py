import numpy as np

def calculate_individual_mad_uncertainty(depth_maps):
    """
    MAD: Median Absolute Deviation
    Uncertainty = absolute deviation from median of the rest of the ensemble.
    Calculates an individual uncertainty map for each depth map based on MAD.    
    """
    num_maps, _, _ = depth_maps.shape
    uncertainty_maps = np.zeros_like(depth_maps)

    for k in range(num_maps):        
        other_maps = np.concatenate((depth_maps[:k, :, :], depth_maps[k+1:, :, :]), axis=0)
        ensemble_median = np.median(other_maps, axis=0)
        uncertainty_maps[k, :, :] = np.abs(depth_maps[k, :, :] - ensemble_median)

    return uncertainty_maps

def get_iqr_uncertainty(depth_maps):
    """
    IQR: Interquartile Range
    kth percentile = the value below which k% of the data falls
    1st, 2nd, 3rd quartile = 25th, 50th, 75th percentile 
    IQR = 75th percentile - 25th percentile = middle 50% of the data
    Low IQR: high consensus among middle 50% of the data
    High IQR: low consensus among middle 50% of the data
    Uncertainty = absolute deviation from median, normalized by IQR
    """
    depth_names= list(depth_maps.keys())
    num_maps = len(depth_names)
    depth_maps = np.stack(list(depth_maps.values()), axis=0)
    uncertainty_maps = {}
    epsilon = 1e-4 # To prevent division by zero

    for k in range(num_maps):        
        other_maps = np.concatenate((depth_maps[:k, :, :], depth_maps[k+1:, :, :]), axis=0)
        ensemble_median = np.median(other_maps, axis=0)
        
        q1 = np.percentile(other_maps, 25, axis=0)
        q3 = np.percentile(other_maps, 75, axis=0)
        iqr = q3 - q1
        # Uncertainty = absolute deviation from the median, normalized by the IQR
        uncertainty_maps[depth_names[k]] = np.abs(depth_maps[k, :, :] - ensemble_median) / (iqr + epsilon)

    return uncertainty_maps

def simple_weighted_fusion(depth_maps, uncertainty_maps):
    """
    Performs a simple weighted mean fusion using the uncertainty maps as weights.
    Weights are inversely proportional to uncertainty.
    """
    # A small epsilon to avoid division by zero
    epsilon = 1e-4
    
    # Weights are the inverse of uncertainty. Higher uncertainty -> lower weight.
    weights = 1.0 / (uncertainty_maps + epsilon)

    # Perform a weighted average for each pixel
    numerator = np.sum(depth_maps * weights, axis=0)
    denominator = np.sum(weights, axis=0)
    
    fused_depth = numerator / denominator
    
    return fused_depth