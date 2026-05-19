'''
Python file for computing Spatial Y-Binning relational features.
This module executes Piecewise Aggregate Approximation (PAA) by slicing the Y-axis 
into strict vertical floors to isolate localised topological errors and structural flow.
'''

import numpy as np

def extract_spatial_y_binning(base, n_bins=5):
    '''
    Extracts vertically binned spatial features and strict quartiles.
    Parameters:
        base (dict): Foundational 1D calculus arrays (y, x_v, err, abs_err, k_v, theta_v, theta_s).
        n_bins (int): The number of vertical floors to divide the Y-axis into (default quintiles).
    Returns:
        dict: A dictionary containing the rigorously extracted PAA spatial features.
    '''
    f = {}
    y = base['y']
    x_v = base['x_v']
    err = base['err']
    abs_err = base['abs_err']
    k_v = base['k_v']
    
    # Safely compute Turning Angle Error with strict circular wrap-around
    theta_diff = np.abs(base['theta_v'] - base['theta_s'])
    theta_err = np.minimum(theta_diff, 2 * np.pi - theta_diff)
    
    y_min, y_max = np.min(y), np.max(y)
    y_total = y_max - y_min
    
    # -----------------------------------------
    # GLOBAL QUARTILES (Metrics 17 & 18)
    # -----------------------------------------
    # Isolate the strict upper and lower 25% of the clay to evaluate the lip and base independently
    upper_q_mask = y > (y_min + 0.75 * y_total)
    lower_q_mask = y < (y_min + 0.25 * y_total)
    
    f['binned_mae_upper_quartile'] = np.mean(abs_err[upper_q_mask]) if np.any(upper_q_mask) else 0.0
    f['binned_mae_lower_quartile'] = np.mean(abs_err[lower_q_mask]) if np.any(lower_q_mask) else 0.0
    
    # -----------------------------------------
    # PIECEWISE AGGREGATE APPROXIMATION (PAA)
    # -----------------------------------------
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    
    max_divergence_val = -1
    max_divergence_bin = -1
    
    for i in range(n_bins):
        # Mask for the current vertical floor
        # The upper bound is inclusive only for the final bin to prevent dropping the absolute lip coordinate
        if i == n_bins - 1:
            mask = (y >= bin_edges[i]) & (y <= bin_edges[i+1])
        else:
            mask = (y >= bin_edges[i]) & (y < bin_edges[i+1])
            
        prefix = f'y_bin_{i+1}_'
        
        # Handle mathematically empty bins (should not occur in a continuous 250-point interpolation, but safe to guard)
        if not np.any(mask):
            f[prefix + 'paa_mean_width'] = 0.0
            f[prefix + 'paa_mae'] = 0.0
            f[prefix + 'paa_signed_error'] = 0.0
            f[prefix + 'mean_curvature'] = 0.0
            f[prefix + 'turning_angle_error'] = 0.0
            continue
            
        # 23. PAA Mean Width
        f[prefix + 'paa_mean_width'] = np.mean(x_v[mask])
        
        # 24. PAA Mean Absolute Error
        f[prefix + 'paa_mae'] = np.mean(abs_err[mask])
        
        # 25. PAA Signed Error Bias
        # Positive means the vase bulges wider than the target; Negative means it pinches narrower
        f[prefix + 'paa_signed_error'] = np.mean(err[mask])
        
        # 26. Binned Curvature Mean
        f[prefix + 'mean_curvature'] = np.mean(k_v[mask])
        
        # 27. Binned Turning Angle Error
        f[prefix + 'turning_angle_error'] = np.mean(theta_err[mask])
        
        # 19. Point of Maximum Divergence Tracking
        bin_max_err = np.max(abs_err[mask])
        if bin_max_err > max_divergence_val:
            max_divergence_val = bin_max_err
            max_divergence_bin = i + 1
            
    # Record the specific vertical zone where the geometry suffered the most catastrophic collapse
    f['point_of_max_divergence_bin_id'] = max_divergence_bin
    
    return f