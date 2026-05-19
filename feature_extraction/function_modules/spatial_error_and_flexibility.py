'''
Python file for computing Strict Spatial Error and Topological Flexibility features.
This module calculates the exact physical deviation between the digital clay and the target S-curve,
measuring absolute Euclidean distances, spatial bounding errors, and elastic path warping.
'''

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, directed_hausdorff
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

def extract_spatial_error_and_flexibility(base):
    '''
    Extracts absolute spatial deviations and topological elasticity metrics.
    Parameters:
        base (dict): Foundational 1D calculus arrays (x_v, x_s, y, err, abs_err, arc_len_v, arc_len_s).
    Returns:
        dict: A dictionary containing rigorously extracted spatial matching features.
    '''
    f = {}
    x_v, x_s, y = base['x_v'], base['x_s'], base['y']
    err, abs_err = base['err'], base['abs_err']
    k_v, k_s = base['k_v'], base['k_s']
    
    # -----------------------------------------
    # STRICT SPATIAL ERROR
    # -----------------------------------------
    
    # 1. Root Mean Square Error (RMSE)
    # The standard mathematical penalty for deviating from the S-curve
    f['rmse'] = np.sqrt(np.mean(err**2))
    
    # 2. Mean Absolute Error (MAE)
    # The average physical horizontal distance, highly robust against a single sharp outlier
    f['mae'] = np.mean(abs_err)
    
    # 3. Maximum Absolute Error (Chebyshev Norm)
    # Isolates the single most aggressive, extreme physical deviation between the vase and the ideal
    f['chebyshev_max_error'] = np.max(abs_err)
    
    # 4. Open Procrustes Distance (1D Proxy)
    # The remaining squared error after mathematically sliding the vase to optimally fit the S-curve
    shift_v = x_v - np.mean(x_v)
    shift_s = x_s - np.mean(x_s)
    f['open_procrustes_distance'] = np.sum((shift_v - shift_s)**2)
    
    # -----------------------------------------
    # PURE GEOMETRY & TOPOLOGY
    # -----------------------------------------
    
    # 5. Area Between Curves
    # The true physical 2D "gap" existing between the two overlaid lines using trapezoidal integration
    f['area_between_curves'] = np.trapz(abs_err, y)
    
    # 6. Curve Length Ratio
    # Measures how much extra meandering path was required to draw the vase compared to the clean S-curve
    f['curve_length_ratio'] = base['arc_len_v'] / base['arc_len_s'] if base['arc_len_s'] > 0 else 0
    
    # 7. Intersection Count
    # How many times the vase profile physically weaves back and forth across the target S-curve
    f['intersection_count'] = np.sum(np.diff(np.sign(err)) != 0)
    
    # -----------------------------------------
    # TOPOLOGICAL FLEXIBILITY
    # -----------------------------------------
    # Build strict 2D coordinate arrays for spatial distance algorithms
    coords_v = np.column_stack((x_v, y))
    coords_s = np.column_stack((x_s, y))
    
    # 8. Hausdorff Distance
    # Evaluates the maximum of the shortest distances between the two spatial paths
    h_v_s = directed_hausdorff(coords_v, coords_s)[0]
    h_s_v = directed_hausdorff(coords_s, coords_v)[0]
    f['hausdorff_distance'] = max(h_v_s, h_s_v)
    
    # 9. Spatial Dynamic Time Warping (DTW)
    # Calculates how closely the curves align when allowing for elastic "stretching" on the Y-axis,
    # ensuring a high belly can map to a slightly lower belly without catastrophic penalty.
    dtw_dist, _ = fastdtw(coords_v, coords_s, dist=euclidean)
    f['spatial_dtw_distance'] = dtw_dist
    
    # 10. Modified Hausdorff Distance (MHD)
    # Averages the shortest distances instead of taking the absolute maximum outlier
    dist_matrix = cdist(coords_v, coords_s)
    mhd_v_s = np.mean(np.min(dist_matrix, axis=1))
    mhd_s_v = np.mean(np.min(dist_matrix, axis=0))
    f['modified_hausdorff_distance'] = max(mhd_v_s, mhd_s_v)
    
    # 15. Longest Common Subsequence (LCSS)
    # How much of the vase stays within a strict epsilon error margin of the S-curve
    epsilon = 0.05 * (np.max(x_s) - np.min(x_s)) # 5% of target width as tolerance
    n, m = len(coords_v), len(coords_s)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if np.linalg.norm(coords_v[i-1] - coords_s[j-1]) < epsilon:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    f['longest_common_subsequence_ratio'] = dp[n][m] / max(n, m)
    
    # 16. Elastic Shape Distance (SRVF)
    # Computes the Square Root Velocity Function to measure pure bending/stretching energy
    dy = np.gradient(y)
    dx_v, dx_s = np.gradient(x_v), np.gradient(x_s)
    
    norm_v = np.sqrt(dx_v**2 + dy**2) + 1e-8
    norm_s = np.sqrt(dx_s**2 + dy**2) + 1e-8
    
    q_v_x, q_v_y = dx_v / np.sqrt(norm_v), dy / np.sqrt(norm_v)
    q_s_x, q_s_y = dx_s / np.sqrt(norm_s), dy / np.sqrt(norm_s)
    
    srvf_integrand = (q_v_x - q_s_x)**2 + (q_v_y - q_s_y)**2
    f['elastic_shape_distance_srvf'] = np.trapz(srvf_integrand, y)
    
    # 20. Earth Mover's Distance (Wasserstein)
    # The physical cost to mathematically shovel the X-distribution to match the S-curve
    f['earth_movers_distance'] = wasserstein_distance(x_v, x_s)
    
    # 21. Quantile-Quantile (Q-Q) Deviation
    # Strips away Y-coordinates entirely to compare sorted width severity
    f['qq_deviation'] = np.sum((np.sort(x_v) - np.sort(x_s))**2)
    
    # 22. Jensen-Shannon Divergence (Curvature)
    # Measures how fundamentally different the probability distributions of the bending are
    hist_v, bin_edges = np.histogram(k_v, bins=50, density=True)
    hist_s, _ = np.histogram(k_s, bins=bin_edges, density=True)
    
    # Add microscopic epsilon to prevent log(0) explosions
    hist_v += 1e-8
    hist_s += 1e-8
    
    js_metric = jensenshannon(hist_v, hist_s)
    f['jensen_shannon_divergence'] = js_metric**2  # JSD is the square of the JS metric
    
    # -----------------------------------------
    # GLOBAL AREA BIAS & PHASE ANGLES
    # -----------------------------------------
    
    # 11. Area Error Ratio
    # Ratio of the absolute error area to the total physical area of the target S-curve
    area_target = np.trapz(x_s, y)
    # Reuses the 'area_between_curves' integrated gap
    f['area_error_ratio'] = f['area_between_curves'] / area_target if area_target > 0 else 0.0

    # 14. Signed Area Bias
    # The true integral of the signed differences. 
    # Positive indicates the vase is globally fatter than the ideal; Negative indicates it is thinner.
    f['signed_area_bias'] = np.trapz(err, y)
    
    # 12. Max Curvature Spatial Deviation
    # The physical Euclidean distance between the sharpest bend on the vase and the sharpest bend on the target.
    idx_max_k_v = np.argmax(np.abs(k_v))
    idx_max_k_s = np.argmax(np.abs(k_s))
    
    pt_v = np.array([x_v[idx_max_k_v], y[idx_max_k_v]])
    pt_s = np.array([x_s[idx_max_k_s], y[idx_max_k_s]])
    f['max_curvature_spatial_deviation'] = np.linalg.norm(pt_v - pt_s)

    # 28. Contour Phase Angle
    # The angular difference between the global centroid vectors of the two shapes.
    # First, compute the strict 2D lamina centroids for both profiles.
    area_v = np.trapz(x_v, y)
    cx_v = np.trapz(0.5 * x_v**2, y) / area_v if area_v > 0 else 0
    cy_v = np.trapz(x_v * y, y) / area_v if area_v > 0 else np.min(y)
    
    cx_s = np.trapz(0.5 * x_s**2, y) / area_target if area_target > 0 else 0
    cy_s = np.trapz(x_s * y, y) / area_target if area_target > 0 else np.min(y)
    
    # Calculate the vector angles from the base origin
    theta_centroid_v = np.arctan2(cy_v - np.min(y), cx_v)
    theta_centroid_s = np.arctan2(cy_s - np.min(y), cx_s)
    
    # Enforce circular wrap-around for the absolute shortest phase difference
    phase_diff = np.abs(theta_centroid_v - theta_centroid_s)
    f['contour_phase_angle'] = min(phase_diff, 2 * np.pi - phase_diff) * (180 / np.pi) # Converted to degrees
    
    # 17. Discrete Fréchet Distance (The Dog-Walking Metric)
    # Enforces sequential topological order to find the minimum continuous leash length.
    # We build an iterative Dynamic Programming array to prevent maximum recursion depth crashes.
    
    def _calculate_discrete_frechet(P, Q):
        """
        Computes the discrete Fréchet distance between two 2D polygonal curves.
        P and Q must be (N, 2) arrays.
        """
        dist_matrix = cdist(P, Q, metric='euclidean')
        n, m = dist_matrix.shape
        
        # Initialise the dynamic programming cost array
        ca = np.zeros((n, m))
        ca[0, 0] = dist_matrix[0, 0]
        
        # Populate the base boundaries
        for i in range(1, n):
            ca[i, 0] = max(ca[i-1, 0], dist_matrix[i, 0])
        for j in range(1, m):
            ca[0, j] = max(ca[0, j-1], dist_matrix[0, j])
            
        # Iterate through the grid to find the optimal topological path
        for i in range(1, n):
            for j in range(1, m):
                min_previous = min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1])
                ca[i, j] = max(min_previous, dist_matrix[i, j])
                
        return ca[-1, -1]

    # Execute the DP matrix on the spatial coordinates
    f['frechet_distance'] = _calculate_discrete_frechet(coords_v, coords_s)
    
    return f