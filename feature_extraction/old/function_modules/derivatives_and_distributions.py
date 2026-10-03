'''
Python file for computing Derivative Alignment relational features.
This module executes strict kinematic comparisons between the vase and the target S-curve,
bypassing raw spatial coordinates to evaluate the pure structural flow (slopes, angles, and bending). This includes:
- Turning Angle Distance
    - Turning Angle Distance is calculated by integrating the absolute circular difference in tangent angles
      between the vase and S-curve along the Y-axis using the trapezoidal rule. It evaluates pure trajectory
      deviation independent of spatial location, penalising mismatched directional flow at each height.
- Derivative Dtw Distance
    - Derivative Dtw Distance applies Dynamic Time Warping to the first derivatives (slopes) of both profiles,
      allowing elastic alignment along the height axis. It prevents false matches between structurally
      different regions such as flat walls and curved bellies by comparing rate-of-change rather than position.
- Max Curvature Cross Correlation
    - Max Curvature Cross Correlation slides the standardised curvature sequence of the vase across the
      standardised curvature of the S-curve and records the peak of the resulting cross-correlation signal,
      normalised by the signal length. It identifies the best possible bend-matching alignment between the
      two profiles, revealing whether the vase reproduces the S-curve's bending pattern in the right order.
'''

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def extract_derivative_alignment(base):
    '''
    Extracts derivative alignment metrics to compare structural flow.
    Parameters:
        base (dict): Foundational 1D calculus arrays (y, dx_v, dx_s, theta_v, theta_s, k_v, k_s).
    Returns:
        dict: A dictionary containing rigorously extracted derivative matching features.
    '''
    f = {}
    y = base['y']
    dx_v, dx_s = base['dx_v'], base['dx_s']
    theta_v, theta_s = base['theta_v'], base['theta_s']
    k_v, k_s = base['k_v'], base['k_s']
    
    # -----------------------------------------
    # DERIVATIVE ALIGNMENT METRICS
    # -----------------------------------------
    
    # 1. Turning Angle Distance
    # Integrates the absolute difference in tangent angles along the Y-axis. 
    # Evaluates pure trajectory deviation independent of spatial location.
    theta_diff = np.abs(theta_v - theta_s)
    
    # Enforce circular wrap-around to find the true shortest angular distance
    theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)
    f['turning_angle_distance'] = np.trapz(theta_diff, y)
    
    # 2. Derivative Dynamic Time Warping (DDTW)
    # Aligns the first derivatives (slopes) to prevent false matches between flat walls and curved bellies.
    # We reshape the 1D arrays to (N, 1) as required by the fastdtw algorithm.
    distance, _ = fastdtw(dx_v.reshape(-1, 1), dx_s.reshape(-1, 1), dist=euclidean)
    f['derivative_dtw_distance'] = distance
    
    # 3. Max Curvature Cross-Correlation
    # Slides the curvature sequence of the vase across the S-curve to find the absolute peak bend-matching alignment.
    # Both arrays must be strictly standardised to isolate structural timing from magnitude scaling.
    std_k_v = np.std(k_v)
    std_k_s = np.std(k_s)
    
    if std_k_v > 1e-8 and std_k_s > 1e-8:
        k_v_norm = (k_v - np.mean(k_v)) / std_k_v
        k_s_norm = (k_s - np.mean(k_s)) / std_k_s
        
        # Compute full sliding cross-correlation
        cross_corr = np.correlate(k_v_norm, k_s_norm, mode='full')
        
        # Normalise by the length of the overlapping signals to lock the output into a [-1.0, 1.0] range
        f['max_curvature_cross_correlation'] = np.max(cross_corr) / len(k_v)
    else:
        # If either curve is perfectly straight, structural cross-correlation is geometrically undefined.
        f['max_curvature_cross_correlation'] = 0.0
        
    return f