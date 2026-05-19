'''
Python file for computing the Category 3: Curvature and Kinematics features.
This module strictly evaluates the 1st, 2nd, and 3rd spatial derivatives of the digital clay, 
mapping the robust statistics of its bending energy, inflection points, and contour jerk.
'''

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks

def extract_kinematics(base):
    '''
    Extracts curvature, kinematic, and spatial derivative features from the vase contour.
    Parameters:
        base (dict): A dictionary containing the foundational calculus arrays (x, y, k, perimeter).
    Returns:
        dict: A dictionary containing 20 rigorously extracted kinematic features.
    '''
    f = {}
    k = base['k']
    x, y, p = base['x'], base['y'], base['perimeter']
    
    # -----------------------------------------
    # ARC LENGTH DIFFERENTIAL & JERK CALCULUS
    # -----------------------------------------
    # True physical jerk is the derivative of curvature with respect to arc length (dk/ds), 
    # not with respect to the array index.
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx**2 + dy**2)
    ds[ds == 0] = 1e-8  # Prevent division by zero on perfectly duplicate points
    
    dk_ds = np.gradient(k) / ds
    
    # -----------------------------------------
    # CATEGORY 3: CURVATURE & KINEMATICS
    # -----------------------------------------
    
    # Bending Energy & Extremes
    f['total_bending_energy'] = np.sum(k**2)
    f['maximum_local_curvature'] = np.max(np.abs(k))
    
    # Inflection & Crossings
    # A sign change indicates the curve passing through absolute flatness
    zero_crossings = np.diff(np.sign(k)) != 0
    f['inflection_count'] = np.sum(zero_crossings)
    f['zero_crossing_rate'] = f['inflection_count'] / len(k)
    
    # Convex / Concave Isolation
    k_pos = k[k > 0]
    k_neg = k[k < 0]
    f['mean_convex_curvature'] = np.mean(k_pos) if len(k_pos) > 0 else 0
    f['mean_concave_curvature'] = np.mean(k_neg) if len(k_neg) > 0 else 0
    
    sum_sq_pos = np.sum(k_pos**2)
    sum_sq_neg = np.sum(k_neg**2)
    f['convex_concave_energy_ratio'] = sum_sq_pos / sum_sq_neg if sum_sq_neg > 0 else sum_sq_pos
    
    # Robust Distributional Statistics
    f['mean_absolute_curvature'] = np.mean(np.abs(k))
    f['median_curvature'] = np.median(k)
    
    f['curvature_variance'] = np.var(k)
    f['curvature_range'] = np.ptp(k)  # Equivalent to max(k) - min(k)
    
    q75, q25 = np.percentile(k, [75, 25])
    f['curvature_iqr'] = q75 - q25
    
    median_k = np.median(k)
    f['median_k'] = median_k
    f['curvature_median_abs_deviation'] = np.median(np.abs(k - median_k))
    
    mean_k = np.mean(k)
    f['mean_abs_k'] = np.mean(np.abs(k))
    f['curvature_coef_of_variation'] = np.std(k) / mean_k if mean_k != 0 else 0
    f['curvature_kurtosis'] = kurtosis(k)
    
    # Spatial Density & Smoothness
    # Smoothness is the inverse of the standard deviation of the discrete change in curvature
    std_delta_k = np.std(np.diff(k))
    f['curvature_smoothness_index'] = 1 / std_delta_k if std_delta_k > 0 else 0
    
    # Detect true local maxima (peaks) in absolute bending to count physical bulges/pinches
    peaks, _ = find_peaks(np.abs(k))
    f['local_maxima_density'] = len(peaks) / p if p > 0 else 0
    
    # Jerk (3rd Derivative) Metrics
    f['mean_absolute_jerk'] = np.mean(np.abs(dk_ds))
    f['jerk_variance'] = np.var(dk_ds)
    
    return f