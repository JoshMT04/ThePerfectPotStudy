'''
Python file for computing the Category 3: Curvature and Kinematics features.
This module strictly evaluates the 1st, 2nd, and 3rd spatial derivatives of the digital clay,
mapping the robust statistics of its bending energy, inflection points, and contour jerk. This includes:
- Total Bending Energy
    - Total Bending Energy is calculated as the sum of squared curvature values across all profile points.
      It measures the total mechanical energy stored in the bending of the contour, with higher values
      indicating a more aggressively curved or S-shaped profile.
- Maximum Local Curvature
    - Maximum Local Curvature is the largest absolute curvature value found anywhere along the profile.
      It identifies the sharpest single bend, pinpointing the most extreme physical inflection in the drawn clay.
- Inflection Count
    - Inflection Count records the number of times the curvature changes sign along the profile by detecting
      zero-crossings in the curvature array. It counts how many times the curve switches between bending
      convexly and concavely.
- Zero Crossing Rate
    - Zero Crossing Rate is the Inflection Count divided by the total number of curvature samples. It
      normalises the inflection frequency by profile length, providing a scale-independent measure of
      structural oscillation.
- Mean Convex Curvature
    - Mean Convex Curvature is the average of all strictly positive curvature values. It characterises the
      typical sharpness of outward-bowing regions of the vase profile.
- Mean Concave Curvature
    - Mean Concave Curvature is the average of all strictly negative curvature values. It characterises the
      typical sharpness of inward-bowing (hollow) regions of the profile.
- Convex Concave Energy Ratio
    - Convex Concave Energy Ratio divides the total squared energy of positive curvature by the total squared
      energy of negative curvature. It measures whether the vase expends more bending energy bulging outward
      or curving inward.
- Mean Absolute Curvature
    - Mean Absolute Curvature is the average of the absolute curvature values across the full profile.
      It provides a single robust summary of overall bending intensity regardless of direction.
- Median Curvature
    - Median Curvature is the middle value of the sorted curvature distribution. It provides a robust
      central tendency measure of bending that is resistant to extreme outlier peaks.
- Curvature Variance
    - Curvature Variance is the variance of the curvature array. It measures the spread of bending intensity
      around the mean, with higher values indicating a more spatially inconsistent curvature profile.
- Curvature Range
    - Curvature Range is the peak-to-peak difference (max minus min) of the curvature array. It captures
      the full dynamic range of bending from the sharpest convex peak to the sharpest concave valley.
- Curvature Iqr
    - Curvature Iqr is the interquartile range of the curvature distribution (75th percentile minus 25th
      percentile). It measures the dispersion of the central 50% of curvature values, robust to outlier
      extremes.
- Median K
    - Median K is a second record of the median curvature value, stored explicitly for direct downstream
      use in computing the Median Absolute Deviation without re-sorting.
- Curvature Median Abs Deviation
    - Curvature Median Absolute Deviation is the median of the absolute deviations from the median curvature.
      It provides the most outlier-resistant measure of curvature variability.
- Mean Abs K
    - Mean Abs K is the mean of the absolute curvature values, stored explicitly as a named feature for
      direct downstream retrieval alongside the coefficient of variation calculation.
- Curvature Coef Of Variation
    - Curvature Coefficient of Variation is the standard deviation of curvature divided by its mean. It
      measures relative variability, enabling comparison of bending consistency across vases of different
      overall curvature magnitudes.
- Curvature Kurtosis
    - Curvature Kurtosis is the fourth standardised moment of the curvature distribution. It measures whether
      bending extremes are concentrated in sharp, rare spikes (leptokurtic) or spread across many moderate
      bends (platykurtic).
- Curvature Smoothness Index
    - Curvature Smoothness Index is calculated as 1 divided by the standard deviation of successive curvature
      differences. It measures how gradually or abruptly the bending changes along the profile, with higher
      values indicating a smoother, more gradual curvature transition.
- Local Maxima Density
    - Local Maxima Density counts the physical peaks in the absolute curvature signal using scipy's
      find_peaks and divides by the total perimeter length. It measures how frequently the vase produces
      distinct bulges or pinches per unit of arc length.
- Mean Absolute Jerk
    - Mean Absolute Jerk is the mean of the absolute rate of change of curvature with respect to arc length
      (dk/ds). It measures the typical abruptness of curvature transitions, penalising sudden sharp changes
      in bending direction.
- Jerk Variance
    - Jerk Variance is the variance of the arc-length-normalised curvature derivative (dk/ds). It quantifies
      the inconsistency of curvature transitions, with higher values indicating erratic, unpredictable
      changes in bending.
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