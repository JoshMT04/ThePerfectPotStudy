'''
Python file for computing Category 4 (Vertical Asymmetry) and Category 8 (Distributional Binning).
This module slices the digital clay into strict vertical floors to measure the volumetric
imbalance across the Y-axis and evaluates the statistical distribution of the widths. This includes:
- Vertical Area Asymmetry
    - Vertical Area Asymmetry is the absolute difference between the 2D area of the top half and the 2D
      area of the bottom half of the vase, split at the vertical midpoint. It measures the volumetric
      imbalance between the upper and lower portions of the shape.
- Midpoint Y Deviation
    - Midpoint Y Deviation is the difference between the Y-weighted centroid of the profile (computed via
      trapezoidal integration) and the geometric midpoint of the Y-axis span. Positive values indicate
      the visual mass centre sits above the midpoint; negative values indicate it sits below.
- Vertical Bending Ratio
    - Vertical Bending Ratio divides the sum of squared curvature in the top half by the sum of squared
      curvature in the bottom half. It measures whether the vase expends more bending energy in the upper
      shoulder and lip region or in the lower belly and base region.
- Vertical Hausdorff Distance
    - Vertical Hausdorff Distance is the Hausdorff distance between the top half coordinates and the
      bottom half coordinates after folding the bottom half upward over the midpoint axis. It measures the
      geometric dissimilarity between the shape's upper and lower halves when superimposed.
- Vertical Contour Length Ratio
    - Vertical Contour Length Ratio divides the arc length of the top half of the profile by the arc
      length of the bottom half. It quantifies whether the upper portion of the vase has a longer,
      more complex path than the lower portion.
- Width Interquartile Range
    - Width Interquartile Range is the difference between the 75th and 25th percentiles of the X-coordinate
      (width) array across the full profile. It measures the spread of widths in the central portion of the
      distribution, robust to extreme base or lip values.
- Radial Dist Interquartile Range
    - Radial Distribution Interquartile Range applies the same 75th-minus-25th-percentile calculation to
      the X-coordinates of the core profile region (excluding the bottom 5% and top 5% by height). It
      focuses the width spread measurement on the main body, discarding the foot and lip extremes.
- Extreme Convexity Proportion
    - Extreme Convexity Proportion is the fraction of curvature values exceeding the 90th percentile. It
      measures how much of the profile is occupied by the sharpest outward-bending regions.
- Extreme Concavity Proportion
    - Extreme Concavity Proportion is the fraction of curvature values below the 10th percentile. It
      measures how much of the profile is occupied by the sharpest inward-bowing regions.
- Flatness Proportion
    - Flatness Proportion is the fraction of curvature values with absolute magnitude below 0.001. It
      measures how much of the vase profile is essentially straight, with negligible bending in either
      direction.
- Bottom Third Area Mass
    - Bottom Third Area Mass is the 2D area of the lowest third of the vase by height divided by the
      total 2D area. It measures what proportion of the vase's overall shape mass is concentrated in the
      base third.
- Mid Third Area Mass
    - Mid Third Area Mass is the 2D area of the middle third of the vase by height divided by the total
      2D area. It measures what proportion of the vase's overall shape mass is concentrated in the belly
      region.
- Top Third Area Mass
    - Top Third Area Mass is the 2D area of the top third of the vase by height divided by the total 2D
      area. It measures what proportion of the vase's overall shape mass is concentrated in the shoulder
      and lip region.
- Upper Half Average Width
    - Upper Half Average Width is the mean X-coordinate (radius) of all profile points in the top half of
      the vase. It provides a single summary of the typical width above the vertical midpoint.
- Lower Half Average Width
    - Lower Half Average Width is the mean X-coordinate (radius) of all profile points in the bottom half
      of the vase. It provides a single summary of the typical width below the vertical midpoint.
- Y Axis Width Variance
    - Y Axis Width Variance is the variance of the X-coordinate (width) array across the full profile.
      It measures the overall spread of widths from base to lip, with higher values indicating a more
      dramatically varying silhouette.
'''

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def extract_vertical_asymmetry_and_binning(base):
    '''
    Extracts vertical asymmetry and spatial binning features from the vase contour.
    Parameters:
        base (dict): A dictionary containing foundational calculus arrays (x, y, k).
    Returns:
        dict: A dictionary containing 16 rigorously extracted asymmetry and binning features.
    '''
    f = {}
    x = np.abs(base['x'])  # Enforce positive radii
    y = base['y']
    k = base['k']
    
    y_min, y_max = np.min(y), np.max(y)
    y_total = y_max - y_min
    y_mid = y_min + (y_total / 2)
    
    # Area of full 2D shape for normalisation
    total_area = 2 * np.trapz(x, y)
    
    # -----------------------------------------
    # VERTICAL SLICING MASKS
    # -----------------------------------------
    top_mask = y > y_mid
    bot_mask = y <= y_mid
    
    x_top, y_top = x[top_mask], y[top_mask]
    x_bot, y_bot = x[bot_mask], y[bot_mask]
    
    # -----------------------------------------
    # CATEGORY 4: VERTICAL ASYMMETRY
    # -----------------------------------------
    
    # 1. Vertical Area Asymmetry
    area_top = 2 * np.trapz(x_top, y_top) if len(x_top) > 1 else 0
    area_bot = 2 * np.trapz(x_bot, y_bot) if len(x_bot) > 1 else 0
    f['vertical_area_asymmetry'] = np.abs(area_top - area_bot)
    
    # 2. Midpoint Y-Deviation
    cy_2d = np.trapz(y * x, y) / np.trapz(x, y) if np.trapz(x, y) > 0 else y_min
    f['midpoint_y_deviation'] = cy_2d - y_mid
    
    # 3. Vertical Bending Ratio
    k_top, k_bot = k[top_mask], k[bot_mask]
    sum_k_top_sq = np.sum(k_top**2)
    sum_k_bot_sq = np.sum(k_bot**2)
    f['vertical_bending_ratio'] = sum_k_top_sq / sum_k_bot_sq if sum_k_bot_sq > 0 else sum_k_top_sq
    
    # 4. Vertical Hausdorff Distance
    # Mathematically fold the bottom half up over the y_mid axis to superimpose it on the top
    y_bot_flipped = y_mid + (y_mid - y_bot)
    top_coords = np.column_stack((x_top, y_top))
    bot_flipped_coords = np.column_stack((x_bot, y_bot_flipped))
    
    if len(top_coords) > 0 and len(bot_flipped_coords) > 0:
        hausdorff_top_to_bot = directed_hausdorff(top_coords, bot_flipped_coords)[0]
        hausdorff_bot_to_top = directed_hausdorff(bot_flipped_coords, top_coords)[0]
        f['vertical_hausdorff_distance'] = max(hausdorff_top_to_bot, hausdorff_bot_to_top)
    else:
        f['vertical_hausdorff_distance'] = 0.0
        
    # 5. Vertical Contour Length Ratio
    perim_top = np.sum(np.sqrt(np.diff(x_top)**2 + np.diff(y_top)**2)) if len(x_top) > 1 else 0
    perim_bot = np.sum(np.sqrt(np.diff(x_bot)**2 + np.diff(y_bot)**2)) if len(x_bot) > 1 else 0
    f['vertical_contour_length_ratio'] = perim_top / perim_bot if perim_bot > 0 else 0
    
    # -----------------------------------------
    # CATEGORY 8: DISTRIBUTIONAL BINNING
    # -----------------------------------------
    
    # 6. Width Interquartile Range
    q75_w, q25_w = np.percentile(x, [75, 25])
    f['width_interquartile_range'] = q75_w - q25_w
    
    # 7. Radial Dist. Interquartile Range 
    # Core mask targets the middle 90% of the vase to aggressively discard lip/base variance
    core_mask = (y > (y_min + 0.05 * y_total)) & (y < (y_max - 0.05 * y_total))
    x_core = x[core_mask] if np.any(core_mask) else x
    q75_rad, q25_rad = np.percentile(x_core, [75, 25])
    f['radial_dist_interquartile_range'] = q75_rad - q25_rad
    
    # 8 & 9. Extreme Curvature Proportions
    k_90 = np.percentile(k, 90)
    k_10 = np.percentile(k, 10)
    f['extreme_convexity_proportion'] = np.sum(k > k_90) / len(k)
    f['extreme_concavity_proportion'] = np.sum(k < k_10) / len(k)
    
    # 10. Flatness Proportion (Micro-bin)
    f['flatness_proportion'] = np.sum(np.abs(k) < 1e-3) / len(k)
    
    # 11, 12, 13. Thirds Area Mass
    y_third_1 = y_min + (y_total / 3)
    y_third_2 = y_min + (2 * y_total / 3)
    
    bot_third_mask = y <= y_third_1
    mid_third_mask = (y > y_third_1) & (y <= y_third_2)
    top_third_mask = y > y_third_2
    
    area_bot_third = 2 * np.trapz(x[bot_third_mask], y[bot_third_mask]) if np.sum(bot_third_mask) > 1 else 0
    area_mid_third = 2 * np.trapz(x[mid_third_mask], y[mid_third_mask]) if np.sum(mid_third_mask) > 1 else 0
    area_top_third = 2 * np.trapz(x[top_third_mask], y[top_third_mask]) if np.sum(top_third_mask) > 1 else 0
    
    total_area_safe = total_area if total_area > 0 else 1
    f['bottom_third_area_mass'] = area_bot_third / total_area_safe
    f['mid_third_area_mass'] = area_mid_third / total_area_safe
    f['top_third_area_mass'] = area_top_third / total_area_safe
    
    # 14 & 15. Half Average Widths
    f['upper_half_average_width'] = np.mean(x_top) if len(x_top) > 0 else 0
    f['lower_half_average_width'] = np.mean(x_bot) if len(x_bot) > 0 else 0
    
    # 16. Y-Axis Width Variance
    f['y_axis_width_variance'] = np.var(x)
    
    return f