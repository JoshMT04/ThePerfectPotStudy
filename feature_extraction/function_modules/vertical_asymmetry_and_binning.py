'''
Python file for computing Category 4 (Vertical Asymmetry) and Category 8 (Distributional Binning).
This module slices the digital clay into strict vertical floors to measure the volumetric 
imbalance across the Y-axis and evaluates the statistical distribution of the widths.
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