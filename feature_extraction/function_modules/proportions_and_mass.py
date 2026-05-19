'''
Python file for computing the proportions and mass-related features of the vase contour based on the foundational 
calculus and spatial states computed in function_modules.core_engine.py. This includes:
- Aspect Ratio
    - Aspect Ratio is calculated as the ratio of the total height (y_total) to the total width (x_total) of the contour. 
      It provides a measure of how tall or wide the vase is, with higher values indicating a taller shape.
- Compactness
    - Compactness is calculated using the formula: (4 * π * area) / (perimeter^2). 
      It quantifies how closely the shape resembles a perfect circle, 
      with values closer to 1 indicating a more circular shape.
- Max Width Height
    - Max Width Height is determined by finding the y-coordinate at which the maximum width (x) occurs 
      and normalising it by the total height (y_total). 
      It indicates the relative height at which the vase is widest, with values closer to 0 indicating a 
      wider base and values closer to 1 indicating a wider top.
- Neck-to-Base Ratio
    - Neck-to-Base Ratio is calculated by dividing the minimum width in the top half of the vase by the absolute base width.
      It measures the structural tapering of the design.
- Lip-to-Belly Ratio
    - Lip-to-Belly Ratio is calculated by dividing the absolute top width by the absolute maximum width.
      It indicates whether the opening is tightly closed or widely flared.
- Bounding Box Solidity
    - Bounding Box Solidity is the true 2D area divided by the bounding box area.
      It measures how much of its own theoretical space the vase actually fills.
- Max Calliper Diameter
    - Max Calliper Diameter is the absolute longest physical dimension across the mirrored vase.
- Base Stability Index
    - Base Stability Index is calculated by dividing the base width by the maximum width.
      It provides a measure of physical groundedness versus top-heaviness.
- Shoulder Slope Angle
    - Shoulder Slope Angle quantifies the aggressiveness of the shoulder transition using the arctangent of the neck and max width coordinates.
- Enclosing Circle Deficit
    - Enclosing Circle Deficit is calculated as 1 - (area / area of the minimum enclosing circle).
      It indicates how efficiently the vase packs into a perfect circle.
- Perimeter Convexity
    - Perimeter Convexity is the ratio of the convex hull perimeter to the true perimeter.
      It quantifies the extent to which the contour meanders inward.
- Y Centroid Normalised
    - Y Centroid Normalised is calculated by normalising the y-coordinate of the 2D centroid (cy) by the total height (y_total). 
      It provides a measure of how the mass of the vase is distributed along its height, with values closer to 0 
      indicating a lower centre of mass and values closer to 1 indicating a higher centre of mass.
- Convex Hull Deficit
    - Convex Hull Deficit is calculated as 1 - (area / convex hull area). 
      It quantifies how much the shape deviates from its convex hull, with higher values indicating a more concave shape.
- Top/Bottom Area Ratio
    - Top/Bottom Area Ratio is calculated by splitting the contour at its vertical midpoint and comparing the area of the top half to the bottom half. 
      It provides insight into the distribution of mass and shape between the upper and lower parts of the vase.
- Rotational Volume
    - Rotational Volume calculates the theoretical 3D capacity by integrating the profile as a solid of revolution.
- Surface-to-Volume Ratio
    - Surface-to-Volume Ratio indicates the physical efficiency of the shape, dividing 3D surface area by rotational volume.
- Vertical 3D Centre of Mass
    - Vertical 3D Centre of Mass pinpoints the true volumetric weight anchor, shifting the 2D centroid into physical reality.
- Eccentricity
    - Eccentricity mathematically fits an ellipse to the proportions to measure how naturally stretched the geometry is.
- Mean Centroid Distance
    - Mean Centroid Distance computes the average physical radius of the clay measured outward from its visual core.
- Base-to-Centroid & Centroid-to-Top Heights
    - These metrics measure the absolute vertical distance supporting and suspended above the main visual weight.
'''

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def extract_proportions_and_mass(base):
    '''
    Extracts proportions and mass-related features from the vase contour.
    Parameters:
        base (dict): A dictionary containing the base features of the vase contour.
    Returns:
        dict: A dictionary containing the extracted proportions and mass-related features.
    '''
    f = {}
    x, y, p = np.abs(base['x']), base['y'], base['perimeter']
    k = base['k']
    y_min, y_max = np.min(y), np.max(y)
    y_total = y_max - y_min
    
    # Mirror the profile to simulate the full 2D symmetric vase for bounded metrics
    full_x = np.concatenate([x, -x[::-1]])
    full_y = np.concatenate([y, y[::-1]])
    full_coords = np.column_stack((full_x, full_y))
    
    # Base areas and perimeters of the TRUE 2D shape
    area_2d = 2 * np.trapz(x, y)
    perimeter_2d = 2 * np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)) + (x[0]*2) + (x[-1]*2)
    
    # -----------------------------------------
    # CATEGORY 1: MACRO-PROPORTIONS
    # -----------------------------------------
    max_x_idx = np.argmax(x)
    x_max = x[max_x_idx]
    y_max_x = y[max_x_idx]
    
    x_base = x[0] if x[0] > 0 else 1e-8
    x_lip = x[-1]
    
    # Isolate the top half to find the true neck
    y_mid = y_min + (y_total / 2)
    top_half_mask = y > y_mid
    x_top_half = x[top_half_mask]
    y_top_half = y[top_half_mask]
    neck_idx = np.argmin(x_top_half) if len(x_top_half) > 0 else -1
    x_neck = x_top_half[neck_idx] if neck_idx != -1 else x_lip
    y_neck = y_top_half[neck_idx] if neck_idx != -1 else y_max
    
    # Aspect Ratio
    # Aspect Ratio is calculated as the ratio of the total height (y_total) to the total width (x_total) of the contour. 
    # It provides a measure of how tall or wide the vase is, with higher values indicating a taller shape.
    f['aspect_ratio'] = y_total / (x_max * 2) if x_max > 0 else 0
    
    # Compactness
    # Compactness is calculated using the formula: (4 * π * area) / (perimeter^2). 
    # It quantifies how closely the shape resembles a perfect circle, with values closer to 1 indicating a more circular shape.
    f['compactness'] = (4 * np.pi * area_2d) / (perimeter_2d**2) if perimeter_2d > 0 else 0
    
    # Max Width Height
    # Max Width Height is determined by finding the y-coordinate at which the maximum width (x) occurs and normalising it. 
    # It indicates the relative height at which the vase is widest.
    f['max_width_height'] = (y_max_x - y_min) / y_total
    
    # Neck-to-Base Ratio
    f['neck_to_base_ratio'] = x_neck / x_base
    
    # Lip-to-Belly Ratio
    f['lip_to_belly_ratio'] = x_lip / x_max if x_max > 0 else 0
    
    # Bounding Box Solidity
    f['bounding_box_solidity'] = area_2d / (y_total * x_max * 2) if x_max > 0 else 0
    
    # Max Calliper Diameter (approximate as max distance between points on the full profile)
    max_calliper_diameter = np.max(pdist(full_coords))
    f['max_calliper_diameter'] = max_calliper_diameter
    
    # Base Stability Index
    f['base_stability_index'] = x_base / x_max if x_max > 0 else 0
    
    # Shoulder Slope Angle
    # Quantifies the aggressiveness of the shoulder transition.
    dy_shoulder = y_max_x - y_neck
    dx_shoulder = x_max - x_neck
    f['shoulder_slope_angle'] = np.arctan(dx_shoulder / dy_shoulder) * (180 / np.pi) if dy_shoulder != 0 else 90
    
    # Perimeter Convexity 
    # The perimeter convexity is 'how much shorter is a tight string around the shape than the actual perimeter?'
    try:
        hull = ConvexHull(full_coords)
        f['perimeter_convexity'] = hull.area / perimeter_2d if perimeter_2d > 0 else 0
        f['convex_hull_deficit'] = 1 - (area_2d / hull.volume)
    except:
        f['perimeter_convexity'] = 1
        f['convex_hull_deficit'] = 0

    # Enclosing Circle Deficit
    # The enclosing circle deficit is calculated as 1 - (area / area of the minimum enclosing circle). 
    enclosing_circle_area = np.pi * (max_calliper_diameter / 2)**2
    f['enclosing_circle_deficit'] = 1 - (area_2d / enclosing_circle_area) if enclosing_circle_area > 0 else 0

    # -----------------------------------------
    # CATEGORY 2: MASS & VOLUME
    # -----------------------------------------
    
    # Rotational Volume
    # Measure of how much volume the shape would have if rotated around its central vertical axis.
    # Calculated strictly using trapezoidal integration for maximum geometric precision.
    vol_3d = np.trapz(np.pi * x**2, y)
    f['rotational_volume'] = vol_3d
    
    # Surface-to-Volume Ratio
    # Provides insight into how much surface area the shape has relative to its volume.
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    x_mid = (x[:-1] + x[1:]) / 2
    surface_area_3d = np.sum(2 * np.pi * x_mid * ds) + (np.pi * x_base**2)
    f['surface_to_volume_ratio'] = surface_area_3d / vol_3d if vol_3d > 0 else 0
    
    # Y Centroid Normalised
    # Provides a measure of how the mass of the vase is distributed along its height.
    cy_2d = np.trapz(y * x, y) / np.trapz(x, y) if np.trapz(x, y) > 0 else y_min
    f['y_centroid_normalised'] = (cy_2d - y_min) / y_total
    
    # Vertical 3D Centre of Mass
    # Calculates the true physical centre of mass based on volumetric slices.
    cy_3d = np.trapz(y * np.pi * x**2, y) / vol_3d if vol_3d > 0 else y_min
    f['3d_centre_of_mass_vertical'] = (cy_3d - y_min) / y_total
    
    # Top/Bottom Area Ratio
    # Splitting the contour at its vertical midpoint and comparing the area of the top half to the bottom half. 
    area_top = 2 * np.trapz(x[top_half_mask], y[top_half_mask])
    bottom_half_mask = y <= y_mid
    area_bottom = 2 * np.trapz(x[bottom_half_mask], y[bottom_half_mask])
    f['top_to_bottom_area_ratio'] = area_top / area_bottom if area_bottom > 0 else 1
    
    # Eccentricity
    # Mathematically fits an ellipse bounding box to evaluate stretching independently of bounding solidity.
    a, b = max(y_total, x_max*2)/2, min(y_total, x_max*2)/2
    f['eccentricity'] = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
    
    # Mean Centroid Distance
    distances_to_centroid = np.sqrt((x - base['cx'])**2 + (y - base['cy'])**2)
    f['mean_centroid_distance'] = np.mean(distances_to_centroid)
    
    # Base-to-Centroid Height
    f['base_to_centroid_height_ratio'] = cy_2d - y_min
    
    # Centroid-to-Top Height
    f['centroid_to_top_height_ratio'] = y_max - cy_2d
    
    # -----------------------------------------
    # NEW CATEGORY: PHYSICAL AFFORDANCE & FUNCTION
    # -----------------------------------------
    
    # 1. Critical Tipping Angle (Degrees)
    # Requires cy_3d (3D Centre of Mass Y-coordinate) from the volumetric calculations
    x_base = np.abs(x[0])  # The physical radius of the foot
    if cy_3d > np.min(y):
        tipping_radians = np.arctan(x_base / (cy_3d - np.min(y)))
        f['critical_tipping_angle'] = tipping_radians * (180 / np.pi)
    else:
        f['critical_tipping_angle'] = 90.0

    # 2. Evaporation Exposure Quotient
    x_lip = np.abs(x[-1])
    lip_area = np.pi * (x_lip ** 2)
    f['evaporation_exposure_quotient'] = lip_area / f['rotational_volume'] if f['rotational_volume'] > 0 else 0

    # 3. Ergonomic Grip Security
    # Isolate the top half of the vase to search for a hand-hold
    y_mid = np.min(y) + (y_total / 2)
    top_half_mask = y > y_mid
    k_top = k[top_half_mask]
    
    # We are looking for the most extreme concavity (negative curvature)
    if np.any(k_top < 0):
        f['ergonomic_grip_security'] = np.max(np.abs(k_top[k_top < 0]))
    else:
        f['ergonomic_grip_security'] = 0.0

    # 4. Stem Support Index
    x_belly = np.max(np.abs(x))
    # Neck is the minimum width in the top half
    x_top_half = np.abs(x[top_half_mask])
    x_neck = np.min(x_top_half) if len(x_top_half) > 0 else x_lip
    
    f['stem_support_index'] = (x_belly - x_neck) / x_belly if x_belly > 0 else 0.0

    # 5. Material Efficiency Ratio
    arc_len_v = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    f['material_efficiency_ratio'] = f['rotational_volume'] / arc_len_v if arc_len_v > 0 else 0.0
    
    # -----------------------------------------
    # NEW CATEGORY: STRUCTURAL VULNERABILITY & FRAGILITY
    # -----------------------------------------
    
    # 6. Cantilever Snap Stress (Decapitation Metric)
    # Find the narrowest point (neck) to act as the structural fulcrum
    idx_neck = np.argmin(np.abs(x))
    y_neck = y[idx_neck]
    x_neck = np.abs(x[idx_neck])
    
    # Isolate the mass strictly above the neck
    mask_above = y > y_neck
    if np.any(mask_above) and x_neck > 0:
        x_above = np.abs(x[mask_above])
        y_above = y[mask_above]
        
        # Calculate approximate volume and CoM of the top section
        vol_above = np.trapz(np.pi * (x_above**2), y_above)
        if vol_above > 0:
            cy_above = np.trapz(np.pi * (x_above**2) * y_above, y_above) / vol_above
            lever_arm = cy_above - y_neck
            # Torque = Force (Volume) * Lever Arm. Divide by neck width acting as resistance.
            f['cantilever_snap_stress'] = (vol_above * lever_arm) / (x_neck ** 2)
        else:
            f['cantilever_snap_stress'] = 0.0
    else:
        f['cantilever_snap_stress'] = 0.0

    # 7. Overhang Slump Risk (Kiln Collapse)
    # Detect aggressive outward horizontal expansions (dx/dy is large and positive)
    dy = np.gradient(y)
    dx = np.gradient(np.abs(x))
    # Avoid division by zero
    dy[dy == 0] = 1e-6 
    slopes = dx / dy
    
    # Isolate outward slopes that are flatter than 45 degrees (slope > 1)
    overhang_mask = slopes > 1.0
    if np.any(overhang_mask):
        f['overhang_slump_risk'] = np.sum(slopes[overhang_mask]) / len(y)
    else:
        f['overhang_slump_risk'] = 0.0

    # 8. Protrusion Chipping Vulnerability
    # Find the single sharpest outward-facing point (highest convex curvature)
    if np.any(k > 0):
        f['protrusion_chip_vulnerability'] = np.max(k[k > 0])
    else:
        f['protrusion_chip_vulnerability'] = 0.0

    # 9. Foot Leverage Fracture Risk
    x_max = np.max(np.abs(x))
    idx_max = np.argmax(np.abs(x))
    y_max = y[idx_max]
    x_base_val = np.abs(x[0]) if np.abs(x[0]) > 0 else 1e-6
    
    f['foot_leverage_fracture_risk'] = (x_max / x_base_val) * (y_max - np.min(y))
    
    # -----------------------------------------
    # NEW CATEGORY: LIFECYCLE & FLUID DYNAMICS
    # -----------------------------------------
    
    # 10. Cleaning Accessibility Index
    # Ratio of the total depth of the vase to the width of the access bottleneck (neck)
    y_total = np.max(y) - np.min(y)
    idx_neck = np.argmin(np.abs(x))
    x_neck_val = np.abs(x[idx_neck]) if np.abs(x[idx_neck]) > 0 else 1e-6
    
    f['cleaning_accessibility_index'] = y_total / x_neck_val

    # 11. Fluid Evacuation Constraint (Pouring Bottleneck)
    # Total volume fighting to escape through the neck's cross-sectional area
    neck_area = np.pi * (x_neck_val ** 2)
    f['fluid_evacuation_constraint'] = f['rotational_volume'] / neck_area if neck_area > 0 else 0.0

    # 12. Dynamic Ballast Distribution
    # Ratio of water storage in the bottom half vs the top half
    y_mid = np.min(y) + (y_total / 2)
    mask_bottom = y <= y_mid
    mask_top = y > y_mid
    
    vol_bottom = np.trapz(np.pi * (np.abs(x[mask_bottom])**2), y[mask_bottom]) if np.any(mask_bottom) else 0.0
    vol_top = np.trapz(np.pi * (np.abs(x[mask_top])**2), y[mask_top]) if np.any(mask_top) else 0.0
    
    # If top volume is 0 (a flat plate), ballast is perfectly bottom-heavy (infinity handled by high float)
    f['dynamic_ballast_distribution'] = vol_bottom / vol_top if vol_top > 0 else vol_bottom * 1000

    # 13. Tabletop Real Estate Efficiency
    # Volume of water held per square unit of table space stolen
    base_area = np.pi * (np.abs(x[0]) ** 2) if np.abs(x[0]) > 0 else 1e-6
    f['tabletop_real_estate_efficiency'] = f['rotational_volume'] / base_area

    return f