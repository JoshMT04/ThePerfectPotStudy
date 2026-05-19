'''
Python file for computing Category 7 (Typological / Skeleton).
This module isolates anatomical landmarks (Foot, Belly, Neck, Lip) to measure 
the physical transition states and topological skeleton of the digital clay.
'''

import numpy as np

def extract_typological_skeleton(base):
    '''
    Extracts landmark geometry and skeletal transition features.
    Parameters:
        base (dict): A dictionary containing foundational calculus arrays (x, y, dx, dy, k).
    Returns:
        dict: A dictionary containing 8 rigorously extracted typological features.
    '''
    f = {}
    x = np.abs(base['x'])  # Enforce positive radii
    y = base['y']
    k = base['k']
    dx = base['dx']
    dy = base['dy']
    
    y_min, y_max = np.min(y), np.max(y)
    y_total = y_max - y_min
    
    # -----------------------------------------
    # ANATOMICAL LANDMARK ISOLATION
    # -----------------------------------------
    # 1. The Belly (Absolute Maximum Width)
    idx_belly = np.argmax(x)
    y_belly = y[idx_belly]
    
    # 2. The Neck (Minimum Width in the Top Half)
    top_half_mask = y > (y_min + y_total / 2)
    y_top_half = y[top_half_mask]
    x_top_half = x[top_half_mask]
    
    if len(x_top_half) > 0:
        local_neck_idx = np.argmin(x_top_half)
        x_neck = x_top_half[local_neck_idx]
        y_neck = y_top_half[local_neck_idx]
        # Map local top-half index back to global array index
        idx_neck = np.where(y == y_neck)[0][0] 
    else:
        # Fallback if no top half exists
        idx_neck = len(x) - 1
        x_neck = x[-1]
        y_neck = y[-1]
        
    # 3. The Lip (Absolute Top Point)
    idx_lip = -1
    
    # -----------------------------------------
    # CATEGORY 7: TYPOLOGICAL / SKELETON
    # -----------------------------------------
    
    # 66. Shoulder Prominence (Max dx/dy in the top 30%)
    top_30_mask = y > (y_max - 0.3 * y_total)
    dx_top_30 = dx[top_30_mask]
    dy_top_30 = dy[top_30_mask]
    
    # Avoid division by zero
    safe_dy = np.where(np.abs(dy_top_30) < 1e-8, 1e-8, dy_top_30)
    shoulder_gradients = dx_top_30 / safe_dy
    f['shoulder_prominence'] = np.max(shoulder_gradients) if len(shoulder_gradients) > 0 else 0
    
    # 67. Foot Definition (Max curvature in the bottom 5%)
    bot_5_mask = y < (y_min + 0.05 * y_total)
    k_bot_5 = k[bot_5_mask]
    f['foot_definition'] = np.max(np.abs(k_bot_5)) if len(k_bot_5) > 0 else 0
    
    # 68. Neck Elongation (Vertical span where width is within 5% of true neck)
    neck_tolerance = x_neck * 1.05
    neck_zone_mask = (x <= neck_tolerance) & (y >= y_belly) # Must occur above the belly
    if np.any(neck_zone_mask):
        y_neck_zone = y[neck_zone_mask]
        f['neck_elongation'] = np.max(y_neck_zone) - np.min(y_neck_zone)
    else:
        f['neck_elongation'] = 0.0
        
    # 69. Belly-to-Neck Distance
    f['belly_to_neck_distance'] = np.abs(y_belly - y_neck)
    
    # 70. Medial Axis Length
    # For a symmetrical 1D right-hand profile, the topological core spine is the Y-axis.
    f['medial_axis_length'] = y_total
    
    # 71. Lip Flare Angle
    # Angle difference between the tangent at the neck and the tangent at the lip
    theta_neck = np.arctan2(dy[idx_neck], dx[idx_neck])
    theta_lip = np.arctan2(dy[idx_lip], dx[idx_lip])
    
    # Convert absolute angle difference to degrees
    angle_diff = np.abs(theta_lip - theta_neck) * (180 / np.pi)
    f['lip_flare_angle'] = min(angle_diff, 360 - angle_diff)
    
    # 72. Osculating Transition Radius
    # The tightest bend (max curvature) occurring in the transition zone between Belly and Neck
    if idx_neck > idx_belly:
        transition_k = k[idx_belly:idx_neck]
    else:
        transition_k = k[idx_neck:idx_belly]
        
    if len(transition_k) > 0:
        max_transition_k = np.max(np.abs(transition_k))
        # Radius is 1 / curvature. Cap at 1000 to prevent infinite radius on flat walls.
        f['osculating_transition_radius'] = 1 / max_transition_k if max_transition_k > 1e-3 else 1000.0
    else:
        f['osculating_transition_radius'] = 0.0
        
    # 73. Inflection Point Density
    zero_crossings = np.sum(np.diff(np.sign(k)) != 0)
    f['inflection_point_density'] = zero_crossings / y_total if y_total > 0 else 0

    return f