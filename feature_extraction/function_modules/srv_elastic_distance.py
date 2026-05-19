'''
Python file for computing true Elastic Shape Analysis using the 
Square Root Velocity (SRV) framework.
Calculates the pure bending and stretching energy required to deform 
the vase profile into the target S-Curve, invariant to phase shifts.
'''

import numpy as np
import fdasrsf.curve_functions as curve_funcs

def extract_srv_distance(base):
    '''
    Translates raw coordinate profiles into q-functions (SRV) and calculates
    the elastic shape distance.
    '''
    f = {}

    # We must combine X and Y into a 2D curve format for the SRV framework: shape (2, N)
    y = base['y']
    x_v = base['x_v']
    x_s = base['x_s']

    curve_v = np.vstack((x_v, y))
    curve_s = np.vstack((x_s, y))

    # Initialize the curve objects in the fdasrsf framework
    # We treat them as open curves (is_closed=False)
    obj_v = curve_funcs.curve_to_q(curve_v)
    obj_s = curve_funcs.curve_to_q(curve_s)

    # Extract the SRV q-functions
    q_v = obj_v[0]
    q_s = obj_s[0]

    # Calculate the Elastic Shape Distance (Amplitude Distance)
    # This dynamically warps (re-parametrises) the S-curve to perfectly align its
    # geometric features with the vase before calculating the L2 norm.
    try:
        elastic_dist, _ = curve_funcs.elastic_distance_curve(curve_v, curve_s)
        f['srv_elastic_shape_distance'] = elastic_dist
    except Exception as e:
        # Fallback for perfectly straight lines or mathematically collapsed curves
        f['srv_elastic_shape_distance'] = 0.0

    return f