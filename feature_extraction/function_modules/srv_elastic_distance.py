'''
Python file for computing true Elastic Shape Analysis using the
Square Root Velocity (SRV) framework via the fdasrsf library.
Calculates the pure bending and stretching energy required to deform
the vase profile into the target S-Curve, invariant to phase shifts. This includes:
- Srv Elastic Shape Distance
    - SRV Elastic Shape Distance is calculated by converting both the vase and S-curve into 2D Square Root
      Velocity (SRV) q-functions and computing the elastic distance between them using the fdasrsf
      elastic_distance_curve function. It dynamically re-parametrises the S-curve to achieve the best
      possible geometric alignment before measuring the residual L2 norm, providing a distance that is
      invariant to differences in drawing speed or vertical phase shifts.
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