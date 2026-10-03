'''
Relational Pipeline: main_curve_matcher.py
This module constructs the comparative base state between a vase half-profile and a target
S-curve, then orchestrates all four relational feature extraction modules: spatial error,
derivative alignment, spatial Y-binning, and amplitude binning.
'''

import numpy as np

from feature_extraction.old.function_modules.srv_elastic_distance import extract_srv_distance
from feature_extraction.old.function_modules.spatial_error_and_flexibility import extract_spatial_error_and_flexibility
from feature_extraction.old.function_modules.derivatives_and_distributions import extract_derivative_alignment
from feature_extraction.old.function_modules.spatial_y_binning import extract_spatial_y_binning
from feature_extraction.old.function_modules.amplitude_binning import extract_amplitude_binning


def compare_vase_to_target(y_coords, vase_x, target_x, target_name):
    '''
    Builds the relational base state and executes all comparative feature extraction
    between a vase profile and a target S-curve.

    Parameters:
        y_coords  (np.ndarray): Shared 250-point Y-axis coordinates (bottom to top).
        vase_x    (np.ndarray): Absolute right-hand X-coordinates of the vase profile.
        target_x  (np.ndarray): X-coordinates of the standardised target S-curve.
        target_name      (str): Identifier prefix applied to every returned feature key.

    Returns:
        dict: All relational features, each key prefixed with '{target_name}_'.
    '''

    y   = y_coords
    x_v = vase_x
    x_s = target_x

    # -----------------------------------------
    # SIGNED & ABSOLUTE HORIZONTAL ERROR
    # -----------------------------------------
    err     = x_v - x_s       # Positive = vase wider than target; Negative = vase narrower
    abs_err = np.abs(err)

    # -----------------------------------------
    # FIRST & SECOND DERIVATIVES (shared Y parameter)
    # -----------------------------------------
    dy  = np.gradient(y)
    dy[dy == 0] = 1e-8         # Guard against collapsed segments

    dx_v  = np.gradient(x_v)
    dx_s  = np.gradient(x_s)

    ddx_v = np.gradient(dx_v)
    ddx_s = np.gradient(dx_s)
    ddy   = np.gradient(dy)

    # -----------------------------------------
    # CURVATURE  k = (dx·ddy − dy·ddx) / (dx²+dy²)^1.5
    # -----------------------------------------
    denom_v = (dx_v**2 + dy**2)**1.5
    denom_v[denom_v == 0] = 1e-8
    k_v = (dx_v * ddy - dy * ddx_v) / denom_v

    denom_s = (dx_s**2 + dy**2)**1.5
    denom_s[denom_s == 0] = 1e-8
    k_s = (dx_s * ddy - dy * ddx_s) / denom_s

    # -----------------------------------------
    # TURNING ANGLES  (tangent direction relative to Y-axis)
    # -----------------------------------------
    theta_v = np.arctan2(dx_v, dy)
    theta_s = np.arctan2(dx_s, dy)

    # -----------------------------------------
    # ARC LENGTHS
    # -----------------------------------------
    arc_len_v = np.sum(np.sqrt(np.diff(x_v)**2 + np.diff(y)**2))
    arc_len_s = np.sum(np.sqrt(np.diff(x_s)**2 + np.diff(y)**2))

    relational_base = {
        'x_v':       x_v,
        'x_s':       x_s,
        'y':         y,
        'err':       err,
        'abs_err':   abs_err,
        'dx_v':      dx_v,
        'dx_s':      dx_s,
        'theta_v':   theta_v,
        'theta_s':   theta_s,
        'k_v':       k_v,
        'k_s':       k_s,
        'arc_len_v': arc_len_v,
        'arc_len_s': arc_len_s,
    }

    # -----------------------------------------
    # EXECUTE ALL RELATIONAL MODULES
    # -----------------------------------------
    combined = {}
    combined.update(extract_spatial_error_and_flexibility(relational_base))
    combined.update(extract_derivative_alignment(relational_base))
    combined.update(extract_spatial_y_binning(relational_base))
    combined.update(extract_amplitude_binning(relational_base))
    combined.update(extract_srv_distance(relational_base))

    # Prefix every key with the target name to prevent collision across 5 S-curve comparisons
    return {f'{target_name}_{key}': value for key, value in combined.items()}
