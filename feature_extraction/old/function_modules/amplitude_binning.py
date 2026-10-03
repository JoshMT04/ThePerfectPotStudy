'''
Python file for computing the Amplitude (Horizontal) Binning relational features.
This module mathematically slices the 1D profile into strict radial zones (amplitudes)
to map where the spatial error and kinematic bending occur relative to the centreline. This includes:
- Amplitude Bin N Peak/Valley Residence (amp_bin_N_peak_valley_residence)
    - For each of the N radial zones, Peak/Valley Residence is calculated as the proportion of profile
      points that fall within that horizontal distance band. It indicates how much of the drawn line
      spends time at a given amplitude, revealing whether the vase hugs the centreline or swings wide.
- Amplitude Bin N Vertical Span (amp_bin_N_vertical_span)
    - For each radial zone, Vertical Span is the difference between the maximum and minimum Y-coordinates
      of all points inside that amplitude band. It measures the height range traversed while the vase wall
      stays within a specific horizontal distance from the centreline.
- Amplitude Bin N Iso Crossing Count (amp_bin_N_iso_crossing_count)
    - For each radial zone, Iso Crossing Count counts the number of times the profile crosses the lower
      boundary of that amplitude band by detecting sign changes in the shifted X-coordinate array.
      It quantifies how many times the curve oscillates in and out of that horizontal zone.
- Amplitude Bin N Mean Curvature (amp_bin_N_mean_curvature)
    - For each radial zone, Mean Curvature is the average signed curvature of all profile points inside
      that amplitude band. It reveals how sharply the vase bends whilst occupying a specific horizontal range.
- Amplitude Bin N Mae (amp_bin_N_mae)
    - For each radial zone, Mae is the mean absolute deviation of all points inside that amplitude band
      from the target S-curve. It localises the drawing error to specific horizontal distances, isolating
      whether mistakes happen near the centreline or at the widest extents.
- Amplitude Discontinuity
    - Amplitude Discontinuity counts the total number of radial zones that contain zero profile points.
      It detects topological gaps where the vase wall completely skips a horizontal distance range,
      indicating an unusual or abrupt profile shape.
'''

import numpy as np

def extract_amplitude_binning(base, n_bins=5):
    '''
    Extracts amplitude-conditioned features by binning the X-axis.
    Parameters:
        base (dict): A dictionary containing foundational calculus arrays (x_v, y, k_v, abs_err).
        n_bins (int): The number of radial zones to divide the amplitude into.
    Returns:
        dict: A dictionary containing the rigorously extracted amplitude features.
    '''
    f = {}
    x_v = base['x_v']
    y = base['y']
    k_v = base['k_v']
    abs_err = base['abs_err']
    
    # -----------------------------------------
    # RADIAL ZONE (AMPLITUDE) DEFINITION
    # -----------------------------------------
    # Define strict amplitude boundaries based on the vase's absolute physical width limits
    min_x, max_x = np.min(x_v), np.max(x_v)
    
    # Prevents zero-division or binning collapse on a perfectly straight vertical line
    if max_x == min_x:
        max_x = min_x + 1e-8 
        
    bin_edges = np.linspace(min_x, max_x, n_bins + 1)
    empty_bins = 0
    
    # -----------------------------------------
    # CATEGORY: AMPLITUDE BINNING METRICS
    # -----------------------------------------
    for i in range(n_bins):
        # 1. Mask for the current radial zone
        # The upper bound is inclusive only for the absolute final bin to prevent coordinate dropping
        if i == n_bins - 1:
            mask = (x_v >= bin_edges[i]) & (x_v <= bin_edges[i+1])
        else:
            mask = (x_v >= bin_edges[i]) & (x_v < bin_edges[i+1])
            
        prefix = f'amp_bin_{i+1}_'
        
        # 2. Handle Topological Discontinuities (Empty Bins)
        if not np.any(mask):
            f[prefix + 'peak_valley_residence'] = 0.0
            f[prefix + 'vertical_span'] = 0.0
            f[prefix + 'iso_crossing_count'] = 0.0
            f[prefix + 'mean_curvature'] = 0.0
            f[prefix + 'mae'] = 0.0
            empty_bins += 1
            continue
            
        # 3. Peak/Valley Residence
        # Percentage of the 1D line existing at this specific horizontal distance
        f[prefix + 'peak_valley_residence'] = np.sum(mask) / len(x_v)
        
        # 4. Vertical Span of Amplitude
        # The vertical Y-axis distance covered whilst staying inside this X-zone
        f[prefix + 'vertical_span'] = np.max(y[mask]) - np.min(y[mask])
        
        # 5. Iso-Amplitude Crossing Count
        # Robust topological crossing detection evaluating sign changes across the lower bin boundary
        shifted_x = x_v - bin_edges[i]
        # Remove perfect zeroes to prevent false double-counts on exact boundary hits
        non_zero_shifted = shifted_x[shifted_x != 0]
        crossings = np.sum(np.diff(np.sign(non_zero_shifted)) != 0) if len(non_zero_shifted) > 1 else 0
        f[prefix + 'iso_crossing_count'] = crossings
        
        # 6. Amplitude-Conditioned Curvature
        # Average bending within this strictly defined radial zone
        f[prefix + 'mean_curvature'] = np.mean(k_v[mask])
        
        # 7. Amplitude-Conditioned Error
        # Average absolute deviation from the target S-curve inside this zone
        f[prefix + 'mae'] = np.mean(abs_err[mask])
        
    # 8. Amplitude Discontinuity
    # Total count of horizontal zones completely skipped by the 1D line
    f['amplitude_discontinuity'] = empty_bins
    
    return f