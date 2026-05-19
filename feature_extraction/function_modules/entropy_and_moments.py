'''
Python file for computing Complexity & Entropy and Statistical Moments features.
This module translates the spatial coordinates into information-theoretic vocabularies, 
spectral frequencies, and thermodynamic entropy to measure the structural unpredictability of the vase.
It should be noted that curvature arrays are converted into a ternary alphabet (Convex, Flat, Concave) to 
compute the Lempel-Ziv complexity, which quantifies the number of unique sub-patterns in the curvature profile, 
reflecting the vase's structural intricacy and design variability.
'''

import numpy as np
import cv2
from scipy.stats import entropy, skew
from scipy.fft import fft

def _approximate_entropy(U, m=2, r=None):
    '''
    Calculates the Approximate Entropy (ApEn) of a sequence.
    Measures the logarithmic likelihood that similar patterns remain similar.
    '''
    if r is None:
        r = 0.2 * np.std(U)
        
    def _phi(m):
        N = len(U)
        x = np.array([U[i:i+m] for i in range(N - m + 1)])
        # Calculate maximum absolute distance between all pairs of windows
        C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)
        
    return np.abs(_phi(m) - _phi(m + 1))

def _lempel_ziv_complexity(binary_sequence):
    '''
    Calculates the LZ76 complexity by counting the number of distinct 
    non-repeating sub-patterns in a discretised sequence.
    '''
    s = "".join(map(str, binary_sequence))
    i, k, complexity = 0, 1, 1
    n = len(s)
    while True:
        if i + k > n:
            break
        if s[i:i+k] in s[0:i+k-1]:
            k += 1
        else:
            complexity += 1
            i += k
            k = 1
    return complexity

def extract_entropy_and_moments(base):
    '''
    Extracts algorithmic complexity, spectral frequencies, and statistical moments.
    Parameters:
        base (dict): A dictionary containing foundational calculus arrays (x, y, k, perimeter).
    Returns:
        dict: A dictionary containing 18 rigorously extracted complexity and moment features.
    '''
    f = {}
    x, y, k, p = base['x'], base['y'], base['k'], base['perimeter']
    
    # -----------------------------------------
    # CATEGORY 5: COMPLEXITY & ENTROPY
    # -----------------------------------------
    
    # 1. Curvature Shannon Entropy
    hist, _ = np.histogram(k, bins=20, density=True)
    hist_prob = hist[hist > 0] / np.sum(hist[hist > 0])
    f['curvature_shannon_entropy'] = entropy(hist_prob, base=2)
    
    # 2. Approximate Entropy (ApEn)
    f['approximate_entropy'] = _approximate_entropy(k)
    
    # 3. Lempel-Ziv Complexity
    # Discretise curvature into a Ternary Alphabet (Concave, Flat, Convex)
    noise_threshold = 1e-3
    ternary_k = np.zeros_like(k, dtype=int)
    ternary_k[k > noise_threshold] = 2  # Convex
    ternary_k[k < -noise_threshold] = 0 # Concave
    ternary_k[(k >= -noise_threshold) & (k <= noise_threshold)] = 1 # Flat
    f['lempel_ziv_complexity'] = _lempel_ziv_complexity(ternary_k)
    
    # 4. Fractal Dimension (1D Box-Counting on the profile curve)
    # We normalise the x and y axes to ensure the grid scales symmetrically
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    
    def box_count(eps):
        xbins = np.floor(x_norm / eps)
        ybins = np.floor(y_norm / eps)
        hashed_coords = xbins * 100000 + ybins # Unique hash for each grid cell
        return len(np.unique(hashed_coords))
    
    epsilons = np.logspace(-2.5, -0.5, 10)
    counts = [box_count(eps) for eps in epsilons]
    coeffs = np.polyfit(np.log(1/epsilons), np.log(counts), 1)
    f['fractal_dimension'] = coeffs[0]
    
    # 5. Spectral FFT Analysis (Frequency Domain)
    # Convert 2D coordinates into a 1D Centroid Distance Signature
    cx, cy = np.mean(x), np.mean(y)
    dist_sig = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    freq_data = np.abs(fft(dist_sig))
    half_n = len(freq_data) // 2
    
    f['low_freq_fft_energy'] = np.sum(freq_data[1:4]**2) # Massive, global macro-shapes
    f['high_freq_fft_energy'] = np.sum(freq_data[half_n-15:half_n]**2) # Microscopic surface noise
    f['low_order_fourier_magnitude'] = freq_data[2] + freq_data[3] # Magnitude of 2nd and 3rd harmonics
    
    # 6. Contour Autocorrelation Decay
    # Measures how fast the vase "forgets" its current curvature
    k_centred = k - np.mean(k)
    autocorr = np.correlate(k_centred, k_centred, mode='full')
    autocorr = autocorr[autocorr.size // 2:] # Take only positive lags
    autocorr /= autocorr[0] if autocorr[0] > 0 else 1
    # Find the lag where autocorrelation first drops below 0.5
    decay_idx = np.where(autocorr < 0.5)[0]
    f['contour_autocorrelation_decay'] = decay_idx[0] if len(decay_idx) > 0 else len(k)
    
    # 7. Contour Tortuosity
    euclidean_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    f['contour_tortuosity'] = (p / euclidean_dist) - 1 if euclidean_dist > 0 else 0

    # -----------------------------------------
    # CATEGORY 6: STATISTICAL SHAPE MOMENTS
    # -----------------------------------------
    
    # Hu Moments (Scale, Rotation, and Translation Invariants)
    # We mirror the right-hand profile to calculate accurate spatial mass invariants
    full_x = np.concatenate([x, -x[::-1]])
    full_y = np.concatenate([y, y[::-1]])
    
    # Shift to positive integer space for cv2.moments processing
    full_x_shifted = full_x - np.min(full_x)
    full_y_shifted = full_y - np.min(full_y)
    contour_pts = np.column_stack((full_x_shifted, full_y_shifted)).astype(np.float32)
    
    moments = cv2.moments(contour_pts)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    for i in range(7):
        # Logarithmic transform required to prevent floating-point underflow
        val = hu_moments[i]
        f[f'hu_moment_{i+1}'] = -np.sign(val) * np.log10(np.abs(val)) if val != 0 else 0
        
    f['curvature_skewness'] = skew(k)
    
    # -----------------------------------------
    # ZERNIKE MOMENTS (Radial Mass Distribution)
    # -----------------------------------------
    # Zernike moments require projecting the shape onto a 2D unit disk.
    # We must rasterise the continuous symmetric coordinates into a binary pixel grid.
    
    grid_size = 200
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Scale full symmetric coordinates to fit strictly inside the grid (with a 10px margin)
    x_scaled = (full_x_shifted / (np.max(full_x_shifted) + 1e-8) * (grid_size - 20)) + 10
    y_scaled = (full_y_shifted / (np.max(full_y_shifted) + 1e-8) * (grid_size - 20)) + 10
    
    poly_pts = np.column_stack((x_scaled, y_scaled)).astype(np.int32)
    cv2.fillPoly(mask, [poly_pts], 1)
    
    # Create the coordinate space for the Unit Disk (-1.0 to 1.0)
    Y_grid, X_grid = np.indices(mask.shape)
    Y_grid = (Y_grid - grid_size/2) / (grid_size/2)
    X_grid = (X_grid - grid_size/2) / (grid_size/2)
    
    # Convert Cartesian to Polar (Radius and Theta)
    R = np.sqrt(X_grid**2 + Y_grid**2)
    Theta = np.arctan2(Y_grid, X_grid)
    
    # Mask out everything outside the unit disk, keeping only the filled clay
    disk_mask = (R <= 1.0) & (mask == 1)
    
    # Zernike Polynomial Z_2,2 (Measures structural elongation and orientation)
    # The radial polynomial is R_2,2(r) = r^2
    # The complex angular component is e^(-i * 2 * theta)
    V_2_2_real = R[disk_mask]**2 * np.cos(-2 * Theta[disk_mask])
    V_2_2_imag = R[disk_mask]**2 * np.sin(-2 * Theta[disk_mask])
    
    # Integrate over the unit disk (multiply by normalization factor n+1 / pi)
    z_2_2_complex = complex(np.sum(V_2_2_real), np.sum(V_2_2_imag)) * (3 / np.pi)
    
    # Extract the absolute Amplitude and the directional Phase
    f['zernike_moment_amplitude'] = np.abs(z_2_2_complex)
    f['zernike_moment_phase'] = np.angle(z_2_2_complex)

    return f