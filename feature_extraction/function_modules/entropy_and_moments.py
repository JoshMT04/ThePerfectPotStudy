'''
Python file for computing Complexity & Entropy and Statistical Moments features.
This module translates the spatial coordinates into information-theoretic vocabularies,
spectral frequencies, and thermodynamic entropy to measure the structural unpredictability of the vase.
Curvature arrays are converted into a ternary alphabet (Convex, Flat, Concave) to compute Lempel-Ziv
complexity, which quantifies the number of unique sub-patterns in the curvature profile. This includes:
- Curvature Shannon Entropy
    - Curvature Shannon Entropy is calculated by binning the curvature distribution into a 20-bin histogram
      and computing the Shannon entropy of the resulting probability mass. It measures the informational
      unpredictability of the bending pattern, with higher values indicating more varied curvature.
- Approximate Entropy
    - Approximate Entropy (ApEn) measures the logarithmic likelihood that similar windows of curvature
      values remain similar at the next step, using a template length of 2 and a tolerance of 0.2 standard
      deviations. It quantifies the regularity and predictability of the curvature sequence.
- Gzip Complexity Ratio
    - Gzip Complexity Ratio is calculated by compressing the normalised, quantised X-coordinate array with
      GZIP at maximum level and dividing the compressed byte size by the original byte size. It approximates
      the Kolmogorov complexity of the profile shape, with values closer to 1 indicating higher structural
      complexity.
- Gzip Compressed Size
    - Gzip Compressed Size is the raw byte length of the GZIP-compressed normalised X-coordinate array.
      It provides an absolute measure of the algorithmic information content of the profile independent of
      the original array length.
- Lempel Ziv Complexity
    - Lempel-Ziv Complexity counts the number of distinct non-repeating sub-patterns in the ternary curvature
      sequence (Concave=0, Flat=1, Convex=2) using the LZ76 algorithm. It measures the structural intricacy
      and design variability of the vase's bending profile.
- Fractal Dimension
    - Fractal Dimension is estimated via 1D box-counting on the normalised (x, y) profile curve across ten
      logarithmically spaced grid scales, fitting a log-log regression to extract the scaling exponent.
      Values above 1 indicate a curve that fills space more than a straight line.
- Low Freq Fft Energy
    - Low Freq Fft Energy is the sum of squared magnitudes of the 1st through 3rd harmonics of the centroid
      distance signature's FFT. It captures the energy concentrated in large, global macro-shape oscillations.
- High Freq Fft Energy
    - High Freq Fft Energy is the sum of squared magnitudes of the highest 15 frequency bins of the centroid
      distance signature's FFT. It captures the energy in microscopic, high-frequency surface noise or detail.
- Low Order Fourier Magnitude
    - Low Order Fourier Magnitude is the sum of the FFT magnitudes at the 2nd and 3rd harmonics of the
      centroid distance signature. It measures the contribution of the dominant periodic shape components
      beyond the fundamental frequency.
- Contour Autocorrelation Decay
    - Contour Autocorrelation Decay records the lag index at which the normalised curvature autocorrelation
      first drops below 0.5. It measures how quickly the vase "forgets" its current bending state, with
      lower values indicating a rapidly changing, less self-similar curvature profile.
- Contour Tortuosity
    - Contour Tortuosity is calculated as (perimeter / Euclidean end-to-end distance) - 1. It measures how
      much longer the actual contour path is compared to a straight line between its endpoints, quantifying
      the overall windiness of the profile.
- Hu Moment 1 through Hu Moment 7
    - The seven Hu Moments are computed from the OpenCV moments of the full mirrored symmetric contour and
      log-transformed as -sign(val) * log10(|val|) to prevent floating-point underflow. Each moment is
      invariant to scale, rotation, and translation, together encoding the global spatial mass distribution
      of the vase shape.
- Curvature Skewness
    - Curvature Skewness is the third standardised moment of the curvature distribution. It measures whether
      the bending is asymmetrically concentrated towards convex or concave regions, with positive values
      indicating a tail of sharp convex peaks.
- Zernike Moment Amplitude
    - Zernike Moment Amplitude is the absolute magnitude of the Z(2,2) Zernike moment computed by projecting
      the rasterised symmetric vase shape onto a unit disk. It measures the degree of structural elongation
      and directional orientation of the shape's mass distribution.
- Zernike Moment Phase
    - Zernike Moment Phase is the complex argument (angle) of the Z(2,2) Zernike moment. It encodes the
      dominant axis of elongation of the vase shape within the unit disk projection.
'''

import gzip
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
    
    # 3. GZIP Kolmogorov Complexity Estimate
    # Normalise x to [0, 1] first so the measure captures structural complexity
    # rather than scale. Quantise to uint16 (65536 levels) to match the precision
    # used in the original R implementation (gzip on a numeric vector).
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x_bytes = (x_norm * 65535).astype(np.uint16).tobytes()
    compressed = gzip.compress(x_bytes, compresslevel=9)
    f['gzip_complexity_ratio'] = len(compressed) / len(x_bytes)
    f['gzip_compressed_size']  = len(compressed)

    # 4. Lempel-Ziv Complexity
    # Discretise curvature into a Ternary Alphabet (Concave, Flat, Convex)
    noise_threshold = 1e-3
    ternary_k = np.zeros_like(k, dtype=int)
    ternary_k[k > noise_threshold] = 2  # Convex
    ternary_k[k < -noise_threshold] = 0 # Concave
    ternary_k[(k >= -noise_threshold) & (k <= noise_threshold)] = 1 # Flat
    f['lempel_ziv_complexity'] = _lempel_ziv_complexity(ternary_k)
    
    # 5. Fractal Dimension (1D Box-Counting on the profile curve)
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
    
    # 6. Spectral FFT Analysis (Frequency Domain)
    # Convert 2D coordinates into a 1D Centroid Distance Signature
    cx, cy = np.mean(x), np.mean(y)
    dist_sig = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    freq_data = np.abs(fft(dist_sig))
    half_n = len(freq_data) // 2
    
    f['low_freq_fft_energy'] = np.sum(freq_data[1:4]**2) # Massive, global macro-shapes
    f['high_freq_fft_energy'] = np.sum(freq_data[half_n-15:half_n]**2) # Microscopic surface noise
    f['low_order_fourier_magnitude'] = freq_data[2] + freq_data[3] # Magnitude of 2nd and 3rd harmonics
    
    # 7. Contour Autocorrelation Decay
    # Measures how fast the vase "forgets" its current curvature
    k_centred = k - np.mean(k)
    autocorr = np.correlate(k_centred, k_centred, mode='full')
    autocorr = autocorr[autocorr.size // 2:] # Take only positive lags
    autocorr /= autocorr[0] if autocorr[0] > 0 else 1
    # Find the lag where autocorrelation first drops below 0.5
    decay_idx = np.where(autocorr < 0.5)[0]
    f['contour_autocorrelation_decay'] = decay_idx[0] if len(decay_idx) > 0 else len(k)
    
    # 8. Contour Tortuosity
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