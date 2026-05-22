# Feature Extraction

This directory extracts geometric, kinematic, and information-theoretic features from vase half-profiles. Each profile is a 125-point (x, y) coordinate sequence representing the right-hand silhouette of a vase from base to lip.

Features fall into two categories:
- **Intrinsic** — properties of the vase shape alone (proportions, curvature, complexity, etc.)
- **Relational** — comparisons against reference S-curves from Hübner et al. (2023), measuring how far the vase deviates from an idealised form

---

## Pipeline

| File | Role |
|---|---|
| `extraction_run.ipynb` | End-to-end notebook: runs extraction, removes collinear features, exports matrices |
| `function_modules/main_extractor.py` | Orchestrates parallel extraction across all vases; handles batching, checkpointing, and resume |
| `function_modules/main_curve_matcher.py` | Builds the comparative base state between a vase and each target S-curve, then calls all four relational modules |
| `hubner_lines_0326/` | Reference S-curve profiles (Hübner et al. 2023) used as relational targets |

---

## Intrinsic Feature Modules

These modules operate on a single vase profile with no reference curve.

### `core_engine.py`
Computes foundational calculus arrays reused by all other modules. Not a feature module itself — it produces the `base` dict containing:
- Raw x, y coordinates and perimeter
- 1st and 2nd derivatives (dx, dy, ddx, ddy)
- Signed curvature array `k`
- 2D centroid (cx, cy)

---

### `proportions_and_mass.py`
**Category: Macro-proportions, Mass & Volume, Physical Affordance, Structural Vulnerability, Lifecycle & Fluid Dynamics**

Geometrically characterises the overall shape, volumetric distribution, and functional properties of the vase.

| Feature | Description |
|---|---|
| `aspect_ratio` | Height / (2 × max width). Higher = taller shape. |
| `compactness` | (4π × area) / perimeter². Closeness to a perfect circle. |
| `max_width_height` | Normalised y-position of the widest point. 0 = wide base, 1 = wide top. |
| `neck_to_base_ratio` | Minimum top-half width / base width. Structural tapering. |
| `lip_to_belly_ratio` | Lip width / max width. Whether the opening is closed or flared. |
| `bounding_box_solidity` | Area / bounding box area. How much of its own space the vase fills. |
| `max_calliper_diameter` | Longest physical dimension across the mirrored profile. |
| `base_stability_index` | Base width / max width. Physical groundedness vs. top-heaviness. |
| `shoulder_slope_angle` | Arctangent of the neck-to-belly transition. Aggressiveness of the shoulder. |
| `perimeter_convexity` | Convex hull perimeter / true perimeter. How much the contour meanders inward. |
| `convex_hull_deficit` | 1 − (area / convex hull area). Degree of concavity. |
| `enclosing_circle_deficit` | 1 − (area / minimum enclosing circle area). Packing efficiency. |
| `rotational_volume` | Theoretical 3D volume from revolving the profile around its axis. |
| `surface_to_volume_ratio` | 3D surface area / rotational volume. Physical efficiency of the shape. |
| `y_centroid_normalised` | 2D centroid height normalised by total height. Mass distribution. |
| `3d_centre_of_mass_vertical` | Volumetric centre of mass height, normalised. |
| `top_to_bottom_area_ratio` | Area above midpoint / area below midpoint. |
| `eccentricity` | Ellipse fit eccentricity. How naturally stretched the geometry is. |
| `mean_centroid_distance` | Mean distance from all profile points to the 2D centroid. |
| `base_to_centroid_height_ratio` | Absolute height from base to 2D centroid. |
| `centroid_to_top_height_ratio` | Absolute height from 2D centroid to lip. |
| `critical_tipping_angle` | Angle at which the vase would topple, based on 3D CoM and base width. |
| `evaporation_exposure_quotient` | Lip opening area / rotational volume. |
| `ergonomic_grip_security` | Maximum concavity in the upper half — strength of any hand-hold. |
| `stem_support_index` | (Max width − neck width) / max width. Degree of narrowing above the belly. |
| `material_efficiency_ratio` | Rotational volume / profile arc length. Volume gained per unit of clay. |
| `cantilever_snap_stress` | Torque above the narrowest point divided by neck cross-section. Decapitation risk. |
| `overhang_slump_risk` | Proportion of profile where outward slope exceeds 45°. Kiln collapse risk. |
| `protrusion_chip_vulnerability` | Peak convex curvature. Sharpest outward protrusion. |
| `foot_leverage_fracture_risk` | (Max width / base width) × height to max width. Lever-arm fracture risk at the foot. |
| `cleaning_accessibility_index` | Total depth / neck width. How hard the interior is to clean. |
| `fluid_evacuation_constraint` | Rotational volume / neck cross-sectional area. Pouring bottleneck. |
| `dynamic_ballast_distribution` | Bottom-half volume / top-half volume. Water weight distribution. |
| `tabletop_real_estate_efficiency` | Rotational volume / base footprint area. |

---

### `kinematics.py`
**Category: Curvature & Kinematics**

Evaluates the 1st, 2nd, and 3rd spatial derivatives — bending energy, inflection structure, and contour jerk.

| Feature | Description |
|---|---|
| `total_bending_energy` | Sum of squared curvatures. Total integrated bending effort. |
| `maximum_local_curvature` | Peak absolute curvature anywhere on the profile. |
| `inflection_count` | Number of curvature sign changes (concave ↔ convex transitions). |
| `zero_crossing_rate` | Inflection count / profile length. |
| `mean_convex_curvature` | Mean curvature over outward-bending regions only. |
| `mean_concave_curvature` | Mean curvature over inward-bending regions only. |
| `convex_concave_energy_ratio` | Ratio of squared convex to squared concave curvature energy. |
| `mean_absolute_curvature` | Mean of \|k\| across the full profile. |
| `median_curvature` | Median signed curvature. |
| `curvature_variance` | Variance of the curvature distribution. |
| `curvature_range` | Max − min curvature. Total span of bending. |
| `curvature_iqr` | Interquartile range of curvature. Robust spread. |
| `median_k` | Median absolute curvature (duplicate of median for pipeline compatibility). |
| `curvature_median_abs_deviation` | Median absolute deviation of curvature from its median. |
| `mean_abs_k` | Mean absolute curvature (alias). |
| `curvature_coef_of_variation` | Std / mean curvature. Normalised variability. |
| `curvature_kurtosis` | Kurtosis of the curvature distribution. Peakedness of bending. |
| `curvature_smoothness_index` | 1 / std(Δk). Higher = smoother transitions between bends. |
| `local_maxima_density` | Number of curvature peaks / perimeter. How densely features are packed. |
| `mean_absolute_jerk` | Mean rate of change of curvature with arc length. |
| `jerk_variance` | Variance of jerk. Consistency of curvature transitions. |

---

### `entropy_and_moments.py`
**Category: Complexity & Entropy, Statistical Shape Moments**

Translates the profile into information-theoretic and frequency-domain vocabularies, and computes shape invariants.

| Feature | Description |
|---|---|
| `curvature_shannon_entropy` | Shannon entropy of the curvature histogram. Unpredictability of bending distribution. |
| `approximate_entropy` | ApEn of the curvature sequence. Likelihood that similar curvature patterns recur. |
| `gzip_complexity_ratio` | Compressed bytes / original bytes of the normalised x sequence. Kolmogorov complexity estimate — lower = more regular profile. |
| `gzip_compressed_size` | Raw GZIP-compressed byte count of the x sequence. |
| `lempel_ziv_complexity` | LZ76 complexity of the ternary-discretised curvature (Concave / Flat / Convex). Number of distinct sub-patterns. |
| `fractal_dimension` | 1D box-counting fractal dimension of the profile curve. |
| `low_freq_fft_energy` | Energy in FFT harmonics 1–3 of the centroid-distance signature. Global macro-shape energy. |
| `high_freq_fft_energy` | Energy in the top 15 FFT harmonics. Microscopic surface texture energy. |
| `low_order_fourier_magnitude` | Sum of 2nd and 3rd harmonic magnitudes. |
| `contour_autocorrelation_decay` | Lag at which curvature autocorrelation first drops below 0.5. How quickly the profile "forgets" its current bend. |
| `contour_tortuosity` | (Perimeter / straight-line length) − 1. Path windiness. |
| `hu_moment_1` … `hu_moment_7` | Log-transformed Hu moments. Scale-, rotation-, and translation-invariant shape descriptors. |
| `curvature_skewness` | Skewness of the curvature distribution. Asymmetry of bending. |
| `zernike_moment_amplitude` | Amplitude of the Z₂,₂ Zernike moment. Structural elongation. |
| `zernike_moment_phase` | Phase of the Z₂,₂ Zernike moment. Directional orientation of elongation. |

---

### `typological_skeleton.py`
**Category: Typological / Skeleton**

Isolates anatomical landmarks (Foot, Belly, Neck, Lip) and measures the physical transition states between them.

| Feature | Description |
|---|---|
| `shoulder_prominence` | Maximum gradient at the belly-to-neck transition. Aggressiveness of the shoulder. |
| `foot_definition` | Maximum absolute curvature in the bottom 5% of the profile. Sharpness of the foot. |
| `neck_elongation` | Vertical extent of the region near minimum width in the upper half. How elongated the neck is. |
| `belly_to_neck_distance` | Absolute vertical distance from the widest point to the narrowest upper point. |
| `medial_axis_length` | Total profile height (y_max − y_min). Skeleton backbone length. |
| `lip_flare_angle` | Angular difference between the tangent directions at the lip and neck. Degree of lip flare. |
| `osculating_transition_radius` | Radius of curvature at the sharpest transition point. Tightness of the most extreme bend. |
| `inflection_point_density` | Number of curvature sign changes per unit height. |

---

### `vertical_asymmetry_and_binning.py`
**Category: Vertical Asymmetry, Distributional Binning**

Slices the profile into vertical floors to measure volumetric imbalance and evaluates the statistical distribution of widths.

| Feature | Description |
|---|---|
| `vertical_area_asymmetry` | Absolute difference between top-half and bottom-half 2D areas. |
| `midpoint_y_deviation` | Distance between the 2D centroid and the geometric vertical midpoint. |
| `vertical_bending_ratio` | Sum of squared curvatures in top half / bottom half. Where bending energy is concentrated. |
| `vertical_hausdorff_distance` | Hausdorff distance between the top and bottom half x-profiles. Structural self-dissimilarity. |
| `vertical_contour_length_ratio` | Arc length of top half / bottom half. |
| `width_interquartile_range` | IQR of x-values across the full profile. Spread of widths. |
| `radial_dist_interquartile_range` | IQR of distances from each profile point to the centroid. |
| `extreme_convexity_proportion` | Proportion of points with curvature above the 90th percentile. |
| `extreme_concavity_proportion` | Proportion of points with curvature below the 10th percentile. |
| `flatness_proportion` | Proportion of points with near-zero curvature (\|k\| < 0.001). |
| `bottom_third_area_mass` | Area of the bottom third / total area. |
| `mid_third_area_mass` | Area of the middle third / total area. |
| `top_third_area_mass` | Area of the top third / total area. |
| `upper_half_average_width` | Mean x-value in the upper half of the profile. |
| `lower_half_average_width` | Mean x-value in the lower half of the profile. |
| `y_axis_width_variance` | Variance of x across the full profile height. |

---

## Relational Feature Modules

These modules compare each vase against each of the Hübner et al. S-curve targets. Features are computed once per target curve, producing a separate set of columns for each (e.g. `SCurve_4_rmse`, `SCurve_5_rmse`).

### `spatial_error_and_flexibility.py`
**Category: Strict Spatial Error & Topological Flexibility**

Measures physical deviation between the vase x-profile and the target S-curve at each height level.

| Feature | Description |
|---|---|
| `rmse` | Root mean squared error of x-deviations from the target. |
| `mae` | Mean absolute error of x-deviations. |
| `chebyshev_max_error` | Maximum single-point absolute deviation (worst-case error). |
| `open_procrustes_distance` | Procrustes distance after centring both curves. Scale-free shape mismatch. |
| `area_between_curves` | Area enclosed between the vase and target profiles. |
| `curve_length_ratio` | Vase arc length / target arc length. Relative perimeter scaling. |
| `intersection_count` | Number of times the vase and target profiles cross each other. |
| `hausdorff_distance` | Maximum of the directed Hausdorff distances in both directions. |
| `spatial_dtw_distance` | Dynamic time warping distance on the spatial profiles. |
| `modified_hausdorff_distance` | Mean of directed Hausdorff distances. Less sensitive to outliers. |
| `longest_common_subsequence_ratio` | LCSS ratio under a tolerance threshold. Proportion of structurally shared sequence. |
| `elastic_shape_distance_srvf` | SRV-framework elastic distance (phase-invariant bending + stretching cost). |
| `earth_movers_distance` | Wasserstein-1 distance between x-value distributions. |
| `qq_deviation` | Sum of squared differences between sorted x-values. Distributional shape mismatch. |
| `jensen_shannon_divergence` | JS divergence² between the x-value histograms. |
| `area_error_ratio` | Area between curves / target area. Normalised enclosed deviation. |
| `signed_area_bias` | Signed integral of (vase − target). Systematic over/under-width bias. |
| `max_curvature_spatial_deviation` | Euclidean distance between the points of maximum curvature on each curve. |
| `contour_phase_angle` | Angular difference between the dominant FFT phase of each profile. |
| `frechet_distance` | Discrete Fréchet distance. Minimum leash length needed to walk both curves simultaneously. |

---

### `derivatives_and_distributions.py`
**Category: Derivative Alignment**

Bypasses raw coordinates to compare the structural flow (slopes, angles, bending) between the vase and target.

| Feature | Description |
|---|---|
| `turning_angle_distance` | Integrated absolute difference in tangent angles along the profile height. |
| `derivative_dtw_distance` | DTW distance on the first-derivative sequences. Flow-level shape mismatch. |
| `max_curvature_cross_correlation` | Peak of the normalised cross-correlation between curvature sequences. Maximum structural phase alignment. |

---

### `spatial_y_binning.py`
**Category: Spatial Y-Binning (PAA)**

Slices the y-axis into 5 equal-height bins and measures localised error in each zone.

| Feature | Description |
|---|---|
| `binned_mae_upper_quartile` | MAE of x-deviations in the top 25% of the profile height. |
| `binned_mae_lower_quartile` | MAE of x-deviations in the bottom 25% of the profile height. |
| `bin_{1..5}_mae` | MAE within each of the 5 vertical bins (bottom to top). |
| `bin_{1..5}_signed_error` | Mean signed error within each bin. Reveals systematic over/under-width bias per zone. |
| `bin_{1..5}_peak_deviation` | Maximum absolute deviation within each bin. |
| `bin_{1..5}_curvature_alignment` | Mean absolute difference in curvature within each bin. |
| `bin_{1..5}_width_ratio` | Mean vase width / mean target width within each bin. |
| `point_of_max_divergence_bin_id` | Index (1–5) of the bin with the greatest MAE. Where the vase deviates most. |

---

### `amplitude_binning.py`
**Category: Amplitude (Horizontal) Binning**

Slices the x-axis (radial amplitude) into 5 equal-width zones and maps where spatial error and curvature occur relative to the centreline.

| Feature | Description |
|---|---|
| `amp_bin_{1..5}_peak_residence` | Proportion of profile points whose x-value falls in each amplitude bin. |
| `amp_bin_{1..5}_valley_residence` | Proportion of concave curvature points in each amplitude zone. |
| `amp_bin_{1..5}_vertical_span` | Vertical height range covered by each amplitude zone. |
| `amp_bin_{1..5}_iso_crossing_count` | Number of times the profile enters/exits each amplitude zone. |
| `amp_bin_{1..5}_mean_curvature` | Mean curvature of points within each amplitude zone. |
| `amp_bin_{1..5}_mae` | Mean absolute x-error (vs. target) for points in each amplitude zone. |
| `amplitude_discontinuity` | Number of amplitude bins that contain zero profile points. Gaps in radial coverage. |

---

### `srv_elastic_distance.py`
**Category: Elastic Shape Analysis**

Computes the true elastic deformation cost using the Square Root Velocity (SRV) framework, invariant to reparametrisation.

| Feature | Description |
|---|---|
| `srv_elastic_shape_distance` | SRV elastic distance between the vase and target profiles. The minimum bending + stretching energy required to deform one into the other, independent of speed of traversal. |

---

## Outputs

| File | Contents |
|---|---|
| `outdir/master_feature_matrix_compiled.csv` | Full extracted matrix — all vases × all features, before any filtering |
| `outdir/master_feature_matrix_compiled.parquet` | Parquet version of the above |
| `outdir/master_feature_matrix_all_features.parquet` | All features before correlation-based champion selection |
| `outdir/legacy_master_feature_matrix_purged.csv` | Reduced matrix after collinearity pruning (one champion feature per cluster) |
| `plots/` | Dendrogram, collinearity network, and top-feature bar chart |
